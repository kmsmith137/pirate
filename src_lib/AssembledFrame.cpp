#include "../include/pirate/AssembledFrame.hpp"
#include "../include/pirate/SlabAllocator.hpp"
#include "../include/pirate/XEngineMetadata.hpp"

#include "../include/pirate/file_utils.hpp"   // FileDeleteGuard
#include "../include/pirate/GpuDequantizationKernel.hpp"  // ReferenceDequantizationKernel (dequantize())
#include "../include/pirate/inlines.hpp"      // xdiv(), align_up()
#include "../include/pirate/constants.hpp"    // bytes_per_gpu_cache_line
#include "../include/pirate/avx2_utils.hpp"   // avx2_simulate_4bit_noise(), avx2_4bit_postquant_noise_rms()
#include "../include/pirate/simpulse.hpp"     // simpulse::SinglePulse (pulse injection)

#include <ksgpu/xassert.hpp>
#include <ksgpu/mem_utils.hpp>
#include <ksgpu/rand_utils.hpp>    // rand_int()

#include <cuda_fp16.h>             // __half, __float2half_rn

#include <asdf/asdf.hxx>

#include <fcntl.h>     // open(), O_RDONLY
#include <unistd.h>    // fsync(), close()

#include <algorithm>   // std::min, std::max (pulse-injection clamp)
#include <cmath>       // std::sqrt, std::floor
#include <cstring>
#include <fstream>
#include <map>
#include <mutex>
#include <random>
#include <sstream>
#include <stdexcept>

using namespace std;
using namespace ksgpu;

namespace pirate {
#if 0
}  // editor auto-indent
#endif


// Call with lock held!
void AssembledFrame::_reap_locked()
{
    if (data.size == 0)
        return;  // already reaped

    // Unreapable while a write is pending (paths queued, not yet on SSD) --
    // UNLESS the write already failed. save_error is terminal: the data will
    // never reach disk (FileWriter's NFS thread only drains an errored
    // frame's entries as error notifications), so freeing it is safe. It is
    // also necessary: the FrbServer reaper visits each frame once (rb_reaped
    // is monotone), so an errored frame left pinned here would leak its slab
    // until the ringbuf slot is overwritten -- and under sustained SSD
    // failure the leaked slabs exhaust the pool and silently wedge the whole
    // pipeline (the allocator worker parks in get_slab, so rb_assembled
    // stalls and the max-unprocessed check never fires). The FileWriter SSD
    // worker calls _reap_locked() right after setting save_error, so an
    // errored frame is freed immediately.
    if (save_paths.size() && !on_ssd && !save_error)
        return;

    // scales_offsets and data share a slab via a common base shared_ptr;
    // dropping both Array<void>s drops the last refs and frees the slab.
    this->scales_offsets = Array<void> ();
    this->data           = Array<void> ();
}


// ASDF metadata (de)serialization helpers.
//
// XEngineMetadata is written field-by-field into a nested ASDF group with
// the keys mirroring the C++ member names. Four members are handled
// specially in AssembledFrame::{write,from}_asdf -- see XEngineMetadata.hpp
// for the rationale.

static shared_ptr<ASDF::sequence> _make_int_seq(const vector<long> &v)
{
    auto seq = make_shared<ASDF::sequence>();
    for (long x : v)
        seq->push_back(make_shared<ASDF::int_entry>(int64_t(x)));
    return seq;
}


static shared_ptr<ASDF::sequence> _make_float_seq(const vector<double> &v)
{
    auto seq = make_shared<ASDF::sequence>();
    for (double x : v)
        seq->push_back(make_shared<ASDF::float_entry>(x));
    return seq;
}


static shared_ptr<ASDF::sequence> _make_float_seq(const std::array<double, 3> &v)
{
    auto seq = make_shared<ASDF::sequence>();
    for (double x : v)
        seq->push_back(make_shared<ASDF::float_entry>(x));
    return seq;
}


static long _read_int(const shared_ptr<ASDF::group> &grp, const string &key)
{
    auto e = grp->at(key);
    auto v = e->get_maybe_int();
    if (!v.has_value())
        throw runtime_error("AssembledFrame::from_asdf(): expected int entry for '" + key + "'");
    return long(v.value());
}


// Accept either a float YAML scalar or an int YAML scalar: yaml-cpp's emitter
// strips the trailing ".0" from whole-number doubles, so a double like 400.0
// round-trips through ASDF as the int_entry `400`. Every int64 is exactly
// representable as a double, so the widening is lossless.
static double _read_float(const shared_ptr<ASDF::group> &grp, const string &key)
{
    auto e = grp->at(key);
    if (auto v = e->get_maybe_float(); v.has_value())
        return v.value();
    if (auto v = e->get_maybe_int(); v.has_value())
        return double(v.value());
    throw runtime_error("AssembledFrame::from_asdf(): expected float or int entry for '" + key + "'");
}


static vector<long> _read_int_vec(const shared_ptr<ASDF::group> &grp, const string &key)
{
    auto e = grp->at(key);
    auto seq = e->get_maybe_sequence();
    if (!seq)
        throw runtime_error("AssembledFrame::from_asdf(): expected sequence for '" + key + "'");
    vector<long> ret;
    ret.reserve(seq->size());
    for (const auto &elt : *seq) {
        auto v = elt->get_maybe_int();
        if (!v.has_value())
            throw runtime_error("AssembledFrame::from_asdf(): non-int element in sequence '" + key + "'");
        ret.push_back(long(v.value()));
    }
    return ret;
}


// See _read_float for the int-fallback rationale.
static vector<double> _read_float_vec(const shared_ptr<ASDF::group> &grp, const string &key)
{
    auto e = grp->at(key);
    auto seq = e->get_maybe_sequence();
    if (!seq)
        throw runtime_error("AssembledFrame::from_asdf(): expected sequence for '" + key + "'");
    vector<double> ret;
    ret.reserve(seq->size());
    for (const auto &elt : *seq) {
        if (auto v = elt->get_maybe_float(); v.has_value())
            ret.push_back(v.value());
        else if (auto v = elt->get_maybe_int(); v.has_value())
            ret.push_back(double(v.value()));
        else
            throw runtime_error("AssembledFrame::from_asdf(): non-numeric element in sequence '" + key + "'");
    }
    return ret;
}


static std::array<double, 3> _read_float_arr3(const shared_ptr<ASDF::group> &grp, const string &key)
{
    vector<double> v = _read_float_vec(grp, key);
    if (v.size() != 3) {
        stringstream ss;
        ss << "AssembledFrame::from_asdf(): expected length-3 sequence for '" << key
           << "', got length " << v.size();
        throw runtime_error(ss.str());
    }
    return { v[0], v[1], v[2] };
}


// Write the "xengine_metadata" sub-map directly to an ASDF::writer. Key
// order follows configs/xengine_metadata.yml. Emits via writer
// (not ASDF::group) so the order is preserved -- group::to_yaml is backed
// by std::map and would alphabetize keys. Skips freq_channels, beam_ids,
// and beam_positions_{x,y}; those are handled per-frame at the top level
// by AssembledFrame::write_asdf. m.freq_channels: IGNORED (skipped on emit).
//
// If verbose, emits explanatory comments. Section breakdown matches
// configs/xengine_metadata.yml; comments here are terse since
// the top-level write_asdf() comment block points readers at that file
// for field-by-field detail.
static void _emit_metadata_yaml(ASDF::writer &w, const XEngineMetadata &m, bool verbose)
{
    w << YAML::BeginMap;

    w << YAML::Key << "version" << YAML::Value << ASDF::int_entry(int64_t(m.version));

    if (verbose) {
        w << YAML::Newline << YAML::Newline << YAML::Comment(
            "Frequency zones. The observed frequency band is divided into\n"
            "zones; within a zone all channels have the same width, but width\n"
            "may differ between zones.\n"
            "  zone_nfreq:      number of channels in each zone.\n"
            "  zone_freq_edges: zone boundary frequencies in MHz; length (len(zone_nfreq) + 1)."
        ) << YAML::Newline << YAML::Newline;
    }
    w << YAML::Key << "zone_nfreq" << YAML::Value << *_make_int_seq(m.zone_nfreq);
    w << YAML::Key << "zone_freq_edges" << YAML::Value << *_make_float_seq(m.zone_freq_edges);

    if (verbose) {
        w << YAML::Newline << YAML::Newline << YAML::Comment(
            "integer identifier for this set of beams (sent by X-engine, opqaue to FRB search)"
        ) << YAML::Newline;
    }
    w << YAML::Key << "beamset" << YAML::Value << ASDF::int_entry(int64_t(m.beamset));

    if (verbose) {
        w << YAML::Newline << YAML::Newline << YAML::Comment(
            "Timekeeping: the X-engine uses an FPGA sequence number ('seq') to\n"
            "track time. To compute absolute time of an FRB time sample with\n"
            "index 't' (within a chunk) and given top-level time_chunk_index:\n"
            "  seq = (time_chunk_index * ntime + t) * seq_per_frb_time_sample\n"
            "  unix_ns = unix_ns_at_seq_0 + seq * dt_ns_per_seq"
        ) << YAML::Newline << YAML::Newline;
    }
    w << YAML::Key << "unix_ns_at_seq_0" << YAML::Value << ASDF::int_entry(int64_t(m.unix_ns_at_seq_0));
    w << YAML::Key << "dt_ns_per_seq" << YAML::Value << ASDF::int_entry(int64_t(m.dt_ns_per_seq));
    w << YAML::Key << "seq_per_frb_time_sample" << YAML::Value << ASDF::int_entry(int64_t(m.seq_per_frb_time_sample));

    if (verbose) {
        w << YAML::Newline << YAML::Newline << YAML::Comment(
            "Telescope alignment and localization.\n"
            "\n"
            "Coordinate Systems\n"
            "\n"
            "Topocentric: origin at the given lat/lon. x-axis is directed exactly East (increasing Longitude),\n"
            "  y-axis is directed exactly North (increasing Latitude), z-axis is directed exactly \"up\" (increasing\n"
            "  altitude). orthogonal, Z = X x Y.\n"
            "\n"
            "Grid: origin at SW corner of Dish Array, aligned with Dish Array.  x-axis directed \"east-ish\"\n"
            "  parallel to dish \"e/w\" separatation vector.  y-axis directed \"north-ish\" parallel to \"n/s\"\n"
            "  separation vector.  Z = X x Y points normal to grid plane.  Dish grid lives in x-y plane at a\n"
            "  constant Z. Orthogonal. Rotated by O(1) degrees from \"Topocentric\"\n"
            "\n"
            "Dish Elevation Axis: The axis around which the dishes pivot. positive is in the east direction.\n"
            "  *Not* parallel to the dish grid \"e/w\" separation. \"Coelevation\" pointing measures an angle around\n"
            "  this axis.\n"
            "\n"
            "Dish Vertical Axis: Local \"up/zenith\" for the dishes. The direction which has coelevation = 0.0.\n"
            "  Orthogonal to the Dish Elevation axis."
        ) << YAML::Newline;
        w << YAML::Newline << YAML::Comment(
            "Position on the Earth in degrees."
        ) << YAML::Newline;
    }
    w << YAML::Key << "tel_origin_itrs_lat_deg" << YAML::Value << ASDF::float_entry(m.tel_origin_itrs_lat_deg);
    w << YAML::Key << "tel_origin_itrs_lon_deg" << YAML::Value << ASDF::float_entry(m.tel_origin_itrs_lon_deg);

    if (verbose) {
        w << YAML::Newline << YAML::Newline << YAML::Comment(
            "Unit vectors in the x & y grid directions, in topocentric coordinates."
        ) << YAML::Newline;
    }
    w << YAML::Key << "tel_grid_x_axis" << YAML::Value << *_make_float_seq(m.tel_grid_x_axis);
    w << YAML::Key << "tel_grid_y_axis" << YAML::Value << *_make_float_seq(m.tel_grid_y_axis);

    if (verbose) {
        w << YAML::Newline << YAML::Newline << YAML::Comment(
            "Unit vectors for the dish frame, the elevation axis and vertical axis, in\n"
            "topocentric coordinates."
        ) << YAML::Newline;
    }
    w << YAML::Key << "tel_dish_elev_axis" << YAML::Value << *_make_float_seq(m.tel_dish_elev_axis);
    w << YAML::Key << "tel_dish_vert_axis" << YAML::Value << *_make_float_seq(m.tel_dish_vert_axis);

    if (verbose) {
        w << YAML::Newline << YAML::Newline << YAML::Comment(
            "The dish pointing angle, co-elevation in degrees: angle away from vertical,\n"
            "north is positive."
        ) << YAML::Newline;
    }
    w << YAML::Key << "tel_dish_coelev_deg" << YAML::Value << ASDF::float_entry(m.tel_dish_coelev_deg);

    if (verbose) {
        w << YAML::Newline << YAML::Newline << YAML::Comment(
            "Values of dish separation in x and y directions in meters."
        ) << YAML::Newline;
    }
    w << YAML::Key << "tel_dish_separation_x_m" << YAML::Value << ASDF::float_entry(m.tel_dish_separation_x_m);
    w << YAML::Key << "tel_dish_separation_y_m" << YAML::Value << ASDF::float_entry(m.tel_dish_separation_y_m);

    if (verbose) {
        w << YAML::Newline << YAML::Newline << YAML::Comment(
            "Per-zone noise variance; length nzones (= len(zone_nfreq)).\n"
            "Temporary kludge: the FRB server assumes mean-zero, time-uncorrelated\n"
            "noise with this variance. Will be generalized later."
        ) << YAML::Newline;
    }
    w << YAML::Key << "noise_variance" << YAML::Value << *_make_float_seq(m.noise_variance);

    w << YAML::EndMap;
}


// Inverse of _metadata_to_asdf_group(). Reads the ~17 round-trip-stable
// fields back from the sub-group. Does NOT populate freq_channels / beam_ids
// / beam_positions_{x,y} (those are reconstructed from per-frame data by
// AssembledFrame::from_asdf), and does NOT call validate() (the returned
// object is intentionally incomplete).
static shared_ptr<XEngineMetadata> _metadata_from_asdf_group(const shared_ptr<ASDF::group> &g)
{
    auto m = make_shared<XEngineMetadata>();
    m->version = _read_int(g, "version");
    m->zone_nfreq = _read_int_vec(g, "zone_nfreq");
    m->zone_freq_edges = _read_float_vec(g, "zone_freq_edges");
    m->beamset = _read_int(g, "beamset");
    // freq_channels / beam_ids / beam_positions_{x,y} reconstructed by caller

    m->unix_ns_at_seq_0 = _read_int(g, "unix_ns_at_seq_0");
    m->dt_ns_per_seq = _read_int(g, "dt_ns_per_seq");
    m->seq_per_frb_time_sample = _read_int(g, "seq_per_frb_time_sample");

    m->tel_origin_itrs_lat_deg = _read_float(g, "tel_origin_itrs_lat_deg");
    m->tel_origin_itrs_lon_deg = _read_float(g, "tel_origin_itrs_lon_deg");
    m->tel_grid_x_axis    = _read_float_arr3(g, "tel_grid_x_axis");
    m->tel_grid_y_axis    = _read_float_arr3(g, "tel_grid_y_axis");
    m->tel_dish_elev_axis = _read_float_arr3(g, "tel_dish_elev_axis");
    m->tel_dish_vert_axis = _read_float_arr3(g, "tel_dish_vert_axis");
    m->tel_dish_coelev_deg = _read_float(g, "tel_dish_coelev_deg");
    m->tel_dish_separation_x_m = _read_float(g, "tel_dish_separation_x_m");
    m->tel_dish_separation_y_m = _read_float(g, "tel_dish_separation_y_m");

    m->noise_variance = _read_float_vec(g, "noise_variance");
    return m;
}


long AssembledFrame::fpga_seq_start() const
{
    xassert(metadata != nullptr);
    return time_chunk_index * ntime * metadata->seq_per_frb_time_sample;
}


long AssembledFrame::fpga_seq_end() const
{
    xassert(metadata != nullptr);
    return (time_chunk_index + 1) * ntime * metadata->seq_per_frb_time_sample;
}


Array<float> AssembledFrame::dequantize() const
{
    // Acquire lock and copy the array members to local variables, to avoid racing
    // against the reaper thread (same pattern as write_asdf()). Copying an Array
    // also copies the shared_ptr in 'base', keeping the memory alive.
    Array<void> local_data;
    Array<void> local_scales_offsets;

    unique_lock<std::mutex> guard(mutex);
    local_data           = data;
    local_scales_offsets = scales_offsets;
    guard.unlock();

    if (local_data.size == 0)
        throw runtime_error("AssembledFrame::dequantize(): attempt to dequantize empty/reaped frame");

    // Single-beam dequantization: add a length-1 beam axis to this frame's arrays,
    // run the CPU reference kernel, and return the (nfreq, ntime) float32 result.
    ReferenceDequantizationKernel kernel(1, nfreq, ntime);

    Array<float> out({nfreq, ntime}, af_rhost | af_zero);
    Array<float> out3 = out.reshape({1, nfreq, ntime});
    Array<void> data3 = local_data.reshape({1, nfreq, ntime});
    Array<void> scoff_v = local_scales_offsets.reshape({1, nfreq, ntime/256, 2});
    Array<__half> scoff = scoff_v.cast<__half>();

    kernel.apply(out3, scoff, data3);
    return out;
}


void AssembledFrame::write_asdf(const std::string &filename, bool sync, bool verbose) const
{
    xassert(nfreq > 0);
    xassert(ntime > 0);
    xassert((ntime % 2) == 0);

    if (!metadata)
        throw runtime_error("AssembledFrame::write_asdf(): metadata is null");

    long mpc = xdiv(ntime, 256);

    // Acquire lock and copy data to local variable, to avoid racing against reaper thread.
    // Copying the Array also copies the shared_ptr in 'base', keeping the memory alive.
    Array<void> local_scales_offsets;
    Array<void> local_data;

    unique_lock<std::mutex> guard(mutex);
    local_scales_offsets = scales_offsets;
    local_data           = data;
    guard.unlock();

    if (local_data.size == 0)
        throw runtime_error("internal error: attempt to write empty/reaped frame");
    if (local_scales_offsets.size == 0)
        throw runtime_error("internal error: attempt to write empty/reaped scales_offsets");

    // Verify data array is valid and contiguous.
    xassert(local_data.data != nullptr);
    xassert(local_data.ndim == 2);
    xassert(local_data.shape[0] == nfreq);
    xassert(local_data.shape[1] == ntime);
    xassert(local_data.is_fully_contiguous());

    // Verify scales_offsets array is valid and contiguous.
    xassert(local_scales_offsets.data != nullptr);
    xassert(local_scales_offsets.ndim == 3);
    xassert(local_scales_offsets.shape[0] == nfreq);
    xassert(local_scales_offsets.shape[1] == mpc);
    xassert(local_scales_offsets.shape[2] == 2);
    xassert(local_scales_offsets.is_fully_contiguous());

    // Look up beam_position_{x,y} for this frame's beam in the metadata.
    // beam_ids / beam_positions_* are NOT written to ASDF (they would be
    // redundant for a single-frame file); instead we emit per-frame scalars.
    long beam_idx = -1;
    for (size_t i = 0; i < metadata->beam_ids.size(); i++) {
        if (metadata->beam_ids[i] == beam_id) {
            beam_idx = long(i);
            break;
        }
    }
    if (beam_idx < 0) {
        stringstream ss;
        ss << "AssembledFrame::write_asdf(): frame->beam_id=" << beam_id
           << " not found in metadata->beam_ids";
        throw runtime_error(ss.str());
    }
    xassert(long(metadata->beam_positions_x.size()) > beam_idx);
    xassert(long(metadata->beam_positions_y.size()) > beam_idx);
    double bx = metadata->beam_positions_x[beam_idx];
    double by = metadata->beam_positions_y[beam_idx];

    // Build ndarrays for scales_offsets and data. We attach the underlying
    // memory via ptr_block_t (no copy); the local Array shared_ptrs above
    // keep the storage alive through w.flush() below.
    //
    // scales_offsets: shape (nfreq, mpc, 2) float16; last axis is {scale, offset}.
    long so_nbytes = nfreq * mpc * 4;
    auto so_block = make_shared<ASDF::ptr_block_t>(local_scales_offsets.data, so_nbytes);
    auto so_mblock = ASDF::make_constant_memoized(shared_ptr<ASDF::block_t>(so_block));
    auto so_arr = make_shared<ASDF::ndarray>(
        so_mblock,
        std::optional<ASDF::block_info_t>(),
        ASDF::block_format_t::block,
        ASDF::compression_t::none,
        0,  // compression_level
        vector<bool>(),  // mask
        make_shared<ASDF::datatype_t>(ASDF::id_float16),
        ASDF::host_byteorder(),
        vector<int64_t>{nfreq, mpc, 2}
    );

    // data: int4 dtype (4 bits per element) stored as uint8 with shape (nfreq, ntime/2).
    long nbytes = nfreq * (ntime / 2);
    auto block = make_shared<ASDF::ptr_block_t>(local_data.data, nbytes);
    auto mblock = ASDF::make_constant_memoized(shared_ptr<ASDF::block_t>(block));
    auto arr = make_shared<ASDF::ndarray>(
        mblock,
        std::optional<ASDF::block_info_t>(),
        ASDF::block_format_t::block,
        ASDF::compression_t::none,
        0,  // compression_level
        vector<bool>(),  // mask
        make_shared<ASDF::datatype_t>(ASDF::id_uint8),
        ASDF::host_byteorder(),
        vector<int64_t>{nfreq, ntime/2}
    );

    // Emit the file manually via ASDF::writer rather than going through
    // ASDF::asdf + ASDF::group. The group representation is backed by
    // std::map, which alphabetizes keys; we want a custom order
    // (resembling configs/xengine_metadata.yml: freq -> beams ->
    // time -> nested xengine_metadata -> ndarrays). scales_offsets is
    // emitted before data so its binary block lands first in the file.
    {
        ofstream os(filename, ios::binary | ios::trunc | ios::out);
        if (!os)
            throw runtime_error("AssembledFrame::write_asdf(): couldn't open " + filename
                                + " for writing: " + string(strerror(errno)));

        ASDF::writer w(os, map<string, string>());

        w << YAML::LocalTag("core/asdf-1.1.0");
        w << YAML::Indent(4);    // 4-space (not yaml-cpp default 2-space) indent throughout
        w << YAML::BeginMap;

        if (verbose) {
            w << YAML::Newline << YAML::Comment(
                "This file contains FRB intensity data for one \"frame\":\n"
                "  - one beam\n"
                "  - all frequency channels\n"
                "  - a specific time range (or time \"chunk\")\n"
                "\n"
                "Written by pirate::AssembledFrame::write_asdf().\n"
                "Note that we define a \"minichunk\" to be 256 time samples.\n"
                "\n"
                "References:\n"
                "  configs/xengine_metadata.yml -- xengine_metadata fields\n"
                "  notes/network_protocol.md    -- wire protocol"
            ) << YAML::Newline << YAML::Newline;
        }

        // asdf-cxx library tag (matches ASDF::asdf::to_yaml).
        w << YAML::Key << "asdf/library" << YAML::Value
          << ASDF::software(ASDF_CXX_NAME, ASDF_CXX_AUTHOR, ASDF_CXX_HOMEPAGE, ASDF_CXX_VERSION);

        if (verbose) {
            w << YAML::Newline << YAML::Newline << YAML::Comment(
                "Per-frame scalar metadata.\n"
                "  nfreq:            total frequency channels across all zones; equals\n"
                "                    sum(xengine_metadata.zone_nfreq) below.\n"
                "  beam_id:          integer id of this beam (an entry of the X-engine's\n"
                "                    beam_ids).\n"
                "  beam_position_x:  see below.\n"
                "  beam_position_y:  see below.\n"
                "  ntime:            number of time samples in this frame (a multiple of 256;\n"
                "                    equals the server's 'time_samples_per_chunk' parameter).\n"
                "  time_chunk_index: chunk index of this frame's first time sample, in units\n"
                "                    of ntime samples.\n"
                "  fpga_seq:         X-engine FPGA sequence number at the beginning of this frame.\n"
                "                    Equals time_chunk_index * ntime * xengine_metadata.seq_per_frb_time_sample.\n"
                "  unix_time_ns:     UNIX time in nanoseconds at the beginning of this frame.\n"
                "                    Equals xengine_metadata.unix_ns_at_seq_0 + fpga_seq * dt_ns_per_seq.\n"
                "\n"
                "beam_position_{x,y} are direction cosines in the grid frame.\n"
                "The grid frame is defined by x & y unit vectors which are orthogonal\n"
                "and lie along (or close to) the axes of the telescope grid. Each beam\n"
                "has a skywards-directed unit vector b.  The grid_x and grid_y values\n"
                "are the x & y direction cosines: b.x and b.y."
            ) << YAML::Newline << YAML::Newline;
        }
        w << YAML::Key << "nfreq" << YAML::Value << ASDF::int_entry(int64_t(nfreq));
        w << YAML::Key << "beam_id" << YAML::Value << ASDF::int_entry(int64_t(beam_id));
        w << YAML::Key << "beam_position_x" << YAML::Value << ASDF::float_entry(bx);
        w << YAML::Key << "beam_position_y" << YAML::Value << ASDF::float_entry(by);
        w << YAML::Key << "ntime" << YAML::Value << ASDF::int_entry(int64_t(ntime));
        w << YAML::Key << "time_chunk_index" << YAML::Value << ASDF::int_entry(int64_t(time_chunk_index));

        // Derived per-frame timing (not stored on AssembledFrame; computed from
        // metadata + time_chunk_index). Surfaced as ASDF keys so a reader can
        // look up the FPGA seq / UNIX timestamp without redoing the arithmetic.
        long fpga_seq = time_chunk_index * ntime * metadata->seq_per_frb_time_sample;
        long unix_time_ns = metadata->unix_ns_at_seq_0 + fpga_seq * metadata->dt_ns_per_seq;
        w << YAML::Key << "fpga_seq" << YAML::Value << ASDF::int_entry(int64_t(fpga_seq));
        w << YAML::Key << "unix_time_ns" << YAML::Value << ASDF::int_entry(int64_t(unix_time_ns));

        if (verbose) {
            w << YAML::Newline << YAML::Newline << YAML::Comment(
                "xengine_metadata: subset of the YAML the X-engine sent over the\n"
                "wire when this frame's data arrived (see notes/network_protocol.md).\n"
                "\n"
                "Reproduced verbatim except for three projections in this single-frame view:\n"
                "  freq_channels:    omitted (one ASDF file aggregates all sender subsets);\n"
                "  beam_ids:         not emitted (the top-level beam_id supersedes it);\n"
                "  beam_positions_*: not emitted (see top-level beam_position_x / _y)."
            ) << YAML::Newline << YAML::Newline;
        }

        // XEngineMetadata sub-map (inner key order also follows xengine_metadata.yml).
        w << YAML::Key << "xengine_metadata" << YAML::Value;
        _emit_metadata_yaml(w, *metadata, verbose);

        if (verbose) {
            w << YAML::Newline << YAML::Newline << YAML::Comment(
                "scales_offsets: per-(freq, minichunk) dequantization parameters as a\n"
                "(nfreq, ntime/256, 2) float16 ndarray. The last axis is (scale, offset).\n"
                "One (scale, offset) pair is applied to every int4 sample in the matching\n"
                "(freq, minichunk) slice of 'data' below. Sent over the wire as a\n"
                "(nbeams, nfreq, 2) float16 array per minichunk; this file aggregates\n"
                "(ntime/256) minichunks for a single beam."
            ) << YAML::Newline << YAML::Newline;
        }
        w << YAML::Key << "scales_offsets" << YAML::Value << *so_arr;

        if (verbose) {
            w << YAML::Newline << YAML::Newline << YAML::Comment(
                "data: int4 intensity samples, shape (nfreq, ntime). int4 is stored on\n"
                "disk as uint8 with shape (nfreq, ntime/2): two int4 values are packed\n"
                "per byte as ((x[1] << 4) | x[0]). Each int4 holds a value in [-8, +7];\n"
                "the reserved value -8 (encoded as 0x8) indicates a masked sample."
            ) << YAML::Newline << YAML::Newline;
        }
        w << YAML::Key << "data" << YAML::Value << *arr;

        w << YAML::EndMap;
        w.flush();

        // ofstream reports write failures by silently setting badbit (and
        // ASDF::writer does no stream checking of its own), so without this
        // check a disk-full/quota error would yield a TRUNCATED file that
        // this method reports as SUCCESS -- the fsync below only catches a
        // MISSING file, and FileWriter would then commit the partial file
        // into the acquisition directory and notify subscribers of a
        // successful write. Stream state is sticky, so one check after the
        // final flush covers every write since open.
        os.flush();
        if (!os)
            throw runtime_error("AssembledFrame::write_asdf(): I/O error writing " + filename
                                + " (disk full?): " + string(strerror(errno)));
    }

    // Re-open and fsync to ensure data is flushed to disk.
    if (sync) {
        int fd = open(filename.c_str(), O_RDONLY);
        if (fd < 0)
            throw runtime_error("AssembledFrame::write_asdf(): open() failed: " + string(strerror(errno)));
        if (fsync(fd) < 0) {
            close(fd);
            throw runtime_error("AssembledFrame::write_asdf(): fsync() failed: " + string(strerror(errno)));
        }
        close(fd);
    }
}


shared_ptr<AssembledFrame> AssembledFrame::from_asdf(const std::string &filename)
{
    // Read ASDF file.
    ASDF::asdf project(filename);
    auto grp = project.get_group();
    xassert(grp != nullptr);

    // Read per-frame scalars.
    long nfreq = _read_int(grp, "nfreq");
    long ntime = _read_int(grp, "ntime");
    long beam_id = _read_int(grp, "beam_id");
    long time_chunk_index = _read_int(grp, "time_chunk_index");
    double beam_position_x = _read_float(grp, "beam_position_x");
    double beam_position_y = _read_float(grp, "beam_position_y");

    xassert(nfreq > 0);
    xassert(ntime > 0);
    xassert((ntime % 2) == 0);

    // Read xengine_metadata sub-group and reconstruct the projected fields.
    auto md_entry = grp->at("xengine_metadata");
    auto md_grp_inner = md_entry->get_maybe_group();
    if (!md_grp_inner)
        throw runtime_error("AssembledFrame::from_asdf(): 'xengine_metadata' is not a group");
    // get_maybe_group returns a shared_ptr<map>; we need a shared_ptr<group>.
    // Easiest: dynamic_pointer_cast on the entry.
    auto md_grp = std::dynamic_pointer_cast<ASDF::group>(md_entry);
    if (!md_grp)
        throw runtime_error("AssembledFrame::from_asdf(): 'xengine_metadata' is not a group entry");

    auto md = _metadata_from_asdf_group(md_grp);

    // Per-frame projection: rebuild the four special members from frame data.
    // (An ASDF file describes one (beam, time-chunk), so these are length-1 / empty.)
    md->freq_channels.clear();
    md->beam_ids = { beam_id };
    md->beam_positions_x = { beam_position_x };
    md->beam_positions_y = { beam_position_y };

    md->validate();

    long mpc = ntime / 256;

    // Allocate the frame (host memory, uninitialized) via the shared factory,
    // which derives nfreq from the metadata's zone_nfreq and enforces the ntime
    // constraints. The scales_offsets/data arrays are filled by the memcpys below.
    auto frame = AssembledFrame::make_uninitialized(md, ntime, beam_id, time_chunk_index);

    // Cross-check the metadata-derived nfreq (used for the array shapes) against
    // the file's explicit 'nfreq' scalar. These can only disagree for a corrupt
    // or internally-inconsistent ASDF file.
    if (frame->nfreq != nfreq) {
        stringstream ss;
        ss << "AssembledFrame::from_asdf(): file nfreq=" << nfreq
           << " disagrees with metadata-derived nfreq=" << frame->nfreq;
        throw runtime_error(ss.str());
    }

    // Read scales_offsets array (float16, shape (nfreq, mpc, 2)).
    {
        auto so_entry = grp->at("scales_offsets");
        auto so_ndarr = so_entry->get_maybe_ndarray();
        xassert(so_ndarr != nullptr);

        auto so_shape = so_ndarr->get_shape();
        xassert(so_shape.size() == 3);
        xassert(so_shape[0] == nfreq);
        xassert(so_shape[1] == mpc);
        xassert(so_shape[2] == 2);

        auto so_mdata = so_ndarr->get_data();
        const void *so_src = so_mdata->ptr();
        size_t so_nbytes = so_mdata->nbytes();
        xassert(so_nbytes == (size_t)(nfreq * mpc * 4));   // 2 float16 = 4 bytes per (freq, imc)

        memcpy(frame->scales_offsets.data, so_src, so_nbytes);
    }

    // Read data array.
    {
        auto data_entry = grp->at("data");
        auto arr = data_entry->get_maybe_ndarray();
        xassert(arr != nullptr);

        // Verify shape: uint8 with shape (nfreq, ntime/2).
        auto shape = arr->get_shape();
        xassert(shape.size() == 2);
        xassert(shape[0] == nfreq);
        xassert(shape[1] == ntime/2);

        // Get data pointer.
        auto mdata = arr->get_data();
        const void *src_ptr = mdata->ptr();
        size_t nbytes = mdata->nbytes();
        xassert(nbytes == (size_t)(nfreq * (ntime / 2)));

        memcpy(frame->data.data, src_ptr, nbytes);
    }

    return frame;
}


// -------------------------------------------------------------------------------------------------
//
// AssembledFrameSet methods.


void AssembledFrameSet::validate() const
{
    if (!metadata)
        throw runtime_error("AssembledFrameSet::validate(): metadata is null");

    long md_nbeams = long(metadata->beam_ids.size());
    if (nbeams != md_nbeams) {
        stringstream ss;
        ss << "AssembledFrameSet::validate(): nbeams=" << nbeams
           << " disagrees with metadata->beam_ids.size()=" << md_nbeams;
        throw runtime_error(ss.str());
    }
    if (long(frames.size()) != nbeams) {
        stringstream ss;
        ss << "AssembledFrameSet::validate(): frames.size()=" << frames.size()
           << " disagrees with nbeams=" << nbeams;
        throw runtime_error(ss.str());
    }

    for (long b = 0; b < nbeams; b++) {
        const auto &f = frames[b];
        if (!f) {
            stringstream ss;
            ss << "AssembledFrameSet::validate(): frames[" << b << "] is null";
            throw runtime_error(ss.str());
        }
        if (f->metadata != metadata)
            throw runtime_error("AssembledFrameSet::validate(): frame metadata pointer disagrees with set");
        if (f->time_chunk_index != time_chunk_index) {
            stringstream ss;
            ss << "AssembledFrameSet::validate(): frames[" << b << "]->time_chunk_index=" << f->time_chunk_index
               << " disagrees with set's time_chunk_index=" << time_chunk_index;
            throw runtime_error(ss.str());
        }
        if (f->nfreq != nfreq) {
            stringstream ss;
            ss << "AssembledFrameSet::validate(): frames[" << b << "]->nfreq=" << f->nfreq
               << " disagrees with set's nfreq=" << nfreq;
            throw runtime_error(ss.str());
        }
        if (f->ntime != ntime) {
            stringstream ss;
            ss << "AssembledFrameSet::validate(): frames[" << b << "]->ntime=" << f->ntime
               << " disagrees with set's ntime=" << ntime;
            throw runtime_error(ss.str());
        }
        if (f->beam_id != metadata->beam_ids[b]) {
            stringstream ss;
            ss << "AssembledFrameSet::validate(): frames[" << b << "]->beam_id=" << f->beam_id
               << " disagrees with metadata->beam_ids[" << b << "]=" << metadata->beam_ids[b];
            throw runtime_error(ss.str());
        }
    }
}


const shared_ptr<AssembledFrame> &AssembledFrameSet::get_frame(long ibeam) const
{
    return frames.at(ibeam);
}


void AssembledFrameSet::randomize(bool normalize, bool gaussian)
{
    // Serial per-frame randomization. See the header for the parallel
    // alternative (SimulatedFrameFactory).
    for (const auto &f : frames) {
        xassert(f);   // AssembledFrameSet invariant: all frames[i] non-null
        f->randomize(normalize, gaussian, nullptr, 0);   // no pulse injection at the set level
    }
}


// -------------------------------------------------------------------------------------------------
//
// Static AssembledFrame member functions, for testing.


// Static member function.
shared_ptr<AssembledFrame> AssembledFrame::make_uninitialized(
    const shared_ptr<const XEngineMetadata> &xmd,
    long ntime, long beam_id, long time_chunk_index)
{
    if (!xmd)
        throw runtime_error("AssembledFrame::make_uninitialized(): xmd is null");

    // Check that beam_id appears in xmd->beam_ids.
    bool found = false;
    for (long b : xmd->beam_ids) {
        if (b == beam_id) {
            found = true;
            break;
        }
    }
    if (!found) {
        stringstream ss;
        ss << "AssembledFrame::make_uninitialized(): beam_id=" << beam_id
           << " is not in xmd->beam_ids";
        throw runtime_error(ss.str());
    }

    long nfreq = xmd->get_total_nfreq();
    xassert(nfreq > 0);
    xassert(ntime > 0);
    xassert((ntime % 2) == 0);
    xassert((ntime % 256) == 0);
    long mpc = ntime / 256;

    // Allocate scales_offsets before data, mirroring slab order in the allocator.
    // The two arrays are left UNINITIALIZED -- the caller fills them.
    auto frame = make_shared<AssembledFrame>();
    frame->nfreq = nfreq;
    frame->ntime = ntime;
    frame->beam_id = beam_id;
    frame->time_chunk_index = time_chunk_index;
    frame->metadata = xmd;
    frame->scales_offsets = Array<void>(Dtype(df_float, 16), {nfreq, mpc, 2}, af_rhost);
    frame->data           = Array<void>(Dtype(df_int, 4),    {nfreq, ntime}, af_rhost);

    return frame;
}


// Helper for AssembledFrame::randomize() pulse injection: throw unless the SinglePulse is
// consistent with the frame's metadata -- same nfreq, per-channel frequency edges, per-channel
// noise variances, and time-sample duration. (These must match for the injected pulse's dispersion
// sweep to land on the right channels/samples and for its snr normalization to be correct.)
static void check_pulse_consistency(long nf, const XEngineMetadata &md, const simpulse::SinglePulse &sp)
{
    const simpulse::SinglePulse::Params &spp = sp.params;

    // nfreq (checked first: the loops below index per channel).
    xassert_eq(spp.freq_variances.size, nf);
    xassert_eq(spp.freq_edges_MHz.size, nf + 1);
    xassert_eq(md.get_total_nfreq(), nf);

    // Time-sample duration: dt_sp is an INTEGER sample offset, so the pulse's dt must equal the
    // frame's (dt_ns_per_seq * seq_per_frb_time_sample).
    double frame_dt_ms = (double) md.dt_ns_per_seq * (double) md.seq_per_frb_time_sample / 1.0e6;
    if (std::fabs(frame_dt_ms - spp.time_sample_ms) > 1.0e-6 * frame_dt_ms) {
        std::stringstream ss;
        ss << "AssembledFrame::randomize: SinglePulse time_sample_ms (" << spp.time_sample_ms
           << ") does not match the frame's time-sample duration (" << frame_dt_ms << " ms)";
        throw std::runtime_error(ss.str());
    }

    // Per-channel frequency edges + noise variances, expanded from the zone structure by the
    // XEngineMetadata accessors (zones ordered low-to-high, matching SinglePulse's channel order).
    std::vector<double> md_edges = md.get_channel_freq_edges();   // nf+1
    std::vector<double> md_var   = md.get_channel_variances();    // nf
    xassert_eq((long) md_edges.size(), nf + 1);
    xassert_eq((long) md_var.size(), nf);

    const double *sp_edges = spp.freq_edges_MHz.data;    // nf+1
    const double *sp_var   = spp.freq_variances.data;    // nf
    const double edge_tol = 1.0e-6 * (md_edges[nf] - md_edges[0]);

    for (long f = 0; f <= nf; f++) {
        if (std::fabs(sp_edges[f] - md_edges[f]) > edge_tol) {
            std::stringstream ss;
            ss << "AssembledFrame::randomize: SinglePulse freq_edges_MHz[" << f << "] ("
               << sp_edges[f] << ") does not match the frame channel edge (" << md_edges[f] << ")";
            throw std::runtime_error(ss.str());
        }
    }

    for (long f = 0; f < nf; f++) {
        double vtol = 1.0e-6 * (md_var[f] > 0.0 ? md_var[f] : 1.0);
        if (std::fabs(sp_var[f] - md_var[f]) > vtol) {
            std::stringstream ss;
            ss << "AssembledFrame::randomize: SinglePulse freq_variances[" << f << "] ("
               << sp_var[f] << ") does not match the frame per-channel noise variance ("
               << md_var[f] << ")";
            throw std::runtime_error(ss.str());
        }
    }
}


// Helper for AssembledFrame::randomize() pulse injection: overwrite each channel's (contiguous)
// pulse samples in the already-noise-filled int4 'data_arr'. Frame row f == SinglePulse channel f
// (both low-to-high). The pulse occupies pulse-time [freq_it0[f], freq_it0[f]+freq_nt[f]); frame
// time it_frame maps to pulse time (it_frame + dt_sp). Each pulse sample becomes
// quantize(signal/S[f] + prequant_rms*gaussian) -- matching avx2_simulate_4bit_noise()'s inverse-CDF
// levels -- where the signal (post-scaled units) is divided by S[f] into pre-scaled units first.
static void inject_single_pulse(const ksgpu::Array<void> &data_arr, long nfreq, long ntime,
                                const std::vector<float> &S, const simpulse::SinglePulse &sp,
                                long dt_sp, std::mt19937 &rng)
{
    std::normal_distribution<float> gdist(0.0f, avx2_4bit_prequant_noise_rms);   // rms = 2.5
    unsigned char *bytes = static_cast<unsigned char *>(data_arr.data);

    const long  *it0v = sp.freq_it0.data;
    const long  *ntv  = sp.freq_nt.data;
    const long  *offv = sp.freq_sd_off.data;
    const float *sd   = sp.sparse_data.data;

    for (long f = 0; f < nfreq; f++) {
        long nt_ch = ntv[f];
        if (nt_ch == 0)
            continue;                                   // no pulse in this channel

        // Frame-time window of this channel's pulse run, clipped to [0, ntime). Partial (or zero)
        // overlap is fine -- a frame is one chunk of a longer stream.
        long t0 = std::max(0L,    it0v[f]         - dt_sp);
        long t1 = std::min(ntime, it0v[f] + nt_ch - dt_sp);
        if (t0 >= t1)
            continue;

        float inv_S = 1.0f / S[f];
        long  row = f * ntime;                          // int4 index of row start (even; ntime % 256 == 0)
        long  sd0 = offv[f], it0 = it0v[f];
        for (long t = t0; t < t1; t++) {
            long  k = (t + dt_sp) - it0;                // invariant: 0 <= k < nt_ch
            float x = sd[sd0 + k] * inv_S + gdist(rng); // pre-scaled signal + pre-scaled noise
            // Round-half-up, clamp to [-7,7] (never -8). Clamp in the FLOAT
            // domain before the int cast: float->int conversion of an
            // out-of-range value is UB, and x can overflow int for
            // pathological signal/scale combinations.
            float xq = std::min(7.0f, std::max(-7.0f, std::floor(x + 0.5f)));
            int   q = (int) xq;
            long  idx = row + t;                        // int4 index; nibble parity == t parity
            unsigned char &b = bytes[idx >> 1];
            if (idx & 1) b = (unsigned char)((b & 0x0F) | ((q & 0xF) << 4));   // high nibble (odd t)
            else         b = (unsigned char)((b & 0xF0) |  (q & 0xF));         // low  nibble (even t)
        }
    }
}


void AssembledFrame::randomize(bool normalize, bool gaussian,
                               const shared_ptr<const simpulse::SinglePulse> &sp, long dt_sp)
{
    // Pulse injection preconditions + consistency, validated up front (before touching buffers).
    if (sp) {
        if (!gaussian || !normalize)
            throw std::runtime_error("AssembledFrame::randomize: signal injection (sp != null)"
                                     " requires gaussian=true and normalize=true");
        xassert(metadata);   // non-null by invariant
        check_pulse_consistency(nfreq, *metadata, *sp);
    }

    // Thread-safety: the array STATE (empty vs nonempty, and which slab the
    // arrays point at) is lock-protected -- a concurrent _reap_locked() on the
    // reaper / ssd-writer thread can drop 'scales_offsets'/'data' and free the
    // underlying slab at any time. We therefore snapshot both Arrays into local
    // copies while holding the lock; copying bumps the shared refcounted 'base',
    // so the slab memory stays alive even if the frame is reaped concurrently.
    // We then release the lock and run the (long) random-fill loops on the local
    // copies, so the lock is not held during the bulk fill. (If a reap races in
    // after the snapshot, we harmlessly fill memory that is about to be
    // discarded -- not a use-after-free, since our copies keep it alive.)
    Array<void> so_arr, data_arr;
    {
        std::lock_guard<std::mutex> guard(mutex);
        so_arr   = scales_offsets;
        data_arr = data;
    }

    // int4 dtype packs 2 elements per byte, so nbytes = data.size / 2.
    // (data.size is the int4 element count, not the byte count -- see
    // AssembledFrameAllocator::_worker_main in this file.)
    long data_nbytes = data_arr.size / 2;
    long so_nelts    = so_arr.size;  // (nfreq, mpc, 2) flat count

    if ((data_nbytes <= 0) && (so_nelts <= 0))
        return;  // empty (e.g. reaped) frame -- nothing to do.

    // ksgpu's default_rng() is per-thread, so concurrent calls from different
    // threads do not race on RNG state. Callers still must not concurrently
    // read/write the same frame's data buffer -- the destination buffer is
    // not protected.
    std::mt19937 &rng = ksgpu::default_rng();

    // Per-frequency calibrated scale S[f] = sqrt(V_f / Vq), computed once (only when 'normalize')
    // and shared by the scales_offsets fill AND the pulse injection.
    //
    // Derivation: the dequantizer computes out = scale*v + offset for each int4 sample v, EXCEPT the
    // sentinel v = -8 dequantizes to 0 (see GpuDequantizationKernel). With offset = 0, out = S*w
    // where w = 0 if v = -8 else v. Var(out) = S^2 * Var(w); matching the per-zone target variance
    // V_f (metadata->noise_variance) gives S = sqrt(V_f / Vq), Vq = Var(w):
    //   - uniform (gaussian=false): v uniform over [-8,7] (folding -8 -> 0) gives Vq = 280/16 = 17.5.
    //   - gaussian (gaussian=true): v never equals -8, so Vq = avx2_4bit_postquant_noise_rms()^2.
    std::vector<float> S;   // empty unless 'normalize'
    if (normalize) {
        double postquant_rms = avx2_4bit_postquant_noise_rms();
        double data_variance = gaussian ? (postquant_rms * postquant_rms) : 17.5;

        xassert(metadata);   // non-null by invariant (immutable; never reaped)
        std::vector<double> cv = metadata->get_channel_variances();
        xassert_eq((long) cv.size(), nfreq);   // metadata's zone_nfreq must sum to this frame's nfreq

        S.resize(nfreq);
        for (long f = 0; f < nfreq; f++)
            S[f] = (float) std::sqrt(cv[f] / data_variance);
    }

    // Fill scales_offsets first (matches slab order). The (scale, offset) pairs
    // are laid out as a contiguous (nfreq, mpc, 2) array, so pair index i maps to
    // frequency channel (i / mpc).
    if (so_nelts > 0) {
        xassert(so_arr.data != nullptr);
        xassert((so_nelts % 2) == 0);
        xassert(so_arr.is_fully_contiguous());   // loop below indexes linearly from the base ptr
        __half *so = static_cast<__half *>(so_arr.data);
        long npairs = so_nelts / 2;

        if (!normalize) {
            // Un-normalized: scales uniform in [0, 1], offsets uniform in [-1, 1].
            std::uniform_real_distribution<float> dist_scale ( 0.0f, 1.0f);
            std::uniform_real_distribution<float> dist_offset(-1.0f, 1.0f);
            for (long i = 0; i < npairs; i++) {
                so[2*i + 0] = __float2half_rn(dist_scale (rng));
                so[2*i + 1] = __float2half_rn(dist_offset(rng));
            }
        }
        else {
            // Calibrated: scale = S[f] (computed above), offset = 0.
            xassert(so_arr.ndim == 3);
            long mpc = so_arr.shape[1];             // (scale, offset) pairs per frequency
            xassert_eq(npairs, nfreq * mpc);        // nfreq is the frame member == so_arr.shape[0]
            __half zero = __float2half_rn(0.0f);
            for (long i = 0; i < npairs; i++) {
                so[2*i + 0] = __float2half_rn(S[i / mpc]);   // scale (frequency = i / mpc)
                so[2*i + 1] = zero;                          // offset = 0
            }
        }
    }

    if (data_nbytes > 0) {
        xassert(data_arr.data != nullptr);
        xassert(data_arr.is_fully_contiguous());   // contiguous sweep below

        if (sp) {
            // Signal + noise (gaussian && normalize guaranteed by the precondition above; S is
            // populated). Fill the whole frame with pure noise (fast SIMD), then overwrite each
            // channel's sparse pulse samples.
            avx2_simulate_4bit_noise(static_cast<unsigned int *>(data_arr.data), data_arr.size);
            inject_single_pulse(data_arr, nfreq, ntime, S, *sp, dt_sp, rng);
        }
        else if (gaussian) {
            // Simulated Gaussian noise quantized to int4 in [-7,7] (the -8 sentinel is never
            // produced). data_arr.size is the int4 element count = nfreq*ntime, a multiple of 64
            // (ntime is a multiple of 256), as required by avx2_simulate_4bit_noise().
            avx2_simulate_4bit_noise(static_cast<unsigned int *>(data_arr.data), data_arr.size);
        }
        else {
            char *p = static_cast<char *>(data_arr.data);

            // mt19937 yields uint32_t (4 random bytes per call). The final
            // 1-3 bytes (if data_nbytes % 4 != 0) get the low bytes of a
            // fresh uint32 via a short-memcpy.
            long i = 0;
            while (i + 4 <= data_nbytes) {
                uint32_t r = rng();
                std::memcpy(p + i, &r, 4);
                i += 4;
            }
            if (i < data_nbytes) {
                uint32_t r = rng();
                std::memcpy(p + i, &r, data_nbytes - i);
            }
        }
    }
}


// -------------------------------------------------------------------------------------------------
//
// Constructor


AssembledFrameAllocator::AssembledFrameAllocator(const shared_ptr<SlabAllocator> &slab_allocator_,
                                                 int num_consumers_,
                                                 long time_samples_per_chunk_)
    : time_samples_per_chunk(time_samples_per_chunk_),
      slab_allocator(slab_allocator_),
      num_consumers(num_consumers_),
      is_dummy_mode(slab_allocator_->is_dummy())
{
    xassert(slab_allocator);
    xassert_gt(num_consumers, 0);
    xassert_gt(time_samples_per_chunk, 0L);

    if (!ksgpu::af_on_host(slab_allocator->aflags))
        throw runtime_error("AssembledFrameAllocator: slab_allocator must be on host");

    // Spawn the worker thread (sole producer of frame sets, in both dummy
    // and non-dummy mode). It parks on its init gate until metadata and
    // the initial chunk are established.
    worker_thread = thread(&AssembledFrameAllocator::worker_main, this);
}


// -------------------------------------------------------------------------------------------------
//
// Destructor


AssembledFrameAllocator::~AssembledFrameAllocator()
{
    this->stop();
    if (worker_thread.joinable())
        worker_thread.join();
}


// -------------------------------------------------------------------------------------------------
//
// stop()


void AssembledFrameAllocator::stop(exception_ptr e) const
{
    unique_lock<mutex> guard(lock);

    if (is_stopped)
        return;

    error = e;
    is_stopped = true;

    guard.unlock();
    metadata_cv.notify_all();
    chunk_cv.notify_all();
    queue_cv.notify_all();
    slot_cv.notify_all();
    lowmem_cv.notify_all();

    // Propagate stop to SlabAllocator (wakes up worker thread if blocked in get_slab).
    slab_allocator->stop(e);
}


bool AssembledFrameAllocator::is_initialized() const
{
    // Delegates to slab_allocator, which delegates to bump_allocator.
    return slab_allocator->is_initialized();
}


// -------------------------------------------------------------------------------------------------
//
// _throw_if_stopped(): helper for entry points. Caller must hold lock.


void AssembledFrameAllocator::_throw_if_stopped(const char *method_name)
{
    if (error)
        rethrow_exception(error);
    
    if (is_stopped)
        throw runtime_error(string(method_name) + " called on stopped instance");
}


// -------------------------------------------------------------------------------------------------
//
// Worker thread
//
// The worker thread is the SOLE producer of AssembledFrameSets, in both
// dummy and non-dummy mode: it pre-initializes whole sets (memsets nbeams
// slabs to 0x88) and pushes them to frame_set_queue, so get_frame_set()
// callers never pay allocation/memset latency. It is throttled by the
// queue bound (constants::assembled_frame_allocator_queue_size) -- the
// only throttle in dummy mode, where get_slab() never blocks -- and by
// blocking get_slab() when the slab pool is exhausted.


void AssembledFrameAllocator::_worker_main()
{
    unique_lock<mutex> guard(lock);

    // Wait until: stopped, or both initialization phases are complete.
    // We need metadata (nfreq, beam_ids) to size and stamp frames, AND
    // we need initial_time_chunk to stamp the first set's chunk index.
    // The two flags are set-once and never cleared, so waiting for them
    // sequentially (each on its own cv) is equivalent to waiting for the
    // conjunction.
    while (!is_stopped && !metadata_is_initialized)
        metadata_cv.wait(guard);
    while (!is_stopped && !initial_chunk_set)
        chunk_cv.wait(guard);
    if (is_stopped)
        return;

    // Snapshot the build parameters into locals, under the lock. All of
    // them are set-once (written before the init gate above opens, and
    // immutable afterward), so one snapshot suffices for the whole main
    // loop -- and the unlocked build section below then touches no
    // lock-protected members.
    long nbeams = beam_ids.size();
    long nfreq_snap = nfreq;
    vector<long> beam_ids_snap = beam_ids;
    shared_ptr<const XEngineMetadata> md = metadata;
    xassert(md);  // non-null once metadata_is_initialized

    long mpc = xdiv(time_samples_per_chunk, 256);
    SlabLayout layout = get_layout(nfreq_snap, time_samples_per_chunk);
    long nbytes = layout.slab_nbytes;

    // Main loop: wait for a free queue slot, build a set with the lock
    // dropped, then stamp its chunk index + push it with the lock re-held.
    for (;;) {
        while (!is_stopped && (long(frame_set_queue.size()) >= constants::assembled_frame_allocator_queue_size))
            slot_cv.wait(guard);
        if (is_stopped)
            return;

        // Drop the lock while building the set (slab acquisition + memset
        // of the nbeams slabs). A throw below (e.g. get_slab() on a
        // stopped slab allocator) unwinds with the lock released;
        // worker_main()'s catch handler then calls stop(), which wakes all
        // waiters. No queue state needs restoring, since nothing is
        // published until the push below.
        guard.unlock();

        // The set is fully initialized here except for time_chunk_index
        // (on the set and on each frame), which is stamped under the lock
        // at push time so it is consistent with the queue state.
        auto set = make_shared<AssembledFrameSet>();
        set->nfreq = nfreq_snap;
        set->ntime = time_samples_per_chunk;
        set->nbeams = nbeams;
        set->metadata = md;
        set->frames.reserve(nbeams);

        // Build nbeams frames, all backed by independent slabs.
        //
        // Slab layout: scales_offsets at offset 0, data at layout.data_offset (both
        // cache-line aligned -- see AssembledFrameAllocator::get_layout()). Both arrays
        // share a single slab shared_ptr; _reap_locked() drops both refs to free.
        //
        // Slab-pool sizing: this loop holds up to nbeams slabs at a time. The
        // pool must have at least nbeams total slabs to make progress, which
        // is also a hard prerequisite for the Receiver's 2-chunk window.
        for (long b = 0; b < nbeams; b++) {
            shared_ptr<void> slab = slab_allocator->get_slab(nbytes, /*blocking=*/true);

            // scales_offsets initial: float16 0.0 (bytes 0x00); also zero the
            // alignment padding between the two arrays (up to data_offset).
            // data initial: int4 -8 (bytes 0x88, two -8 nibbles per byte).
            memset(slab.get(), 0x00, layout.data_offset);
            memset((char *)slab.get() + layout.data_offset, 0x88, layout.data_nbytes);

            auto frame = make_shared<AssembledFrame>();
            frame->nfreq = nfreq_snap;
            frame->ntime = time_samples_per_chunk;
            frame->beam_id = beam_ids_snap[b];
            frame->metadata = md;  // shared, immutable

            // Initialize frame->scales_offsets at slab offset 0.
            frame->scales_offsets.data = slab.get();
            frame->scales_offsets.ndim = 3;
            frame->scales_offsets.shape[0] = nfreq_snap;
            frame->scales_offsets.shape[1] = mpc;
            frame->scales_offsets.shape[2] = 2;
            frame->scales_offsets.size = nfreq_snap * mpc * 2;
            frame->scales_offsets.strides[0] = mpc * 2;
            frame->scales_offsets.strides[1] = 2;
            frame->scales_offsets.strides[2] = 1;
            frame->scales_offsets.dtype = ksgpu::Dtype(ksgpu::df_float, 16);
            frame->scales_offsets.aflags = slab_allocator->aflags;
            frame->scales_offsets.base = slab;   // shares slab with data
            frame->scales_offsets.check_invariants("AssembledFrameAllocator::_worker_main()");

            // Initialize frame->data at slab offset layout.data_offset
            // (cache-line aligned -- see AssembledFrameAllocator::get_layout()).
            frame->data.data = (char *)slab.get() + layout.data_offset;
            frame->data.ndim = 2;
            frame->data.shape[0] = nfreq_snap;
            frame->data.shape[1] = time_samples_per_chunk;
            frame->data.size = nfreq_snap * time_samples_per_chunk;
            frame->data.strides[0] = time_samples_per_chunk;
            frame->data.strides[1] = 1;
            frame->data.dtype = ksgpu::Dtype(ksgpu::df_int, 4);
            frame->data.aflags = slab_allocator->aflags;
            frame->data.base = slab;
            frame->data.check_invariants("AssembledFrameAllocator::_worker_main()");

            set->frames.push_back(std::move(frame));
        }

        guard.lock();

        // Stamp the chunk index after re-acquiring the lock. (Consumers
        // popping fully-received sets off the front while the lock was
        // dropped don't change queue_start_chunk_index + frame_set_queue.size(),
        // since both change together; this is mostly defensive.)
        long chunk_index = queue_start_chunk_index + long(frame_set_queue.size());
        set->time_chunk_index = chunk_index;
        for (long b = 0; b < nbeams; b++)
            set->frames[b]->time_chunk_index = chunk_index;

        set->validate();  // cheap defensive check
        frame_set_queue.push_back({std::move(set), 0});

        // Wake up any waiting get_frame_set() callers. notify_all: every
        // consumer receives every set, so one push can ready several
        // waiters. (Notify under lock, deliberately: the lock stays held
        // into the next loop iteration.)
        queue_cv.notify_all();
    }
}


void AssembledFrameAllocator::worker_main()
{
    try {
        _worker_main();  // only returns if is_stopped
    } catch (...) {
        stop(current_exception());
    }
}


// -------------------------------------------------------------------------------------------------
//
// Slab layout (single source of truth for slab-pool sizing), and small
// lock-synchronized accessors.


AssembledFrameAllocator::SlabLayout
AssembledFrameAllocator::get_layout(long nfreq, long time_samples_per_chunk)
{
    xassert(nfreq > 0);
    xassert(time_samples_per_chunk > 0);
    xassert((time_samples_per_chunk % 256) == 0);   // implies even (int4 byte count exact)

    long mpc = xdiv(time_samples_per_chunk, 256);
    constexpr long nalign = constants::bytes_per_gpu_cache_line;

    SlabLayout layout;
    layout.scales_offsets_nbytes = nfreq * mpc * 4;                  // (nfreq, mpc, 2) float16
    // Individually cache-line-align each array: scales_offsets starts at slab
    // offset 0 (slab base is cache-line aligned by the SlabAllocator), and data
    // starts at the next cache-line boundary so its base is aligned too.
    layout.data_offset           = align_up(layout.scales_offsets_nbytes, nalign);
    layout.data_nbytes           = nfreq * xdiv(time_samples_per_chunk, 2);  // (nfreq, ntime) int4
    layout.slab_nbytes           = layout.data_offset + layout.data_nbytes;
    return layout;
}


long AssembledFrameAllocator::slab_nbytes(long nfreq, long time_samples_per_chunk)
{
    return get_layout(nfreq, time_samples_per_chunk).slab_nbytes;
}


long AssembledFrameAllocator::get_nfreq() const
{
    lock_guard<mutex> lg(lock);
    return nfreq;
}


std::vector<long> AssembledFrameAllocator::get_beam_ids() const
{
    lock_guard<mutex> lg(lock);
    return beam_ids;   // copied under the lock
}


// -------------------------------------------------------------------------------------------------
//
// initialize_metadata() - Entry point


void AssembledFrameAllocator::initialize_metadata(const XEngineMetadata &metadata_)
{
    {
        unique_lock<mutex> guard(lock);
        _throw_if_stopped("AssembledFrameAllocator::initialize_metadata");
    }

    try {
        _initialize_metadata(metadata_);
    } catch (...) {
        this->stop(current_exception());
        throw;
    }
}


void AssembledFrameAllocator::_initialize_metadata(const XEngineMetadata &metadata_)
{
    // Validate the supplied metadata up-front. validate() catches any internal
    // inconsistency before we let the metadata propagate into per-frame state.
    metadata_.validate();

    unique_lock<mutex> guard(lock);

    // Re-check the stopped flag under the lock: the wrapper's gate ran
    // before this body relocked, so a stop() can land in between -- without
    // this, initialize_metadata would silently succeed on a stopped
    // allocator. (Mirrors _initialize_initial_chunk.)
    _throw_if_stopped("AssembledFrameAllocator::initialize_metadata");

    if (!metadata_is_initialized) {
        // First call (from any caller): establish parameters. Make a private
        // shared_ptr copy of the metadata so the allocator (and all frames it
        // creates) keeps it alive independently of any caller-owned object.
        auto md = std::make_shared<XEngineMetadata>(metadata_);

        // Project away freq_channels: the canonical metadata is the consensus
        // across senders, and freq_channels deliberately differs per sender
        // (each sender sends a subset). Storing one arbitrary sender's
        // freq_channels would be more confusing than helpful for downstream
        // callers of get_metadata(). (Mirrors the pre-consolidation behavior
        // that Receiver::_read_yaml used to do on its own private copy.)
        md->freq_channels.clear();

        nfreq = md->get_total_nfreq();
        beam_ids = md->beam_ids;
        metadata = md;
        metadata_is_initialized = true;
    }
    else {
        // Subsequent calls: validate the supplied metadata against the
        // canonical copy via XEngineMetadata::check_sender_consistency.
        // (time_samples_per_chunk is fixed at allocator construction time;
        // there's nothing further to cross-check here.)
        xassert(metadata);  // non-null invariant after first init
        XEngineMetadata::check_sender_consistency(*metadata, metadata_);
    }

    // Notify the worker thread's init gate and any get_metadata() waiters
    // (one-shot latch -> notify_all, after releasing the lock).
    guard.unlock();
    metadata_cv.notify_all();
}


// -------------------------------------------------------------------------------------------------
//
// initialize_initial_chunk() / wait_for_initial_chunk() - Entry points


long AssembledFrameAllocator::initialize_initial_chunk(long target_time_chunk)
{
    {
        unique_lock<mutex> guard(lock);
        _throw_if_stopped("AssembledFrameAllocator::initialize_initial_chunk");
    }

    try {
        return _initialize_initial_chunk(target_time_chunk);
    } catch (...) {
        this->stop(current_exception());
        throw;
    }
}


long AssembledFrameAllocator::_initialize_initial_chunk(long target_time_chunk)
{
    // Defensive guard: chunk indices are non-negative. A negative value here
    // most likely indicates a uint64_t->long wrap on a bogus seq from the
    // wire, so we'd rather throw than silently store a meaningless offset.
    xassert_ge(target_time_chunk, 0L);

    unique_lock<mutex> guard(lock);
    _throw_if_stopped("AssembledFrameAllocator::_initialize_initial_chunk");

    bool just_set = false;
    if (!initial_chunk_set) {
        // First call from any caller establishes the canonical value.
        initial_time_chunk = target_time_chunk;
        initial_chunk_set = true;
        just_set = true;

        // Seed all chunk-indexed counters to initial_time_chunk. (See
        // AssembledFrame.hpp -- these counters track "time chunk index of
        // the next set in queue / received / created" and are only
        // meaningful once initial_chunk_set is true.)
        queue_start_chunk_index = initial_time_chunk;
        first_unreceived_chunk_index = initial_time_chunk;
    }
    long ret = initial_time_chunk;
    guard.unlock();

    // Notify the worker thread's init gate and any wait_for_initial_chunk()
    // waiters (one-shot latch -> notify_all).
    if (just_set)
        chunk_cv.notify_all();
    return ret;
}


long AssembledFrameAllocator::wait_for_initial_chunk()
{
    unique_lock<mutex> guard(lock);
    for (;;) {
        _throw_if_stopped("AssembledFrameAllocator::wait_for_initial_chunk");
        if (initial_chunk_set)
            return initial_time_chunk;
        chunk_cv.wait(guard);
    }
}


// -------------------------------------------------------------------------------------------------
//
// get_metadata() - Entry point


shared_ptr<const XEngineMetadata> AssembledFrameAllocator::get_metadata(bool blocking)
{
    unique_lock<mutex> guard(lock);
    for (;;) {
        _throw_if_stopped("AssembledFrameAllocator::get_metadata");
        if (metadata)
            return metadata;
        if (!blocking)
            return nullptr;
        metadata_cv.wait(guard);
    }
}


// -------------------------------------------------------------------------------------------------
//
// get_frame_set() - Entry point


shared_ptr<AssembledFrameSet> AssembledFrameAllocator::get_frame_set(long time_chunk_index)
{
    {
        unique_lock<mutex> guard(lock);
        _throw_if_stopped("AssembledFrameAllocator::get_frame_set");
    }

    try {
        return _get_frame_set(time_chunk_index);
    } catch (...) {
        this->stop(current_exception());
        throw;
    }
}


shared_ptr<AssembledFrameSet> AssembledFrameAllocator::_get_frame_set(long time_chunk_index)
{
    unique_lock<mutex> guard(lock);

    // Stopped-check FIRST, re-checked under the lock (the wrapper's gate ran
    // before this body relocked): a caller racing a stop() should get the
    // saved root-cause error, not a misleading "called before
    // initialize_metadata()" init-state error.
    _throw_if_stopped("AssembledFrameAllocator::get_frame_set");

    // Check that both initialization phases have completed. The allocator
    // needs metadata (for nfreq, beam_ids) AND the initial_time_chunk
    // (which stamps set->time_chunk_index) before it can mint sets.
    if (!metadata_is_initialized) {
        stringstream ss;
        ss << "AssembledFrameAllocator::get_frame_set(time_chunk_index=" << time_chunk_index
           << ") called before any initialize_metadata()";
        throw runtime_error(ss.str());
    }
    if (!initial_chunk_set) {
        stringstream ss;
        ss << "AssembledFrameAllocator::get_frame_set(time_chunk_index=" << time_chunk_index
           << ") called before any initialize_initial_chunk()";
        throw runtime_error(ss.str());
    }

    if (time_chunk_index < initial_time_chunk) {
        stringstream ss;
        ss << "AssembledFrameAllocator::get_frame_set(): requested time_chunk_index="
           << time_chunk_index << " precedes initial_time_chunk=" << initial_time_chunk;
        throw runtime_error(ss.str());
    }

    for (;;) {
        if (is_stopped)
            _throw_if_stopped("AssembledFrameAllocator::get_frame_set");

        // Recompute queue_pos in each iteration of the for-loop, since the
        // queue can advance while the lock is dropped in queue_cv.wait().
        long queue_pos = time_chunk_index - queue_start_chunk_index;
        long queue_size = frame_set_queue.size();

        if (queue_pos < 0) {
            // The requested set already received its num_consumers receipts
            // and was evicted. Reachable only under a violation of the
            // receipt contract (see get_frame_set() doc in
            // AssembledFrame.hpp) -- e.g. a double-request by some OTHER
            // caller inflated the receipt count, evicting the set
            // prematurely.
            stringstream ss;
            ss << "AssembledFrameAllocator::get_frame_set(): requested time_chunk_index="
               << time_chunk_index << " was already evicted (queue starts at chunk "
               << queue_start_chunk_index << "); see the receipt contract in AssembledFrame.hpp";
            throw runtime_error(ss.str());
        }

        // Catches requests that skip past the next-uncreated chunk (a
        // receipt-contract violation that would otherwise deadlock).
        xassert(queue_pos <= queue_size);

        if (queue_pos == queue_size) {
            // Set is not in the queue yet -- wait for the worker thread to
            // create it, then re-check from the top of the loop (the lock
            // was dropped in queue_cv.wait()).
            queue_cv.wait(guard);
            continue;
        }

        // Get set and increment its receipt count.
        auto &[set_ref, num_received] = frame_set_queue[queue_pos];
        num_received++;

        // Receipt-contract check: in legitimate flows a count can never
        // exceed num_consumers (receipts are non-increasing along the queue
        // -- each consumer requests consecutive indices -- so only the front
        // can reach num_consumers, and it is popped in this same critical
        // section). A double-request that pushed a NON-front set past
        // num_consumers would otherwise jam the queue forever: the pop test
        // below is an equality, so the overshot set would never pop -- an
        // undetectable deadlock. Crash loudly at the offending caller
        // instead (the throw stops the allocator, via the entry-point
        // wrapper in get_frame_set).
        xassert_le(num_received, num_consumers);

        // IMPORTANT: Make a copy of the shared_ptr BEFORE popping from queue!
        // set_ref becomes a dangling reference after pop_front().
        auto result = set_ref;

        // Advance first_unreceived_chunk_index (under the receipt contract,
        // the fastest consumer's next chunk index).
        if (time_chunk_index + 1 > first_unreceived_chunk_index) {
            first_unreceived_chunk_index = time_chunk_index + 1;
            lowmem_cv.notify_all();  // Wake up block_until_low_memory() if waiting
        }

        // Pop sets from front that all consumers have received. If any set
        // was popped, wake the worker thread, whose slot_cv predicate tests
        // the queue size. (notify_one is sound: the worker is structurally
        // the only slot_cv waiter.)
        bool popped = false;
        while (!frame_set_queue.empty() && (frame_set_queue.front().second == num_consumers)) {
            frame_set_queue.pop_front();
            queue_start_chunk_index++;
            popped = true;
        }
        if (popped)
            slot_cv.notify_one();

        return result;
    }
}


// -------------------------------------------------------------------------------------------------
//
// num_free_frames() and num_total_frames()


long AssembledFrameAllocator::num_free_frames(bool permissive) const
{
    // In dummy mode: throw or return 0. (is_dummy_mode is set once in constructor, safe without lock.)
    if (is_dummy_mode) {
        if (permissive)
            return 0;
        throw runtime_error("AssembledFrameAllocator::num_free_frames(): not available in dummy mode");
    }

    // Check that metadata has been set, then compute num_preinitialized
    // under lock. (Metadata alone does not imply the slab pool exists:
    // the worker only starts allocating once initial_chunk_set is ALSO
    // true, so the num_free_slabs() call below can still throw its
    // "pool not created" error. Forwarding 'permissive' into it makes
    // that term a best-effort 0 instead -- this is the window the
    // GetStatus RPC hits when senders have handshook but not yet
    // streamed data.)
    unique_lock<mutex> guard(lock);

    if (!metadata_is_initialized) {
        if (permissive)
            return 0;
        throw runtime_error("AssembledFrameAllocator::num_free_frames(): allocator not initialized");
    }

    // Public API is in *frame* units; internally we count sets (= chunks)
    // and multiply by nbeams. nbeams is fixed once metadata is set.
    long nbeams = long(beam_ids.size());
    long queue_end_chunk_index = queue_start_chunk_index + long(frame_set_queue.size());
    long num_preinitialized_chunks = queue_end_chunk_index - first_unreceived_chunk_index;
    long num_preinitialized_frames = num_preinitialized_chunks * nbeams;
    guard.unlock();

    return num_preinitialized_frames + slab_allocator->num_free_slabs(permissive);
}


long AssembledFrameAllocator::num_total_frames(bool blocking) const
{
    // Gate dummy mode HERE rather than delegating: num_total_slabs() is a
    // SLAB ENTRY POINT, so its dummy-mode throw would STOP the slab
    // allocator (strict stoppable-class policy) -- and thence this whole
    // allocator, via the worker's next get_slab() -- as a side effect of an
    // informational query. This AFA-local throw stops nothing
    // (num_total_frames is an accessor, like num_free_frames above).
    if (is_dummy_mode)
        throw runtime_error("AssembledFrameAllocator::num_total_frames(): not available in dummy mode");

    // Non-dummy delegation is still a slab entry point: with blocking=false,
    // a call BEFORE the worker's first allocation creates the pool throws
    // and stops the allocators. See the warning at the hpp declaration.
    return slab_allocator->num_total_slabs(blocking);
}


// -------------------------------------------------------------------------------------------------
//
// block_until_low_memory() - Entry point


void AssembledFrameAllocator::block_until_low_memory(long nframe_threshold)
{
    try {
        _block_until_low_memory(nframe_threshold);
    } catch (...) {
        this->stop(current_exception());
        throw;
    }
}


void AssembledFrameAllocator::_block_until_low_memory(long nframe_threshold)
{
    unique_lock<mutex> guard(lock);

    for (;;) {
        _throw_if_stopped("AssembledFrameAllocator::block_until_low_memory");
        guard.unlock();

        // Block until slab allocator is empty (all slabs allocated).
        //
        // Liveness note: this returns promptly whenever memory pressure is
        // real, because the worker thread grabs every freed slab while
        // frame_set_queue is below its bound. If free slabs persist
        // instead, the worker must be parked on a full queue, so
        // num_preinitialized equals the queue bound -- which exceeds the
        // reaper's nframe_threshold by the static_assert relating
        // assembled_frame_allocator_queue_size to reaper_lowmem_chunks in
        // constants.hpp -- i.e. memory is genuinely not low, and blocking
        // here is the desired behavior.
        slab_allocator->block_until_empty();
        
        // Check num_preinitialized under lock.
        guard.lock();
        _throw_if_stopped("AssembledFrameAllocator::block_until_low_memory");
        
        // num_preinitialized is in *frame* units (to match the
        // nframe_threshold argument's units), so we multiply chunks by
        // nbeams here, same as num_free_frames().
        long nbeams = long(beam_ids.size());
        long queue_end_chunk_index = queue_start_chunk_index + long(frame_set_queue.size());
        long num_preinitialized_chunks = queue_end_chunk_index - first_unreceived_chunk_index;
        long num_preinitialized = num_preinitialized_chunks * nbeams;

        if (num_preinitialized <= nframe_threshold)
            return;

        // Wait for num_preinitialized to decrease (first_unreceived advance
        // in _get_frame_set), then loop back through block_until_empty().
        lowmem_cv.wait(guard);
    }
}


}  // namespace pirate
