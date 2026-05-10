#ifndef _PIRATE_XENGINE_METADATA_HPP
#define _PIRATE_XENGINE_METADATA_HPP

#include <array>
#include <vector>
#include <string>

namespace YAML { class Emitter; }      // #include <yaml-cpp/yaml.h>
namespace pirate { struct YamlFile; }  // #include <pirate/YamlFile.hpp>


namespace pirate {
#if 0
}  // editor auto-indent
#endif


// XEngineMetadata: documents file format for communication between X-engine and FRB nodes.
//
// This metadata is used in two contexts:
//
//   1. Every X-engine node sends this file to every FRB node, at the beginning
//      of the TCP stream.
//
//   2. As a configuration file for the "fake X-engine" used for testing.
//
// Reference: pirate/configs/xengine/xengine_metadata_*.yml

struct XEngineMetadata
{
    // Version number of the metadata format.
    // Initialized to 0 (an invalid value) so default-constructed instances must be
    // explicitly populated -- guards against forgetting to set 'version' or other fields.
    long version = 0;

    // ---- Frequency channels ----
    //
    // The observed frequency band is divided into "zones".
    // Within each zone, all frequency channels have the same width, but the
    // channel width may differ between zones. For example:
    //
    //   zone_nfreq = {N}      zone_freq_edges={400,800}      one zone, channel width (400/N)
    //   zone_nfreq = {2*N,N}  zone_freq_edges={400,600,800}  width (100/N), (200/N) in lower/upper band

    std::vector<long> zone_nfreq;         // length (nzones)
    std::vector<double> zone_freq_edges;  // length (nzones+1), monotone increasing, in MHz.

    // Optional: which frequency channels are present?
    // A list of distinct integers 0 <= (channel_id) < (total frequency channels).
    // Only makes sense in "context 1" (see above), to indicate which frequency channels
    // are sent by a particular X-engine node.
    std::vector<long> freq_channels;

    // ---- Beams ----
    //
    // 'beamset' is an opaque integer identifier for this set of beams.
    // 'beam_ids' (length nbeams) holds opaque integer beam identifiers.
    // 'beam_positions_x' and 'beam_positions_y' (length nbeams) are direction cosines
    // b.x and b.y in the grid frame; the grid frame is defined by orthogonal unit vectors
    // along (or close to) the telescope grid axes.

    long beamset = 0;
    std::vector<long> beam_ids;
    std::vector<double> beam_positions_x;
    std::vector<double> beam_positions_y;

    // ---- Timekeeping ----
    //
    // The X-engine uses an FPGA sequence number ('seq') to keep track of time.
    // To keep time in pirate, we need:
    //   unix_ns_at_seq_0:         UNIX time at seq=0, in nanoseconds.
    //   dt_ns_per_seq:            nanoseconds per seq tick.
    //   seq_per_frb_time_sample:  seq ticks per FRB time sample.
    //
    // These fields are currently opaque to pirate (passed through the protocol
    // for downstream code).

    long unix_ns_at_seq_0 = 0;
    long dt_ns_per_seq = 0;
    long seq_per_frb_time_sample = 0;

    // ---- Telescope alignment / localization ----
    //
    // 'tel_origin_itrs_*' give the telescope position on Earth in degrees.
    // 'tel_grid_x_axis' and 'tel_grid_y_axis' are unit vectors aligned with the dish
    // grid, expressed in topocentric coordinates (East, North, Up).
    // 'tel_dish_elev_axis' is the axis around which the dishes pivot (positive in the
    // east direction); 'tel_dish_vert_axis' is the dish "up/zenith" direction (the
    // direction of zero coelevation). Both are expressed in topocentric coordinates.
    // 'tel_dish_coelev_deg' is the dish pointing angle (angle from vertical, north positive).
    // 'tel_dish_separation_{x,y}_m' are the dish separations along the grid axes in meters.

    double tel_origin_itrs_lat_deg = 0.0;
    double tel_origin_itrs_lon_deg = 0.0;
    std::array<double, 3> tel_grid_x_axis    {};
    std::array<double, 3> tel_grid_y_axis    {};
    std::array<double, 3> tel_dish_elev_axis {};
    std::array<double, 3> tel_dish_vert_axis {};
    double tel_dish_coelev_deg = 0.0;
    double tel_dish_separation_x_m = 0.0;
    double tel_dish_separation_y_m = 0.0;

    // ---- Noise variance (temporary kludge) ----
    //
    // Per-frequency-zone noise variance, length len(zone_nfreq). The FRB server
    // currently assumes the noise is mean-zero, uncorrelated between samples,
    // constant in time, and depends only on frequency zone. In the YAML, a
    // scalar may be specified instead of a list -- in that case all zones are
    // assigned the same variance (broadcast at parse time, so this field is
    // always length-nzones in C++). This will go away in a future revision.

    std::vector<double> noise_variance;

    // -------------------------------- Member functions --------------------------------

    // Validate that all members have been initialized with sensible values.
    void validate() const;

    // Returns sum of zone_nfreq (i.e. total number of frequency channels across all zones).
    long get_total_nfreq() const;

    // Number of beams (= length of beam_ids).
    long get_nbeams() const { return long(beam_ids.size()); }

    // Write in YAML format.
    // If 'verbose' is true, include comments explaining the meaning of each field.
    void to_yaml(YAML::Emitter &emitter, bool verbose = false) const;
    std::string to_yaml_string(bool verbose = false) const;

    // Construct from YAML.
    static XEngineMetadata from_yaml_string(const std::string &s);
    static XEngineMetadata from_yaml_file(const std::string &filename);
    static XEngineMetadata from_yaml(const YamlFile &file);

    // Check that two XEngineMetadata objects (from different senders) have consistent fields.
    // Checks zone_nfreq, zone_freq_edges, beamset, beam_ids, beam_positions_x/y, all timekeeping
    // fields, all tel_* fields, and noise_variance. Does NOT check freq_channels (which legitimately
    // differ across X-engine nodes). Throws exception on mismatch.
    static void check_sender_consistency(const XEngineMetadata &ref, const XEngineMetadata &m);
};


}  // namespace pirate

#endif // _PIRATE_XENGINE_METADATA_HPP
