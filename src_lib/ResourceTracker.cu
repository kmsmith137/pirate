#include "../include/pirate/ResourceTracker.hpp"
#include "../include/pirate/BumpAllocator.hpp"
#include "../include/pirate/inlines.hpp"

#include <sstream>
#include <stdexcept>
#include <vector>
#include <algorithm>

namespace pirate {
#if 0
}  // editor auto-indent
#endif


void ResourceTracker::_update_dict(Dict &d, const std::string &key, long value)
{
    if (value == 0)
        return;
    if (key.empty())
        throw std::runtime_error("ResourceTracker::_update_dict: key is empty");

    auto it = d.find(key);
    if (it != d.end())
        it->second += value;
    else
        d[key] = value;
}


void ResourceTracker::add_kernel(const std::string &key, long nbytes)
{
    _update_dict(kernel_launches, key, 1);
    _update_dict(gmem_bw_nbytes, key, nbytes);
}


void ResourceTracker::add_memcpy_h2g(const std::string &key, long nbytes)
{
    if (nbytes == 0)
        return;
    
    _update_dict(memcpy_h2g_calls, key, 1);
    _update_dict(h2g_bw_nbytes, key, nbytes);
    _update_dict(gmem_bw_nbytes, key, nbytes);
    _update_dict(hmem_bw_nbytes, key, nbytes);
}


void ResourceTracker::add_memcpy_g2h(const std::string &key, long nbytes)
{
    if (nbytes == 0)
        return;
    
    _update_dict(memcpy_g2h_calls, key, 1);
    _update_dict(g2h_bw_nbytes, key, nbytes);
    _update_dict(gmem_bw_nbytes, key, nbytes);
    _update_dict(hmem_bw_nbytes, key, nbytes);
}


void ResourceTracker::add_gmem_bw(const std::string &key, long nbytes)
{
    _update_dict(gmem_bw_nbytes, key, nbytes);
}


void ResourceTracker::add_hmem_bw(const std::string &key, long nbytes)
{
    _update_dict(hmem_bw_nbytes, key, nbytes);
}


void ResourceTracker::add_gmem_footprint(const std::string &key, long nbytes, bool align)
{
    if (align)
        nbytes = align_up(nbytes, BumpAllocator::nalign);
    _update_dict(gmem_footprint_nbytes, key, nbytes);
}


void ResourceTracker::add_hmem_footprint(const std::string &key, long nbytes, bool align)
{
    if (align)
        nbytes = align_up(nbytes, BumpAllocator::nalign);
    _update_dict(hmem_footprint_nbytes, key, nbytes);
}


long ResourceTracker::get_gmem_bw(const std::string &key) const
{
    if (key.empty()) {
        long total = 0;
        for (const auto &p : gmem_bw_nbytes)
            total += p.second;
        return total;
    }
    auto it = gmem_bw_nbytes.find(key);
    if (it == gmem_bw_nbytes.end())
        throw std::runtime_error("ResourceTracker::get_gmem_bw: key not found: " + key);
    return it->second;
}


long ResourceTracker::get_gmem_footprint(const std::string &key) const
{
    if (key.empty()) {
        long total = 0;
        for (const auto &p : gmem_footprint_nbytes)
            total += p.second;
        return total;
    }
    auto it = gmem_footprint_nbytes.find(key);
    if (it == gmem_footprint_nbytes.end())
        throw std::runtime_error("ResourceTracker::get_gmem_footprint: key not found: " + key);
    return it->second;
}


long ResourceTracker::get_hmem_footprint(const std::string &key) const
{
    if (key.empty()) {
        long total = 0;
        for (const auto &p : hmem_footprint_nbytes)
            total += p.second;
        return total;
    }
    auto it = hmem_footprint_nbytes.find(key);
    if (it == hmem_footprint_nbytes.end())
        throw std::runtime_error("ResourceTracker::get_hmem_footprint: key not found: " + key);
    return it->second;
}


ResourceTracker ResourceTracker::clone() const
{
    return *this;
}


ResourceTracker &ResourceTracker::operator+=(const ResourceTracker &x)
{
    for (const auto &p : x.gmem_bw_nbytes)
        _update_dict(gmem_bw_nbytes, p.first, p.second);
    for (const auto &p : x.hmem_bw_nbytes)
        _update_dict(hmem_bw_nbytes, p.first, p.second);
    for (const auto &p : x.h2g_bw_nbytes)
        _update_dict(h2g_bw_nbytes, p.first, p.second);
    for (const auto &p : x.g2h_bw_nbytes)
        _update_dict(g2h_bw_nbytes, p.first, p.second);
    for (const auto &p : x.memcpy_h2g_calls)
        _update_dict(memcpy_h2g_calls, p.first, p.second);
    for (const auto &p : x.memcpy_g2h_calls)
        _update_dict(memcpy_g2h_calls, p.first, p.second);
    for (const auto &p : x.kernel_launches)
        _update_dict(kernel_launches, p.first, p.second);
    for (const auto &p : x.gmem_footprint_nbytes)
        _update_dict(gmem_footprint_nbytes, p.first, p.second);
    for (const auto &p : x.hmem_footprint_nbytes)
        _update_dict(hmem_footprint_nbytes, p.first, p.second);

    return *this;
}


// -------------------------------------------------------------------------------------------------
//
// Helper function for to_yaml()


// Emit dict entries to YAML.
// The 'scale' factor converts raw integer counts to display values.
// The 'unit' string is appended to each value (e.g. "GB/s", "calls/s", "GiB").
static void _emit_dict(YAML::Emitter &emitter, const std::string &name,
                       const ResourceTracker::Dict &d, double scale,
                       const std::string &unit, bool fine_grained)
{
    if (d.empty())
        return;

    // Compute sum and build sorted vector (by decreasing value, then alphabetically by key).
    long total = 0;
    std::vector<std::pair<std::string, long>> sorted_entries(d.begin(), d.end());
    
    for (const auto &p : sorted_entries)
        total += p.second;

    std::sort(sorted_entries.begin(), sorted_entries.end(),
        [](const auto &a, const auto &b) {
            if (a.second != b.second)
                return a.second > b.second;  // decreasing by value
            return a.first < b.first;         // then alphabetical by key
        });

    double scaled_total = double(total) * scale;

    std::stringstream ss;
    ss << scaled_total << " " << unit;

    if (fine_grained)
        emitter << YAML::Newline << YAML::Newline;
    
    emitter << YAML::Key << name;

    if (!fine_grained) {
        emitter << YAML::Value << ss.str();
        return;
    }

    emitter << YAML::Value << YAML::BeginMap;
    emitter << YAML::Key << "total" << YAML::Value << ss.str();

    for (const auto &p : sorted_entries) {
        double scaled_val = double(p.second) * scale;
        std::stringstream ss2;
        ss2 << scaled_val << " " << unit;
        emitter << YAML::Key << p.first << YAML::Value << ss2.str();
    }

    emitter << YAML::EndMap;
}


// -------------------------------------------------------------------------------------------------


void ResourceTracker::to_yaml(YAML::Emitter &emitter, double multiplier, bool fine_grained) const
{
    constexpr double GB = 1.0e9;              // 10^9 bytes
    constexpr double GiB = 1073741824.0;      // 2^30 bytes
    
    double bw_scale = multiplier / GB;        // bytes -> GB/s
    double count_scale = multiplier;          // counts -> counts/s
    double footprint_scale = 1.0 / GiB;       // bytes -> GiB

    emitter << YAML::BeginMap;

    // Bandwidth stats
    _emit_dict(emitter, "gmem_bw", gmem_bw_nbytes, bw_scale, "GB/s", fine_grained);
    _emit_dict(emitter, "hmem_bw", hmem_bw_nbytes, bw_scale, "GB/s", fine_grained);
    _emit_dict(emitter, "h2g_bw", h2g_bw_nbytes, bw_scale, "GB/s", fine_grained);
    _emit_dict(emitter, "g2h_bw", g2h_bw_nbytes, bw_scale, "GB/s", fine_grained);

    // Count stats
    _emit_dict(emitter, "memcpy_h2g", memcpy_h2g_calls, count_scale, "memcopies/s", fine_grained);
    _emit_dict(emitter, "memcpy_g2h", memcpy_g2h_calls, count_scale, "memcopies/s", fine_grained);
    _emit_dict(emitter, "kernel_launches", kernel_launches, count_scale, "launches/s", fine_grained);

    // Footprint stats
    _emit_dict(emitter, "gmem_footprint", gmem_footprint_nbytes, footprint_scale, "GiB", fine_grained);
    _emit_dict(emitter, "hmem_footprint", hmem_footprint_nbytes, footprint_scale, "GiB", fine_grained);

    emitter << YAML::EndMap;
}


std::string ResourceTracker::to_yaml_string(double multiplier, bool fine_grained) const
{
    YAML::Emitter emitter;
    to_yaml(emitter, multiplier, fine_grained);
    return std::string(emitter.c_str());
}


}  // namespace pirate

