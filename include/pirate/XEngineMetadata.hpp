#ifndef _PIRATE_XENGINE_METADATA_HPP
#define _PIRATE_XENGINE_METADATA_HPP

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
// This struct is used in two contexts:
//   1. Every X-engine node sends this file to every FRB node, at the beginning
//      of the TCP stream.
//   2. As a configuration file for the "fake correlator" used for testing.

struct XEngineMetadata
{
    // Version number of the metadata format.
    long version = 0;

    // Frequency channels. The observed frequency band is divided into "zones".
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

    // Number of beams.
    long nbeams = 0;

    // Beam identifiers (opaque integer identifiers).
    // If empty when read from YAML, defaults to [ 0, 1, ..., (nbeams-1) ].
    std::vector<long> beam_ids;

    // -------------------------------- Member functions --------------------------------

    // Validate that all members have been initialized with sensible values.
    void validate() const;

    // Returns sum of zone_nfreq (i.e. total number of frequency channels across all zones).
    long get_total_nfreq() const;

    // Write in YAML format.
    // If 'verbose' is true, include comments explaining the meaning of each field.
    void to_yaml(YAML::Emitter &emitter, bool verbose = false) const;
    std::string to_yaml_string(bool verbose = false) const;

    // Construct from YAML.
    static XEngineMetadata from_yaml_string(const std::string &s);
    static XEngineMetadata from_yaml_file(const std::string &filename);
    static XEngineMetadata from_yaml(const YamlFile &file);
};


}  // namespace pirate

#endif // _PIRATE_XENGINE_METADATA_HPP
