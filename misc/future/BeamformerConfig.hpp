namespace pirate {
#if 0
}  // pacify editor auto-indent
#endif


// -------------------------------------------------------------------------------------------------
//
// Stream format:
//
//   - Serialized BfGlobalConfig (around ~64KB?)
//   - Serialized BfLocalConfig (around ~4KB?)
//
//     Format: each of these is a multiple of 4096 bytes (after padding).
//       Bytes 0:4 are int32 length of serialized data, before padding.
//       Bytes 4:8 are int32 length of serialized data, after padding.
//       Bytes 8:12 are int32 version number.
//
//   - For each chunk:
//
//       - '', a float16 array with shape:
//
//            (BfLocalConfig::nbeams,
//
//       - 'data' array, an intD array where D = BfGlobalConfig::bit_depth, with shape:
//
//             (BfLocalConfig::nbeams,
//              BfLocalConfig::nfreq_upchannelized,
//              BfGlobalConfig::nt_chunk)
//
//          Assuming shape (200,256,256) and D=4, each 'data' array is 6.25 MB.
//
// Note: protocol 


struct RtGlobalConfig
{
    // To be incremented when 'struct BeamformerConfig' changes
    const int version = 1;

    // Beam info
    int nbeams = 0;
    vector<int> beam_xcoord_arcsec;  // length nbeams
    vector<int> beam_ycoord_arcsec;  // length nbeams

    // FIXME I think it will make sense to add fields here, keeping track of
    // assignment of beams to FRB search nodes. (Or should this info be elsewhere?)
    // For now, we assume that there is only one FRB search node!

    // FIXME in the future we may decide to use a beam-dependent max frequency.
    // In this case we would add more fields here.

    // Frequency channel info.
    int nfreq_coarse = 0;                 // E.g. 2048 for iceboard
    vector<int> upchannelization_factor;  // length nfreq_coarse

    // FIXME I think it will make sense to add fields here, keeping track of
    // assignment of coarse frequencies to GPUs. (Or should this info be elsewhere?)
    // For now, assume round-robin assignment.

    // Time sample info.
    int fpga_counts_per_time_sample = 0;
    int bit_depth = 0;
    int nt_chunk = 0;

    // If frame_size > 0, then both sender and receiver should call 
    int frame_size = 0;
    
    int align_nbytes_inner = 0;
    int align_nbytes_outer = 0;
};


// -------------------------------------------------------------------------------------------------


struct RtLocalConfig
{
    const int version = 1;

    // Beams received by current FRB search node.
    int nbeams = 0;
    vector<int> beam_ids;

    // Frequencies sent by current correlator GPU.
    int nfreq = 0;
    vector<int> coarse_freq_ids;

    uint64_t initial_fpga_count = 0;
};


// -------------------------------------------------------------------------------------------------


struct RtControlBlock
{
    static constexpr int max_coarse_freq_per_stream = 16;
    static constexpr int max_beams_per_stream = 256;
    static constexpr int control_block_nbytes = 4096;

    const int version = 1;
    // FIXME I may add a hash of the RtStreamConfig, just as a check

    // Beams sent in this stream
    int nbeams = 0;
    int beam_id[max_beams_per_stream];
    
    // Frequencies sent in this stream
    int nfreq_coarse = 0;
    int nfreq_upchannelized = 0;
    int ifreq_coarse[max_coarse_freq_per_stream];
    int upchannelization_factor[max_coarse_freq_per_stream];

    uint64_t fpga_count0 = 0;  // must 
    uint64_t timestamp = 0;    // flexible and currently unused
    uint32_t reboot_id = 0;    // flexible and currently unused
};
