#include "../include/pirate/DedispersionTree.hpp"
#include "../include/pirate/inlines.hpp"  // pow2()
#include "../include/pirate/utils.hpp"    // integer_log2()

using namespace std;
using namespace ksgpu;

namespace pirate {
#if 0
}  // editor auto-indent
#endif


// Derived quantities. See the doc-comments in DedispersionTree.hpp. (Everything else about a
// tree -- construction, yaml I/O, decoding out_argmax tokens, the subband index mappings --
// lives in DedispersionPlan.)


// Number of toplevel tree-freq channels spanned by one coarse-freq channel of 'tree'.
//
// This is the same for every tree of a config, including early-trigger trees: the early
// trigger removes exactly as many subband levels as it removes tree rank (see
// FrequencySubbands::restrict_subband_counts), so the channel width never moves. The
// computation below deliberately does NOT rely on that -- it uses only the tree's own
// members, so it stays correct for a tree considered on its own.
static long _toplevel_channels_per_coarse_channel(const DedispersionTree &tree)
{
    // Rank of the underlying dedispersion, i.e. this tree searches the low 2^rr of the
    // toplevel band. (Early triggers restrict the search to a prefix; time-downsampling
    // leaves the freq axis alone.)
    long rr = tree.toplevel_tree_rank() - tree.early_trigger_level;
    return pow2(rr - tree.frequency_subbands.pf_rank);
}


long DedispersionTree::n_to_toplevel_flo(long n) const
{
    return this->frequency_subbands.n_to_flo.at(n) * _toplevel_channels_per_coarse_channel(*this);
}


long DedispersionTree::n_to_toplevel_fhi(long n) const
{
    return this->frequency_subbands.n_to_fhi.at(n) * _toplevel_channels_per_coarse_channel(*this);
}


long DedispersionTree::xdm_rank() const
{
    // The DedispersionPlan constructor pins dm_downsampling = pow2(dd_rank1()).
    return integer_log2(this->dm_downsampling) - this->frequency_subbands.pf_rank;
}


}  // namespace pirate
