#include <iostream>
#include "../include/pirate/DedispersionPlan.hpp"
#include "../include/pirate/CoalescedDdKernel2.hpp"

using namespace std;

namespace pirate {
#if 0
}  // editor auto-indent
#endif


// Called by 'python -m pirate_frb scratch'. Intended for quick throwaway tests.
void scratch()
{
    cout << "pirate::scratch() called -- this is a place for quick throwaway tests" << endl;

    for (int i = 0; i < 200; i++) {
        cout << "\n    *** iteration " << i << " ***\n\n";
        DedispersionConfig::RandomArgs args;
        args.max_rank = 16;
        args.gpu_valid = true;
        args.verbose = true;

        DedispersionConfig config = DedispersionConfig::make_random(args);
        config.print(cout, 4);

        auto plan = make_shared<DedispersionPlan> (config);
        xassert_eq(long(plan->stage2_dd_kernel_params.size()), plan->stage2_ntrees);
        xassert_eq(long(plan->stage2_pf_params.size()), plan->stage2_ntrees);

        if (!args.gpu_valid)
            continue;

        for (long i = 0; i < plan->stage2_ntrees; i++) {
            // Check that we can construct the cdd2 kernel without failing an assert.
            CoalescedDdKernel2 cdd2(
                plan->stage2_dd_kernel_params.at(i),
                plan->stage2_pf_params.at(i)
            );
        }
    }
}


}  // namespace pirate

