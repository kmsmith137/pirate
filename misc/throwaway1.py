"""Throwaway script: build a slow_avar PfAvarApproximation from chord_sb2_et.yml with
random freq_variances, and pickle its picklable members to pirate/misc/throwaway1.pkl.

(The 'plan' and 'tree_fs' members are pybind11 objects and can't be pickled; the
freq_variances used are included in the dict, so the run is reconstructible.)
"""

import os
import time
import pickle
import numpy as np

from pirate_frb import DedispersionConfig, DedispersionPlan
from pirate_frb.slow_avar.PfVariance import PfAvarApproximation


script_dir = os.path.dirname(os.path.abspath(__file__))   # pirate/misc
yml_filename = os.path.join(script_dir, '..', 'configs', 'dedispersion', 'chord_sb2_et.yml')
pkl_filename = os.path.join(script_dir, 'throwaway1.pkl')

config = DedispersionConfig.from_yaml(yml_filename)
plan = DedispersionPlan(config, gpu_runnable=False)
freq_variances = np.random.uniform(0.0, 1.0, size=int(plan.nfreq))

t0 = time.time()
approx = PfAvarApproximation(plan, freq_variances, progress=True)
print(f'PfAvarApproximation construction took {time.time()-t0:.3f} seconds', flush=True)

members = ['ntrees', 'nfreq', 'freq_variances', 'tree_r', 'tree_R', 'tree_L', 'tree_P', 'per_tff', 'per_tf']
d = {name: getattr(approx, name) for name in members}

t0 = time.time()
with open(pkl_filename, 'wb') as f:
    pickle.dump(d, f, protocol=pickle.HIGHEST_PROTOCOL)

nbytes = os.path.getsize(pkl_filename)
print(f'wrote {pkl_filename} ({nbytes} bytes = {nbytes/2**20:.1f} MiB) in {time.time()-t0:.3f} seconds')
