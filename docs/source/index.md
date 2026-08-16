# Pirate - Perimeter Institute RAdio Transient Engine

An experimental GPU-based fast transient search for CHORD.

% Anchor (not a heading) for the sidebar's "Tex notes" dropdown, which links
% here. A heading would pull the toctree below into a "Tex notes" section.
(tex-notes)=
Tex notes -- mathematical details of the search, in three parts:

- <a href="_static/dedispersion.pdf" target="_blank" rel="noopener">Dedispersion (PDF)</a> -- tree gridding, tree dedispersion, subband search, peak-finding, downsampled trees and early triggers, and the GPU implementation.
- <a href="_static/detrending.pdf" target="_blank" rel="noopener">Detrending (PDF)</a> -- the two 1-d time detrenders (local polynomial subtraction, Kalman filter) and the 2-d (frequency-time) detrender.
- <a href="_static/variance_map.pdf" target="_blank" rel="noopener">Variance map (PDF)</a> -- the per-output variances used to normalize the peak-finder, exact and approximate.

```{toctree}
---
maxdepth: 2
caption: User Guide
---
notes/install
notes/intro
notes/quick_start
notes/developer
notes/grouper_interface
notes/build
notes/hardware
cli
formats
python_class_reference
llm_context
```
