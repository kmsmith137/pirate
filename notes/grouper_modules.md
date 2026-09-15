# Grouper modules and naming

The grouper is the complete peakfinding, decoding and clustering pipeline.
Clustering is the stage that associates decoded candidates into events.
Offline and online adapters use the same processing implementation.

## Processing modules

| Module | Responsibility |
| --- | --- |
| `BowtiePeakfinding.py` | Bowtie geometry, peak selection and `StreamingPeakExtractor` across chunks. |
| `GpuArgmaxDecoder.py` | Decode selected coordinates and argmax tokens into physical parameters. |
| `Clustering.py` | Associate candidates with `cluster_candidates`, using the persistent CUDA kernel. |
| `GrouperPipeline.py` | `GrouperSetup`, `StreamingGrouper` and `ClusteringWindow`; manage chunk state and output ownership. |
| `GrouperConfig.py` | Validate the common YAML configuration before allocating GPU state. |
| `OfflineMapReader.py` | Discover, validate and upload saved native S/N and argmax maps. |
| `run_offline_grouper.py` | Run the pipeline on saved maps. |
| `OnlineGrouper.py` | Run the pipeline on live producer outputs. |
| `TriggerCatalog.py` | Validate and write event, candidate and membership tables. |
| `ArgmaxMetadata.py` | Validate the producer's token encoding and timing metadata. |

The producer's `cuda_generator/PeakFinder.py` generates the search reduction
kernel. `BowtiePeakfinding.py` selects local maxima from the resulting maps.

## Configuration types

- `GrouperConfig`: peakfinding, association and execution configuration used by
  both adapters.
- `GrouperConfig.ClusteringConfig`: halo size and DM/time tolerance settings from
  the YAML `grouping` section.
- `Clustering.ClusteringTolerances`: the two numerical tolerances accepted by the
  clustering algorithm.
- `Clustering.ClusteringGeometry`: native tree resolutions and dispersion slopes.
- `Clustering.GpuClusteringResult`: GPU candidate, event and membership tables,
  assignments and completion status.

Example imports:

```python
from pirate_frb.GrouperConfig import GrouperConfig, ClusteringConfig
from pirate_frb.BowtiePeakfinding import StreamingPeakExtractor
from pirate_frb.Clustering import (
    ClusteringGeometry, ClusteringTolerances, cluster_candidates,
)
from pirate_frb.GrouperPipeline import GrouperSetup
from pirate_frb.OfflineMapReader import OfflineMapReader
```

## Import compatibility

Use the names above in new code. The following modules forward existing imports
to the same implementation objects; they contain no separate algorithms.

| Existing import | Canonical import |
| --- | --- |
| `OfflineCandidateGrouper` | `Clustering` |
| `OfflineGrouperConfig` | `GrouperConfig` |
| `FrbOfflineGrouper` | `OfflineMapReader` |
| `SharedGrouper` | `GrouperPipeline` |
| `Peakfinders` | `BowtiePeakfinding` |
| `Peakfinders.OfflinePeakExtractor` | `BowtiePeakfinding.StreamingPeakExtractor` |
| `OfflineCandidateGrouper.GroupingConfig` | `Clustering.ClusteringTolerances` |
| `OfflineGrouperConfig.GroupingConfig` | `GrouperConfig.ClusteringConfig` |
| `OfflineCandidateGrouper.GroupingGeometry` | `Clustering.ClusteringGeometry` |
| `OfflineCandidateGrouper.GpuGroupingResult` | `Clustering.GpuClusteringResult` |
| `OfflineCandidateGrouper.group_candidates` | `Clustering.cluster_candidates` |
| `SharedGrouper.GroupingWindow` | `GrouperPipeline.ClusteringWindow` |
| `OfflineGrouperConfig.load_offline_grouper_config` | `GrouperConfig.load_grouper_config` |

The YAML `grouping` section and existing catalog field names remain the storage
contract, including the matching configuration attributes. The offline command
still uses `run offline_grouper`. The test selector is `test --grouper`, with
`--ofg` retained as an alias. Run the focused checks with:

```bash
python -m pirate_frb test --grouper --live -n 1
```
