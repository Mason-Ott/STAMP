# Experiment plan

## Experiments

- Sampling strategy (first_n, random, tissue, all tiles)
- Number of tiles (256, 512, 1024, 2048, 4096, 8192, all)
- Sampling before vs after feature extraction (computational cost)
- Sampling before vs after encoding (sample then encode, encode all tiles in chunks and average, encode each slide then combine)
- With vs without TITAN (conch1.5)
- Slide embedding aggregation (mean, max, concatenate)

## Potential Future Experiments

- Padding (none, zero, repeat)
- Magnification / tile size
- Feature extractors (uni2, virchow2, gigapath, chief_ctranspath, ctranspath, conch)
- Handcrafted sampling techniques (e.g. grid or spatially stratified, stain-based)
- Feature-based sampling (e.g. k-means on tile features, after extraction only)
- Hospital shift (leave-one-hospital-out)
- Stain normalisation
