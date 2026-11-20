# Experiment plan: patient-level LNM prediction from WSIs

**Status: draft.** "Pipeline" means the patient-level preprocessing pipeline
(`stamp preprocess`, see `src/stamp/preprocessing/README.md`). Experiments are split
into **core** (planned) and **potential** (only if time allows or a core result
motivates them).

## 1. Question and setup

**Question.** How should a patient's WSIs be turned into one input for a
lymph-node-metastasis (`isLNM`) classifier, and which choices matter? The main
comparison is `conch1_5` tile features used **with TITAN** (patient embedding) and
**without TITAN** (tile-level multiple-instance learning).

**Data.** A few hundred WSIs from several hospitals, one binary label per patient,
several slides per patient. Every split is **by patient** (never by slide).

**Held fixed across experiments**, so that differences come from the factor under
test:

- Cross-validation folds: stratified by label, grouped by patient, fixed seed, the
  same folds for every experiment. *(proposed: 5 folds x 3 repeats)*
- Metrics: AUROC (primary), AUPRC, balanced accuracy at a threshold chosen on the
  training folds. Report mean and spread over folds and repeats; with this few
  patients the noise is large, so differences smaller than the fold-to-fold spread
  are not claims.
- Defaults unless the experiment varies them: tiles 256 um / 224 px, extractor
  `conch1_5`, `first_n` sampling with 8192 tiles, `pad_mode: none`.
- Preprocessing outputs of all experiments live in one `output_dir`; folder names
  encode the settings (README, "Outputs"), so nothing is overwritten and shared work
  (tiling, features) is reused.

**Baselines every table needs:** (a) majority class, (b) mean of tile features per
patient + logistic regression (the simplest thing that could work), (c) the default
setup above.

## 2. Core experiments

### C1. Sampling strategy

At a fixed tile budget (8192, and a smaller one such as 1024 where the strategies
differ more):

| strategy | meaning | status |
|---|---|---|
| `first_n` | first tiles in raster order (biased to the top of each slide) | supported |
| `random` | seeded uniform sample per slide | supported (`strategy: random`) |
| `tissue` | tiles with the least background | PR #3 (open) |
| none | every tile | supported (`sampling.enabled: false`); needs chunked encoding for TITAN |

- Run `random` with several seeds (e.g. 5) to measure how much of any difference is
  just which tiles were drawn.
- Check by eye with `save_visualisations: true` (green = sampled, red = dropped).

### C2. Number of tiles

Learning curve over `n_tiles`: 256, 512, 1024, 2048, 4096, 8192, all.

- Status: supported by changing `sampling.n_tiles` (new sampled and feature folders
  only).
- Report how many patients have fewer tiles than each budget (they are not sampled
  at that budget), and accuracy as a function of tiles actually used.

### C3. Where sampling happens relative to feature extraction

`sample_then_extract` vs `extract_then_sample`. For sampling that does not look at
features (`first_n`, `random`, and handcrafted image or position based rules such as
`tissue`), both orders produce the **same training bags** and end in the same folder,
so the comparison is about **computational cost** only: sampling first runs the model
on sampled tiles only; extracting first runs it on every tile once and makes further
sampling settings cheap.

- Cost rule: use `extract_then_sample` when more than about three sampling settings
  will be tried on the same extractor (C1, C2), `sample_then_extract` otherwise.
  Measure GPU hours, wall time and disk for both on the same patients.
- **Handcrafted sampling techniques** (open idea): compare simple rules that need no
  features, e.g. `tissue` (least background), grid or spatially stratified sampling,
  stain-intensity based selection. These can run in either order.
- **Feature-based sampling** (open idea): techniques that need the tile features
  (e.g. k-means or farthest-point selection in feature space) can only run after
  extraction, so they are a genuine reason for `extract_then_sample`, and the only
  case where the order changes the bags. Not implemented.
- Status: both orders supported. Disk: `extract_then_sample` keeps all slide features
  (large).

### C4. Sampling before vs after encoding

For TITAN, the patient embedding can be built from:

| variant | what TITAN sees | status |
|---|---|---|
| A. sample, then encode once | one bag of `n_tiles` sampled tiles | supported |
| B. encode every tile in chunks, average | all tiles, in chunks of `chunk_size` | PR #2 (open) |
| C. sample, then encode in chunks, average | `n_tiles` tiles in smaller chunks | PR #2 |
| D. per-slide embeddings, then mean / max / concat | each slide separately | PR #4 (open, stacked on #2) |

Questions answered: does using more tiles (B vs A) help, does chunking itself hurt
(C vs A), and does treating slides separately matter (D vs A).

### C5. With vs without TITAN (`conch1_5` only)

- **With TITAN:** TITAN patient embedding (768-d) -> small classifier (logistic
  regression, MLP).
- **Without TITAN:** `conch1_5` tile features -> multiple-instance learning model on
  the bag (STAMP's `trans_mil`, `vit`, `mlp`).
- Same folds, same tiles, same seeds.
- Status: both feed from the pipeline's training bags and embeddings. The training
  configuration for each model type still has to be written.

### C6. Slide aggregation

Part of C4-D: `mean` vs `max` vs `concat` (padded to the largest slide count) vs the
patient as one virtual slide. Only the cheap `aggregate` step reruns when switching.

- Caveat: concatenation slots carry no meaning across patients; report the
  distribution of slides per patient next to the result.

## 3. Potential experiments

### P1. Padding

`pad_mode: none` vs `zero` vs `repeat` for tile-level models that batch bags to one
length (the without-TITAN arm of C5). Padding does not apply to encoded embeddings
(the encoder drops padding and repeats).

- Status: supported. `repeat` gives the model duplicated tiles that look real, so
  expect it to behave differently from `zero` (masked).

### P2. Magnification / tile size

Single-magnification runs: tile size in um at a fixed pixel size, for example
64, 128, 256 um (about 40x, 20x, 10x). Tile counts grow roughly 16x per halving of
the side, so use a fixed `n_tiles` and also a fixed-area-covered comparison.

- Status: each tile size is a separate run (`tile_size_um`, `tile_size_px`); all
  stages are keyed by it. **Combining magnifications** in one model (multi-scale
  bags, or concatenating per-magnification embeddings) is not supported and would be
  new work; only build fusion if one magnification is clearly better than the others.
- TITAN expects a specific magnification and patch size; check that the tile size and
  pixel size used for `conch1_5` match before interpreting TITAN results at other
  magnifications.

### P3. Other feature extractors

Repeat the core comparison with other extractors: `uni2`, `virchow2`, `gigapath`,
`chief_ctranspath`, `ctranspath`, `conch` (and `resnet50` as a weak baseline).

- Status: supported (`preprocessing.extractor`); only extraction is repeated.
- TITAN only accepts `conch1_5`, so other extractors can only be used with
  tile-level models (the pipeline skips encoding for them automatically).
- Cost: extraction dominates; estimate GPU time on one patient first.

### P4. Other

1. **Hospital shift:** leave-one-hospital-out evaluation next to random CV. Several
   hospitals means batch effects (scanner, stain) can be learned as shortcuts.
2. **Tissue background threshold** (`sampling.background_cutoff`) for the `tissue`
   strategy, after checking a few slides by eye.
3. **Slide-ensemble at test time:** predict per slide (or per chunk) and average
   predictions, vs averaging embeddings first (C4-D vs C4-B).
4. **Stain normalisation:** not in the pipeline; only if P4.1 shows a hospital
   effect.
5. **Patients with few tiles or few slides:** results stratified by tile count and
   slide count.
6. **Class imbalance:** class weights vs balanced sampling.
7. **Seeds:** training-seed variance separate from fold variance.

## 4. Order of work

1. **Pipeline checks** (cheap): tile counts per patient, sampling overviews on a few
   patients, baselines (b).
2. **C1 + C2** with `conch1_5`. Use `extract_then_sample` so all sampling settings
   share one extraction pass; record the cost for C3 while doing it.
3. **C4 + C6** (TITAN variants), then **C5** (with vs without TITAN).
4. Potential experiments (P1 to P4) as time allows.

## 5. Open items

| # | question | answer |
|---|---|---|
| Q1 | Sampling before vs after encoding = variants A to D of C4 | confirmed |
| Q2 | CV scheme: 5 x 3, grouped and stratified by patient? Also leave-one-hospital-out? | open |
| Q3 | Primary metric AUROC? | open |
| Q4 | Is 256 um / 224 px the right tile size for `conch1_5` and TITAN? | open |
| Q5 | Classifier heads for the TITAN arm: logistic regression, MLP, or both? | open |
| Q6 | Comparison against a published baseline? | open |
