# Comparison scripts

Require at least Python 3.8.
Read the binaries, parameter sets and image sets from `popsift-testing.json` (or
`popsift-testing-<config>.json`), which CMake writes into the build directory
when `PopSift_TEST_WITH_VLFEAT` is ON.

## Runs and parameter sets

A run is `vlfeat` (VLFeat's sift CLI) or the name of a parameter set for
`popsift-demo` from `Testing/paramsets.json`:

| Set | From |
| --- | --- |
| `vlfeat-emulation` | best know parameters for comparison with vlfeat, used by the CTest tests |
| `loop`, `iloop`, `grid`, `igrid`, `notile` | one for each alternative descriptor implementation |
| `oxford-legacy` | forced upsampling example |

Adding a set to the JSON file needs no code. A run can carry overrides:
`loop@gauss-mode=vlfeat` replaces the `--gauss-mode` of `loop`, and
`loop@!filter-grid` removes `--filter-grid`.

## popsift_compare.py

Compares one run against another on every image of a set, as the CTest
tests of `Testing/compareVlfeat` do one image and one step at a time:

```
python3 Testing/scripts/popsift_compare.py --tools build/popsift-testing.json \
        --images oxford/boat vlfeat-emulation vlfeat
```

`--images` is a configured set (see `Testing/datasets/README.md`) or a folder.
Images can be PGM or PPM, where PPM is converted to 8-bit PGM before use.

The first run named is the run under test (a), the second the reference (b).
For every image, `popsift-compareSiftFiles --match-dir` pairs the features of
a and b by position, scale and orientation (not by descriptor) and writes
`matched.csv`, `only_a.csv` and `only_b.csv`. It also finds, for every feature
of a, the feature of b with the most similar descriptor (`desc_nn_a.csv`), and
the same for b in a (`desc_nn_b.csv`). `match_stats.py` turns them into
statistics:

- counts, recall (matched / keypoints of b) and precision (matched /
  keypoints of a),
- pixel distance, offset x and y, scale ratio, rotation difference (b - a)
  and descriptor distance of the matched pairs, each with mean, median, p90,
  p99, max and a histogram,
- the descriptor distance of the matched pairs with the same pose (at most
  0.25 px and 2 degrees apart), whose descriptors should be nearly identical,
- the pixel distance from every keypoint to the keypoint with the most
  similar descriptor in the other run, and the nearest-descriptor agreement:
  the share of matched keypoints whose most similar descriptor is their
  geometric match. A keypoint whose most similar descriptor belongs to a
  keypoint elsewhere points to a serious descriptor problem.

Descriptor distance is always the L2 (Euclidean) distance of the 128
descriptor values in byte units (0-255 per value). In `summary.csv` and
`summary.json` every quantity carries its unit in its name: `_px` for pixels,
`_deg` for degrees and `_l2` for descriptor distances, e.g. `pos_dist_px`,
`ori_diff_deg`, `desc_dist_l2`. `Testing/analysis/glossary.md` explains all
terms.

Per set, the statistics are pooled over all images and also given as the
spread (min / median / max) of the per-image values. The set report goes to
`<work dir>/<set>/report_<a>__vs__<b>/` as `summary.csv`, `summary.json` and
`report.md`, and a one-line summary per image and per set is printed.
When numpy and matplotlib are installed (`Testing/analysis/requirements.txt`),
every image folder and the report also get `cdf.png`, and every image folder
`unmatched.png`, see `Testing/analysis/README.md`. `--no-plots` skips them.

An image fails when a run crashes or when exactly one run finds no features.
An image where neither run finds features also fails, unless the set's folder
lists it in `expectations.json`:

```json
{ "images": { "box-12x12-symm.pgm": { "allow_empty": true } } }
```

Beyond that the script only reports. Optional limits turn it into a gate;
a negative value (the default) disables a limit:

| Option | An image fails when |
| --- | --- |
| `--min-recall R` | recall < R |
| `--min-precision P` | precision < P |
| `--max-desc-dist D` | median descriptor distance (L2) of the matched pairs > D |
| `--min-nn-agreement N` | nearest-descriptor agreement of a or of b < N |
| `--max-avg-dist D` | average best-match descriptor distance > D |

`--image <name>` restricts the script to one image of the set (its path in
the set, with or without extension). `--step run|compare|report` does one
step: run one run on one image, compare the earlier runs of one image, or
write the set report from the images compared so far. PopSift runs are
skipped when `popsift-gpu-probe` finds no usable GPU; the exit code is then
77, which CTest shows as skipped.

## compare_paramsets.py

A Python script for manual comparisons by developers, not used in CTest runs.
It runs each set once per image and compares every pair, or every set against
a baseline:

```
python3 Testing/scripts/compare_paramsets.py --build build \
        --sets loop,grid,igrid,notile --images oxford/boat
python3 Testing/scripts/compare_paramsets.py --build build \
        --sets vlfeat-emulation,loop --baseline vlfeat
python3 Testing/scripts/compare_paramsets.py --build build --list
```

It prints the mean number of keypoints per set and, per pair, recall,
precision, the median pixel distance, rotation difference and descriptor
distance of the matched keypoints and the nearest-descriptor agreement,
pooled over all images. Each pair
gets a report folder `<work dir>/report_<a>__vs__<b>/` like the one of
`popsift_compare.py`, and with matplotlib `<work dir>/cdf_all_pairs.png` shows
the distributions of all pairs in one plot. The
script makes no pass/fail decision.
Build the binaries first (`cmake --build build`, plus `--target vlfeat_sift`
for the `vlfeat` run).

