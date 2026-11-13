# Comparison scripts

Require at least Python 3.8 and nothing beyond its standard library.
They read the binaries, parameter sets and image sets from
`popsift-testing-config.json`, which CMake writes into the build directory when `BUILD_TESTING` is ON. The `vlfeat` run needs
`PopSift_TEST_WITH_VLFEAT` ON as well.

## Runs and parameter sets

A run is `vlfeat` (VLFeat's sift CLI) or the name of a parameter set for
`popsift-demo` from `Testing/paramsets.json`:

| Set | From |
| --- | --- |
| `popsift-vlfeat` | best known parameters for comparison with VLFeat |
| `loop`, `iloop`, `grid`, `igrid`, `notile` | one for each alternative descriptor implementation |
| `oxford-legacy` | forced upsampling example |

Adding a set to the JSON file needs no code. A run can carry overrides:
`loop@gauss-mode=vlfeat` replaces the `--gauss-mode` of `loop`, and
`loop@!filter-grid` removes `--filter-grid`.

## popsift_compare.py

Compares one run against another on every image of a set:

```shell
python3 Testing/scripts/popsift_compare.py --tools build/popsift-testing-config.json \
        --images oxford/boat popsift-vlfeat vlfeat
```

`--images` is a configured set (see `Testing/datasets/README.md`) or a folder
of images; its subfolders are not searched. `--list` prints the parameter
sets and the configured image sets and exits; `compare_paramsets.py --list`
prints the same.
Images can be PGM or PPM, where PPM is converted to 8-bit PGM before use.

The first run named is the run under test (a), the second the reference (b).
For every image, `popsift-compareSiftFiles --pair-dir` pairs the keypoints
of a and b by position and scale only (co-located, not by descriptor) and
rates how far every pair agrees:

| Agreement level | Condition | What a tester learns |
| --- | --- | --- |
| `position` | co-located, orientations differ by more than 2 degrees | the orientation computation differs; nothing can be concluded about the descriptor extraction |
| `orientation` | also the same orientation, descriptor distance (L2) above 12 | a strong hint at a difference in the descriptor extraction |
| `descriptor` | also a similar descriptor | the two agree |

Whether keypoints are co-located at all tells whether the pyramid, the DoG
and the extrema detection agree. A descriptor is judged in absolute terms
against the descriptor of its co-located keypoint, not by whether it is the
most similar one in the image: in a repetitive image another keypoint can
have a still more similar descriptor, and Lowe's ratio test may reject the
pair, without anything being wrong.

The tool writes `colocated.csv` (one row per pair, with its `agreement`),
`only_a.csv` and `only_b.csv`. It also finds, for every keypoint of a, the
keypoint of b with the most similar descriptor (`desc_nn_a.csv`), and the
same for b in a (`desc_nn_b.csv`). `match_stats.py` turns them into
statistics:

- the three indicators: recall (co-located / keypoints of b) and precision
  (co-located / keypoints of a), orientation agreement (same orientation /
  co-located) and descriptor agreement (similar descriptor / same
  orientation), with the counts of every level,
- pixel distance, offset x and y, scale ratio, rotation difference (b - a)
  and descriptor distance of the co-located pairs, each with mean, median,
  p90, p99, max and a histogram,
- the descriptor distance of the pairs with the same orientation, the only
  ones whose descriptors can be judged,
- the pixel distance from every keypoint to the keypoint with the most
  similar descriptor in the other run, and the nearest-descriptor agreement:
  the share of co-located keypoints whose most similar descriptor is their
  co-located keypoint.

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
every image folder and the report also get `agreement.png`, `cdf.png`,
`sorted.png` and `descriptor_bins.png`, and every image folder `unpaired.png` and
`image_heatmap.png`, see `Testing/analysis/README.md`. `--no-plots` skips
them.

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
| `--min-ori-agreement O` | orientation agreement < O |
| `--min-desc-agreement S` | descriptor agreement < S |
| `--max-desc-dist D` | median descriptor distance (L2) of the pairs with the same orientation > D |
| `--min-nn-agreement N` | nearest-descriptor agreement of a or of b < N |
| `--max-avg-dist D` | average best-match descriptor distance > D |

PopSift runs are skipped when `popsift-gpu-probe` finds no usable GPU; the
exit code is then 77.

`--image <name>` restricts the script to one image of the set (its path in
the set, with or without extension). `--step run|compare|report` does one
step: run one run on one image, compare the earlier runs of one image, or
write the set report from the images compared so far. Both are meant for
CTest tests that run one image and one step at a time; by hand they are
useful to repeat a single image.

## compare_paramsets.py

For comparisons between parameter sets by developers. It runs each set once per image and compares every pair, or every set against
a baseline:

```shell
python3 Testing/scripts/compare_paramsets.py --build build \
        --sets loop,grid,igrid,notile --images oxford/boat
python3 Testing/scripts/compare_paramsets.py --build build \
        --sets popsift-vlfeat,loop --baseline vlfeat
python3 Testing/scripts/compare_paramsets.py --build build --list
```

It prints the mean number of keypoints per set and, per pair, recall,
precision, orientation and descriptor agreement, the median pixel distance,
rotation difference and descriptor distance of the co-located keypoints and
the nearest-descriptor agreement, pooled over all images. Each pair
gets a report folder `<work dir>/report_<a>__vs__<b>/` like the one of
`popsift_compare.py`, and with matplotlib `<work dir>/cdf_all_pairs.png` shows
the distributions of all pairs in one plot. The
script makes no pass/fail decision.
Build the binaries first with `cmake --build build` (with
`PopSift_TEST_WITH_VLFEAT` ON for the `vlfeat` run).

