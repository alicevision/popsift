# Comparison scripts

Require at least Python 3.8 and nothing beyond its standard library.
They read the binaries, parameter sets and image sets from
`popsift-testing-config.json`, which CMake writes into the build directory when `PopSift_BUILD_TESTING` is ON. The `vlfeat` run needs
`PopSift_TEST_WITH_VLFEAT` ON as well.

Extraction and comparison are separate steps. `collect_run.py` runs one
configuration of a SIFT tool on image sets and keeps the keypoints and
descriptors as a result store. `popsift_compare.py` compares two stores. A
store can come from another build, another computer or another SIFT tool,
so PopSift can be compared with anything that writes a store.

## Configurations and parameter sets

A configuration is `vlfeat` (VLFeat's sift CLI) or the name of a parameter
set for `popsift-demo` from `Testing/paramsets.json`:

| Set | From |
| --- | --- |
| `popsift-vlfeat` | best known parameters for comparison with VLFeat |
| `loop`, `iloop`, `grid`, `igrid`, `notile` | one for each alternative descriptor implementation |
| `oxford-legacy` | forced upsampling example |

Adding a set to the JSON file needs no code. A configuration can carry
overrides: `loop@gauss-mode=vlfeat` replaces the `--gauss-mode` of `loop`,
and `loop@!filter-grid` removes `--filter-grid`. Parameter set names must
not contain `+`.

## Result stores

A result store holds the results of one run, that is one SIFT tool with one
configuration, on one or more image sets:

```text
<stores dir>/<run name>/
    run.json                      the manifest
    <set>/<image stem>.sift       the keypoints of one image
    <set>/<image stem>.log        optional: the tool's output for the image
```

- `<stores dir>` is `<build>/Testing/stores` (`stores_dir` in
  `popsift-testing-config.json`); every script takes `--stores <dir>` to use
  another one, e.g. a folder copied from another computer.
- `<set>` is the label of the image set as configure prints it, a folder per
  part: `handcrafted/box-6x6.sift`, `oxford/boat/img1.sift`. Image sets are
  trusted to have the same label and the same images on every computer;
  nothing checks this.
- `<image stem>` is the image's path in the set without its extension.
- A `.sift` file holds one keypoint per line with 132 numbers, as VLFeat's
  `sift` CLI writes them: x, y, sigma, the orientation in radians and the
  128 descriptor values in VLFeat's order (4 x 4 cells of 8 orientation
  bins), as bytes 0-255. x and y are in pixels of the input image, (0, 0) is
  the centre of the top-left pixel and y points down. Lines need not be
  sorted. An empty file means that the tool found no keypoints; a missing
  file, that the image was not processed. Files in the Oxford affine format
  (133 numbers per line) are rejected; other formats need a converter.
- The tool reads the images as they are. For PPM images, the runs of
  `collect_run.py` read an 8-bit grey PGM converted with Rec. 709 luma, as
  ImageMagick 6 does; a tool that converts colour differently differs for
  that reason alone.

`run.json` needs only two fields:

```json
{
  "format": "popsift-result-store/1",
  "name": "other-tool+default"
}
```

`name` is the run name, made of letters, digits and `._=+-`. `label`, if
present, is how reports and the explorer call the run. Everything else is
free-form provenance that the reports print. `collect_run.py` writes
`configuration`, `tool`, `command`, `version`, `build` (the build's
`PopSift_*` options), `device`, `host`, `created` and `image_sets`.

## collect_run.py

Runs one configuration on one or more image sets and writes or extends its
store:

```shell
python3 Testing/scripts/collect_run.py --tools build-b/popsift-testing-config.json \
        --prefix atan2-fast --label "PopSift, fast atan2" \
        --images handcrafted --images oxford/boat popsift-vlfeat
```

The run name is `<prefix>+<configuration>`, here
`atan2-fast+popsift-vlfeat`, or the configuration alone without `--prefix`.
`--images` can be repeated, and takes a configured image set
(`oxford/boat`), the start of the labels of several (`oxford` for every
configured set below `oxford/`), or a folder. A folder of a configured set,
or above configured sets, gets their labels, so that stores of different
builds use the same labels; any other folder is labelled by its name, with
a note. `--list` prints the configured sets.
`--image` restricts it to one image of a single set. Collecting into an
existing store adds or replaces images and sets, but only for the same run:
a store whose configuration, command or build differs is refused, or
emptied first with `--replace`. The CTest tests, `popsift_compare.py` and
`compare_paramsets.py` collect with `--replace`. PopSift runs need a GPU;
without one an image is marked as skipped in the store (`<image
stem>.skipped`) and the exit code is 77.

`collect_run.py` is the only script that runs `popsift-demo` or VLFeat.

## check_store.py

```shell
python3 Testing/scripts/check_store.py /data/stores/other-tool+default \
        --tools build/popsift-testing-config.json
```

Checks a store before it is compared: the manifest, every feature file
(132 numbers per line, the Oxford format named as such), orientations that
look like degrees, descriptors that look like floats in 0-1, and, with
`--tools`, the images of the store's image sets without a feature file.

## popsift_compare.py

Compares one run against another on every image of a set:

```shell
python3 Testing/scripts/popsift_compare.py --tools build/popsift-testing-config.json \
        --images oxford/boat popsift-vlfeat vlfeat
python3 Testing/scripts/popsift_compare.py --tools build/popsift-testing-config.json \
        --images oxford/boat atan2-fast+popsift-vlfeat atan2-default+popsift-vlfeat
```

A run is a store, by its name in the stores dir or by its path. The two
runs come last or anywhere between the options, a first. `--images` takes
the same values as in `collect_run.py`; several sets are compared one after
the other, each with its own report. A
configuration name (`vlfeat` or a parameter set) is collected by
`collect_run.py` into the store of that name first, so the first command
works as it always did; `--no-collect` uses the existing stores of
configurations instead. A store without the image set fails; an image that
a store from earlier lacks is reported as not compared, and does not fail.

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
  orientation), with the counts of every level. The pairs whose orientation
  or descriptor differs are also counted separately when another keypoint
  at the same position (SIFT gives one position several orientations)
  agrees further (`orientation_differs_minor`, `descriptor_differs_minor`,
  from the `best_at_position` column of `colocated.csv`): these are minor
  problems,
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
step: collect one configuration on one image (with `collect_run.py`),
compare two stores on one image, or write the set report from the images
compared so far. Both are meant for CTest tests that run one image and one
step at a time; by hand they are useful to repeat a single image.

The set report names both runs with their label, tool, version, build
options, device, host and time, from their `run.json`, in `report.md` and
`summary.json` (`runs`).

## Comparing two builds

For example, two builds that differ only in `PopSift_TEST_WITH_VL_ATAN2`,
possibly on different computers:

```shell
# extraction, in each build
python3 Testing/scripts/collect_run.py --tools build-a/popsift-testing-config.json \
        --prefix atan2-default --images oxford popsift-vlfeat
python3 Testing/scripts/collect_run.py --tools build-b/popsift-testing-config.json \
        --prefix atan2-fast --images oxford popsift-vlfeat
python3 Testing/scripts/collect_run.py --tools build-a/popsift-testing-config.json \
        --images oxford vlfeat
# comparison, with the tools of one build, after copying
# build-b/Testing/stores/atan2-fast+popsift-vlfeat to build-a/Testing/stores
python3 Testing/scripts/popsift_compare.py --tools build-a/popsift-testing-config.json \
        --images oxford atan2-fast+popsift-vlfeat atan2-default+popsift-vlfeat
python3 Testing/scripts/popsift_compare.py --tools build-a/popsift-testing-config.json \
        --images oxford --no-collect atan2-fast+popsift-vlfeat vlfeat
python3 Testing/analysis/explorer.py build-a/Testing/run
```

## compare_paramsets.py

For comparisons between parameter sets by developers. It collects each set
with `collect_run.py` and compares every pair, or every set against a
baseline:

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

