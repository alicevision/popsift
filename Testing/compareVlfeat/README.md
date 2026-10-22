# VLFeat comparison via CTest.

The CTest modules require from the CMake configuration:
 - BUILD_TESTING=ON
 - PopSift_TEST_WITH_VLFEAT=ON
 - an image source, either through PopSift_TESTDATA_DOWNLOAD=ON or by providing local files in PopSift_TESTDATA_DIR. A directory in the input becomes a test set. A set of tiny test images in `Testing/testImages` is always included.

Every image of every available test set gets three tests, for example for
`oxford/boat/img1`:

| Test | Does | Labels |
| --- | --- | --- |
| `run-vlfeat/oxford/boat/img1` | runs VLFeat on the image | `cpu` |
| `run-popsift/oxford/boat/img1` | runs `popsift-demo` with the parameter set `vlfeat-emulation` | `gpu`, and `cuda` or `hip` |
| `compare/oxford/boat/img1` | compares the two runs and prints a one-line verdict | `cpu` |

and every set gets `report/<set>`, which writes the set report after the
comparisons of the set. All of them are labelled `vlfeat` and with the set
name, so `ctest -L oxford/boat` runs one set and `ctest -R img1` one image
together with what it needs.

The PopSift runs use the GPU one at a time (`RESOURCE_LOCK gpu`), the VLFeat
runs and comparisons run in parallel with `ctest -j`. Each PopSift run first
asks `popsift-gpu-probe` whether there is a usable GPU. Without one it is
skipped, and so are its comparison and, if nothing was compared, the
report. The configure output names the GPU backend.

The VLFeat reference is built by `Testing/vlfeat`, see its README.

The tests run `Testing/scripts/popsift_compare.py` one step at a time, with
the parameter set `vlfeat-emulation` against `vlfeat`. For other purposes,
manual testing is possible. `Testing/scripts/README.md` describes this.

## Running it on a GPU machine

```
cmake -S . -B build -DPopSift_TEST_WITH_VLFEAT=ON -DPopSift_TEST_DATASETS=oxford -DPopSift_TESTDATA_DOWNLOAD=ON
ctest --test-dir build -j8 --output-on-failure
```

CTest builds VLFeat and the PopSift binaries, downloads the Oxford sets, runs
all tests and prints the verdict of every failed test. The reports are in
`build/Testing/compareVlfeat/run/<set>/report_vlfeat-emulation__vs__vlfeat/`.
`-L handcrafted` instead of `-DPopSift_TEST_DATASETS=...` is a quick check.

## Prerequisites

- Python 3.8

## Tuning

- By default a test only reports. It fails when a run crashes or exactly one
  run finds no features. Each of these cache variables turns on a limit
  when it is set to a value >= 0 (default `-1`, off):

  | Variable | An image fails when |
  | --- | --- |
  | `PopSift_VLFEAT_MIN_RECALL` | fewer than this share of VLFeat's features is matched |
  | `PopSift_VLFEAT_MIN_PRECISION` | fewer than this share of PopSift's features is matched |
  | `PopSift_VLFEAT_MAX_DESC_DIST` | the median descriptor distance of matched pairs is larger |
  | `PopSift_VLFEAT_MIN_NN_AGREEMENT` | fewer than this share of matched keypoints (in either direction) have their match as most similar descriptor |
  | `PopSift_VLFEAT_MAX_AVG_DESC_DIST` | the average best-match descriptor distance is larger |

  Run once without limits, read the report (below), then set the limits a
  bit beyond the measured values to make the test a regression gate.
- The `popsift-demo` flags are the parameter set `vlfeat-emulation` in
  `Testing/paramsets.json`, the best-known flags to emulate default VLFeat
  behaviour.
- Outputs of every run are kept in
  `<build>/Testing/compareVlfeat/run/<set>/<image>/`.

## Reports

`report/<set>` writes a report of PopSift (a) against VLFeat (b) to
`<build>/Testing/compareVlfeat/run/<set>/report_vlfeat-emulation__vs__vlfeat/`:
`report.md` to read, `summary.csv` and `summary.json` for scripts. The
quantities are explained in `Testing/scripts/README.md`. With numpy and
matplotlib installed, the report and each image folder also have `cdf.png`,
each image folder `unmatched.png`, and
`Testing/analysis/inspect_matches.py` opens the report interactively, see
`Testing/analysis/README.md`. `ctest -V` shows the per-image verdicts and the set summary.
