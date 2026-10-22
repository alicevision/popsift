# VLFeat comparison via CTest.

The CTest modules require from the CMake configuration:
 - BUILD_TESTING=ON
 - PopSift_TEST_WITH_VLFEAT=ON
 - an image source, either through PopSift_TESTDATA_DOWNLOAD=ON or by providing local files in PopSift_TESTDATA_DIR. A directory in the input becomes a test set. A set of tiny test images in `Testing/testImages` is always included.

There is one test per available test set, named
`vlfeat-comparison/<set>` and labelled with the set name, for example
`ctest -L oxford/boat`.
The VLFeat reference is built by `Testing/vlfeat`, see its README.

Each CTest test runs `Testing/scripts/popsift_compare.py` with the parameter set
`vlfeat-emulation` against `vlfeat`.
For other purposes, manual testing is possible. `Testing/scripts/README.md` describes this.

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
- Outputs of every run are kept in `<build>/Testing/compareVlfeat/run/<set>/`.

## Reports

Each test writes a report of PopSift (a) against VLFeat (b) to
`<build>/Testing/compareVlfeat/run/<set>/report_vlfeat-emulation__vs__vlfeat/`:
`report.md` to read, `summary.csv` and `summary.json` for scripts. The
quantities are explained in `Testing/scripts/README.md`. With numpy and
matplotlib installed, the report and each image folder also have `cdf.png`,
each image folder `unmatched.png`, and
`Testing/analysis/inspect_matches.py` opens the report interactively, see
`Testing/analysis/README.md`. `ctest -V` shows the per-image summary lines.
