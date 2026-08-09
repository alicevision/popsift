# VLFeat comparison via CTest.

Enable with `-DPopSift_COMPARE_VLFEAT=ON`.
Compares all .pgm/.ppm files found under `PopSift_TESTFILE_PATH`.

## Prerequisites

- Provide test images below the directory `PopSift_TESTFILE_PATH`, or
  populate from the Oxford affine-covariant datasets by setting
  `PopSift_TEST_DATASETS` to any subset of this list:
  `boat;bikes;trees;graf;wall;bark;leuven;ubc`
  Relevant only when `PopSift_USE_TEST_CMD` or `PopSift_COMPARE_VLFEAT` are active.
- ImageMagick's `convert` is used to convert .ppm images to .pgm.
  Fails if `convert` is not available and .ppm images are present.

## Tuning

- `PopSift_VLFEAT_MAX_AVG_DESC_DIST`: average best-match descriptor distance
  above which the test fails. Defaults to `-1` (report only, never fail).
  Run once, look at the reported "Average best-match descriptor distance"
  line, then set this to something a bit above the measured value to turn it
  into a real regression gate.
- Change the exact `popsift-demo` flags in `run_comparison.sh.in`.
  The current flags are the best-known flags to emulate default vlfeat
  behaviour. Edit the script when necessary.
- `PopSift_VLFEAT_GIT_TAG` select the last 2018 commits of VLFeat, which was
  the youngest vlfeat commit when this was written in 2026.

## Known limitation

No Windows support. The testing works on Linux.

The Mac patch exists because we want to integrate the SYCL branch of PopSift into
develop enventually, and the SYCL branch works on Mac. The test code was first written
to compare the SYCL code with vlfeat.

