# Testing PopSift

The tools in this folder compare the keypoints and descriptors that PopSift
finds with those of VLFeat, or of one PopSift parameter set with another.
They are built when `BUILD_TESTING` is ON and are run by hand.

## Configure options

| Option | Default | Effect |
| --- | --- | --- |
| `BUILD_TESTING` | OFF | Builds `popsift-compareSiftFiles` and `popsift-gpu-probe`, finds the test image sets and writes `popsift-testing.json` into the build directory. |
| `PopSift_TEST_WITH_VLFEAT` | OFF | Also downloads, patches and builds VLFeat's `sift` CLI as the reference (needs `BUILD_TESTING`). |
| `PopSift_TESTDATA_DIR` | `$POPSIFT_TESTDATA_DIR` | Folder of your own test images. Every folder below it that holds .pgm or .ppm files becomes an image set. |

```sh
cmake -S . -B build -DBUILD_TESTING=ON -DPopSift_TEST_WITH_VLFEAT=ON \
      -DPopSift_TESTDATA_DIR=/data/popsift-testdata
cmake --build build
python3 Testing/scripts/popsift_compare.py --tools build/popsift-testing.json \
        --images handcrafted vlfeat-emulation vlfeat
```

The last command runs PopSift with the parameter set `vlfeat-emulation` and
VLFeat on every image of the set `handcrafted`, and writes a report per image
and per set below `build/Testing/run/`.

## Contents

| Folder | Content |
| --- | --- |
| [`compareSiftFiles`](compareSiftFiles) | `popsift-compareSiftFiles`, which pairs the features of two descriptor files by position, scale and orientation, and by descriptor |
| [`vlfeat`](vlfeat/README.md) | the patched VLFeat `sift` CLI that serves as the reference |
| [`datasets`](datasets/README.md) | how image sets are found, and a download script for the Oxford sets |
| [`testImages`](testImages) | small hand-made images, the set `handcrafted` |
| [`scripts`](scripts/README.md) | the comparison scripts and their reports |
| [`gpuProbe`](gpuProbe) | `popsift-gpu-probe`, which tells the scripts whether a GPU is usable |
| `paramsets.json` | named parameter sets for `popsift-demo` |
