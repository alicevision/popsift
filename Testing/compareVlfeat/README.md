# Comparing PopSift with VLFeat in CTest

The tests exist when CMake is configured with `BUILD_TESTING=ON` and
`PopSift_TEST_WITH_VLFEAT=ON`, and need Python 3.8 and `popsift-demo`
(`PopSift_BUILD_EXAMPLES=ON`). They use every test image set: the hand-made
images in `Testing/testImages` (the set `handcrafted`) and every folder below
`PopSift_TESTDATA_DIR` that holds .pgm or .ppm files, see
`Testing/datasets/README.md`. Nothing is downloaded.

Every image of every set gets three tests, for example for
`oxford/boat/img1`:

| Test | Does | Labels |
| --- | --- | --- |
| `run-vlfeat/oxford/boat/img1` | runs VLFeat on the image | `cpu` |
| `run-popsift/oxford/boat/img1` | runs `popsift-demo` with the parameter set `popsift-vlfeat` | `gpu`, and `cuda` or `hip` |
| `compare/oxford/boat/img1` | compares the two runs and prints a one-line verdict | `cpu` |

and every set gets `report/<set>`, which writes the set report after the
comparisons of the set. All of them are labelled `vlfeat` and with the set
name, so `ctest -L oxford/boat` runs one set and `ctest -R img1` one image
together with the runs it needs.

The PopSift runs use the GPU one at a time (`RESOURCE_LOCK gpu`); the VLFeat
runs and the comparisons run in parallel with `ctest -j`. Each PopSift run
first asks `popsift-gpu-probe` whether there is a usable GPU. Without one it
is skipped, and so are its comparison and, if nothing was compared, the
report.

The tests run `Testing/scripts/popsift_compare.py` one step at a time
(`--step run|compare|report`), with the same work directory and reports as
a manual run, see `Testing/scripts/README.md`.

Two more groups of tests do not need a GPU: `compareSiftFiles/<case>`
(labels `cpu`, `unit`) tests `popsift-compareSiftFiles` on small hand-made
feature files, and `vlfeat_sift_smoke` checks the patched VLFeat `sift`.

## Running it

```sh
cmake -S . -B build -DBUILD_TESTING=ON -DPopSift_TEST_WITH_VLFEAT=ON \
      -DPopSift_TESTDATA_DIR=/data/popsift-testdata
cmake --build build
ctest --test-dir build -j8 --output-on-failure
```

CTest tests what was built, so build first. With a multi-config generator
(Visual Studio, Xcode, Ninja Multi-Config) the comparison tests run in the
Debug configuration only, which is easier to inspect in a debugger when a
comparison points to a bug: `cmake --build build --config Debug` and
`ctest --test-dir build -C Debug`. `ctest -L handcrafted` is a quick check,
`ctest -L unit` runs only the unit tests. A label does not leave out the
runs a selected comparison needs: CTest adds them as fixtures.

## Limits

By default a comparison only reports. It fails when a run crashes or exactly
one run finds no features. `PopSift_TEST_COMPARE_LIMITS` passes limits to
`popsift_compare.py` as a CMake list, for example

```sh
-DPopSift_TEST_COMPARE_LIMITS="--min-recall;0.95;--min-ori-agreement;0.9;--min-desc-agreement;0.8"
```

The limits are those of `popsift_compare.py` (`--min-recall`,
`--min-precision`, `--min-ori-agreement`, `--min-desc-agreement`,
`--max-desc-dist`, `--min-nn-agreement`, `--max-avg-dist`), see
`Testing/scripts/README.md`. Run once without limits, read the report, then
set the limits a bit beyond the measured values to make the tests a
regression gate.

## Reports

`report/<set>` writes the report of PopSift (a) against VLFeat (b) to
`<build>/Testing/run/<set>/report_popsift-vlfeat__vs__vlfeat/`: `report.md`
to read, `summary.csv` and `summary.json` for scripts. With numpy and
matplotlib installed, the report and every image folder also get the plots
described in `Testing/analysis/README.md`. `ctest -V` shows the verdict of
every image and the summary of every set.

To look through the results interactively, open the work dir with
`Testing/analysis/explorer.py`. `Testing/analysis/make_explorer_package.py`
packs the reports, plots and images of the work dir into a zip file for
testers without a build.
