# VLFeat comparison test

Enable with `-DPopSift_COMPARE_VLFEAT=ON`. This wires up a `ctest` target,
`vlfeat-comparison-boat`, that runs PopSift and a patched build of VLFeat's
`sift` CLI on the same image and reports how closely their descriptors match.

## What happens, and when

Turning the option on only adds CMake targets and test definitions at
configure time - it does not touch the network or build anything extra.
VLFeat is fetched, patched (see `patches/`) and built via `ExternalProject_Add`
with `EXCLUDE_FROM_ALL`, so `cmake --build .` never triggers it. The first
time `ctest` (or `cmake --build . --target test`) actually runs and selects
`vlfeat-comparison-boat`, a CTest fixture (`vlfeat_build`) builds the
`vlfeat_external` target on demand. Later runs reuse the already-built
checkout via ExternalProject's stamp files.

This mechanism (ExternalProject_Add + EXCLUDE_FROM_ALL + CTest fixtures) was
verified end to end in a throwaway sandbox: `cmake` configure alone does not
clone anything, `cmake --build .` alone does not either, and `ctest` triggers
the clone/patch/build exactly once and reuses it on repeat runs.

## Prerequisites

- A test image. Point `-DPopSift_VLFEAT_TEST_IMAGE=<path/without/extension>`
  at a `.pgm` or `.ppm` file, or fetch the Oxford dataset with
  `testScripts/downloadOxfordDataset.sh.in` (default expects
  `${PopSift_TESTFILE_PATH}/boat/img1.{pgm,ppm}`).
- ImageMagick's `convert`, only if your test image is a `.ppm`.
- Network access the first time the `vlfeat_build` fixture runs, unless you
  set `-DPopSift_VLFEAT_SOURCE_DIR=/path/to/local/vlfeat/checkout`.

## Tuning

- `PopSift_VLFEAT_MAX_AVG_DESC_DIST`: average best-match descriptor distance
  above which the test fails. Defaults to `-1` (report only, never fail).
  Run once, look at the reported "Average best-match descriptor distance"
  line, then set this to something a bit above the measured value to turn it
  into a real regression gate.
- The exact `popsift-demo` flags used for the comparison live in
  `run_comparison.sh.in` - edit them there.
- `PopSift_VLFEAT_GIT_TAG` pins VLFeat's last commit (the upstream repo has
  been dormant since 2018). Bump it if that ever changes.

## Validation status of the patches (be aware before relying on these)

- **`patches/vlfeat-pgm-fix.patch`** (the `vl/pgm.c` bounds-check fix):
  matches the bug described in the `sycl` branch's `testScripts/README.txt`
  exactly, checked against a real clone of VLFeat's source. High confidence.

- **`patches/vlfeat-macos-arm64.patch`** (Apple Silicon `ARCH=arm64`
  support): built on real Apple Silicon hardware (an M3 Mac) by the PopSift
  maintainer - high confidence. It:
  - adds the `Darwin_arm64_ARCH := arm64` Makefile mapping;
  - does **not** add an arm64 branch to the STD_CFLAGS/STD_LDFLAGS section
    (unlike the maci64 branch, no `-isysroot`/`-mmacosx-version-min` are
    needed - Apple clang's own defaults are enough on Apple Silicon);
  - skips `include make/matlab.mak` / `include make/octave.mak` entirely
    rather than inventing MEX suffix values for an ARCH neither file knows
    about (they otherwise abort the build with an unconditional sanity
    check - see the comment in the patch);
  - drops `-undefined suppress` and `-isysroot $(SDKROOT)` from the dylib
    link step, but only for `ARCH=arm64` - `maci64` keeps both, via a pair
    of new Makefile variables gating those two flags, so this doesn't
    change Intel Mac build behavior.
  - `CMakeLists.txt` also passes `VER=0.9.21` explicitly on the `make`
    command line for this ARCH: VLFeat's own version-string extraction
    (`make/dist.mak`, a `sed` one-liner over `vl/generic.h`) came out empty
    at dylib-link time on the M3 test machine, which broke
    `-compatibility_version`/`-current_version`. Root cause not fully
    tracked down (the `sed` pattern looks portable on paper); passing it
    explicitly sidesteps it. `0.9.21` matches `VL_VERSION_STRING` in
    `vl/generic.h` at the pinned `PopSift_VLFEAT_GIT_TAG` - bump both
    together if that tag ever changes.

- **`patches/vlfeat-linux-arm64.patch`** (Linux ARM64 `ARCH=glnxarm64`
  support, e.g. for Jetson): actually built and iterated on real ARM64
  Linux hardware (Ubuntu 22.04, GCC 11, binutils 2.38), and simplified to
  match the same "skip matlab.mak/octave.mak" approach validated on macOS
  above. Every `.c` file in VLFeat compiles cleanly with this patch plus
  `DISABLE_SSE2=yes DISABLE_AVX=yes DISABLE_OPENMP=yes` (all wired into
  `CMakeLists.txt` for ARM targets - see the `DISABLE_OPENMP` note there for
  why it's on unconditionally). **The final `libvl.so` link step fails** on
  that system with:
  ```
  /usr/bin/ld: bin/glnxarm64/libvl.so: no symbol version section for
  versioned symbol `memcpy@GLIBC_2.2.5'
  ```
  This was narrowed down to linking just two of VLFeat's own object files
  together (`aib.o` + `array.o`, specifically) - not something introduced by
  this patch, not `-Wl,--as-needed` (removed, no change), not
  `-D_FORTIFY_SOURCE` (disabled, no change), and it reproduces with both
  `ld.bfd` and `ld.gold`. A hand-written two-file repro using the same
  compiler flags links fine, so it's specific to something in VLFeat's own
  object files rather than the toolchain in general. Root cause not found -
  worth a look if you have a Jetson or other Linux ARM64 box handy; a
  different binutils version would be the first thing to try.

## Known limitation

No Windows support - `BUILD_COMMAND make ...` assumes a Makefile-based
Unix build. VLFeat's Windows build uses `Makefile.mak` via `nmake`, not
covered here.
