# VLFeat reference build

Produces a local, corrected copy of VLFeat's `sift` command line tool, which
the PopSift/VLFeat comparison uses as its reference. Nothing is downloaded at
configure time; the build downloads VLFeat once and then reuses it.

What it does, in the `vlfeat_sift` target:

1. Fetches VLFeat from `PopSift_VLFEAT_GIT_REPOSITORY` at
   `PopSift_VLFEAT_GIT_TAG` (the last VLFeat commit, from 2018), or copies
   the checkout given in `PopSift_VLFEAT_SOURCE_OOT_DIR`. The original checkout is
   never modified.
2. Applies the patches in `patches/` with `git apply`, through
   `apply_patches.cmake`. A patch that is already in is skipped.
   `vlfeat-pgm-fix.patch` fixes a comparison in `vl/pgm.c` that makes the
   CLI reject every PGM file.
3. Builds only `sift` with `vlfeat-build/CMakeLists.txt` instead of VLFeat's
   Makefile, with SSE2, AVX and OpenMP disabled. This works with gcc, clang
   and MSVC; the executable ends up in `<build>/Testing/vlfeat/vlfeat/bin/`.

## Inside PopSift

Configured when `BUILD_TESTING` and `PopSift_TEST_WITH_VLFEAT` are both ON.
The normal build then also builds `sift`. To build only `sift`:

```sh
cmake --build build --target vlfeat_sift
```

## On its own

This directory is also a standalone CMake project, so VLFeat can be built on
a machine without CUDA or HIP:

```sh
cmake -S Testing/vlfeat -B build-vlfeat
cmake --build build-vlfeat
```

## Changing the patches

Patch files must keep LF line endings for `git apply`; `.gitattributes`
marks them `-text` so Windows checkouts with `core.autocrlf` leave them
alone. Add a new patch file here and to `_vlfeat_patches` in
`CMakeLists.txt`.
