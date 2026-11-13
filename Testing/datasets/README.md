# Test image sets

The comparison scripts in `Testing/scripts` run on image sets that configure
finds on disk when `BUILD_TESTING` is ON. Configure never downloads anything,
and prints the sets it found:

```text
-- Test image sets:
--   handcrafted: 4 images in /src/popsift/Testing/testImages
--   oxford/boat: 6 images in /data/popsift-testdata/oxford/boat
--   oxford/graf: 6 images in /data/popsift-testdata/oxford/graf
```

## The sets

- `handcrafted`: small, purpose-made images in `Testing/testImages`, for
  spotting obvious artifacts. Always used.
- Every folder below `PopSift_TESTDATA_DIR` that holds .pgm or .ppm files,
  named by its path relative to `PopSift_TESTDATA_DIR`: images in
  `<dir>/oxford/boat/` are the set `oxford/boat`. Each set has only the images
  directly in its folder; a subfolder with images is a set of its own.

`PopSift_TESTDATA_DIR` defaults to the environment variable
`POPSIFT_TESTDATA_DIR`. Adding or removing images or folders needs no new
configure: the next build notices and re-runs it.

To add a hand-crafted image, drop a .pgm or .ppm file into
`Testing/testImages`. If it is expected to yield no features at all, add it
to `expectations.json` in its folder with `"allow_empty": true`; this works
for every set.

## The Oxford sets

The Oxford affine-covariant region sets (bark, bikes, boat, graf, leuven,
trees, ubc, wall) are published by the Visual Geometry Group, University of
Oxford, for research use, and are not redistributed with PopSift.

```shell
python3 Testing/datasets/download_oxford.py /data/popsift-testdata          # all eight
python3 Testing/datasets/download_oxford.py /data/popsift-testdata boat graf
```

downloads them into `<dir>/oxford/<set>/` and checks each archive against
its SHA256. A set that already has images is skipped. `--url` takes another
location of the archives, with `{}` for the set name, if the Oxford server
is unavailable.
