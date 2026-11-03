# Test image sets

The tests run on one or more image sets. Configure never downloads anything:
it prints which sets are present and how to get the missing ones.

```
-- Test image sets (PopSift_TESTDATA_DIR = /data/popsift-testdata):
--   handcrafted     present  3 images, Testing/testImages
--   oxford/boat     present  /data/popsift-testdata/oxford/boat
--   oxford/graf     missing  skipped: build target popsift_testdata, or set -DPopSift_TESTDATA_DOWNLOAD=ON
```

## The sets

| Set | Where | Selected by |
| --- | --- | --- |
| `handcrafted` | `Testing/testImages`, in the repository | always |
| `<group>/<set>`, e.g. `oxford/boat` | `PopSift_TESTDATA_DIR/<group>/<set>/` | `PopSift_TEST_DATASETS` |
| `custom` | any folder of .pgm/.ppm files | `PopSift_TEST_IMAGES_OOT_DIR` |

The hand-crafted images are small, purpose-made images for spotting obvious
artifacts. Add one by dropping a .pgm or .ppm file into `Testing/testImages`.
If the image is expected to yield no features at all, add it to
`Testing/testImages/expectations.json` with `"allow_empty": true`.

The Oxford affine-covariant region sets (bark, bikes, boat, graf, leuven,
trees, ubc, wall) are not redistributed with PopSift; they are downloaded from
the Visual Geometry Group's site when asked for.

## Options

- `PopSift_TEST_DATASETS`: groups or sets, e.g. `oxford` (all eight) or
  `oxford/boat;oxford/graf`. Default empty. Old-style bare names such as
  `boat` still work.
- `PopSift_TESTDATA_DIR`: where downloadable sets live. Defaults to the
  environment variable `POPSIFT_TESTDATA_DIR`, else `<build>/testdata`. Point
  it at an existing or shared copy and every build tree reuses it. A set
  counts as present when all its images exist below its folder, as .pgm or
  .ppm, however they got there.
- `PopSift_TESTDATA_DOWNLOAD` (default OFF): let `ctest` download missing
  selected sets before the tests that use them.
- `PopSift_TEST_IMAGES_OOT_DIR`: adds a folder of your own images as the set
  `custom`. `PopSift_TESTFILE_PATH` still works for this but is deprecated.

## Downloading

```
cmake --build build --target popsift_testdata              # all selected sets
cmake --build build --target popsift_testdata_oxford_boat  # one set
```

The download is checked against the SHA256 in the set's descriptor before it
is unpacked. A finished set has a stamp file and is never fetched again; an
interrupted download starts over. A folder that holds images not placed there
by a download is never deleted or overwritten.

## Adding a set

Add `Testing/datasets/<group>/<set>.cmake`, e.g.:

```cmake
set(DATASET_URL     "https://example.org/sets/myset.tar.gz")
set(DATASET_SHA256  "<sha256 of the archive>")
set(DATASET_SIZE    "")                  # archive size in bytes, optional
set(DATASET_IMAGES  img1 img2 img3)      # base names, .pgm or .ppm
set(DATASET_LICENSE "Who publishes it and under which terms.")
```

A new folder is a new group, selectable as a whole by its name. No CMake code
needs to change.
