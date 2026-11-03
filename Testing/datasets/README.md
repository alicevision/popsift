# Test images

`Testing/testImages` holds small, purpose-made images for spotting obvious
artifacts. If an image is expected to yield no features at all, it is listed
in `expectations.json` in its folder with `"allow_empty": true`:
`box-12x12-symm.pgm` is perfectly symmetric, `box-12x12.pgm` breaks the
symmetry and yields keypoints.

## The Oxford sets

The Oxford affine-covariant region sets (bark, bikes, boat, graf, leuven,
trees, ubc, wall) are published by the Visual Geometry Group, University of
Oxford, for research use, and are not redistributed with PopSift.

```
python3 Testing/datasets/download_oxford.py /data/popsift-testdata          # all eight
python3 Testing/datasets/download_oxford.py /data/popsift-testdata boat graf
```

downloads them into `<dir>/oxford/<set>/` and checks each archive against
its SHA256. A set that already has images is skipped. `--url` takes another
location of the archives, with `{}` for the set name, if the Oxford server
is unavailable.
