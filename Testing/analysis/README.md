# Analysis of comparison reports

Optional tools that draw and explore the results of
`Testing/scripts/popsift_compare.py` and `compare_paramsets.py`. They need numpy and matplotlib:

```
python3 -m pip install -r Testing/analysis/requirements.txt
```

The comparison scripts and CTest work without them; only the pictures are
missing then.

Keypoints are drawn as SIFT glyphs: a circle of radius 3 sigma and a line
from the centre in the direction of the orientation. The run under test (a)
is orange, the reference (b) blue, a colour-blind safe pair.

## cdf.png

Written by `popsift_compare.py` into every image folder and the report folder,
and by `compare_paramsets.py` into every report folder plus
`cdf_all_pairs.png` with one colour per pair. Two cumulative distributions:

- Left: descriptor distance (L2) of the matched keypoints. Solid
  for pairs with the same pose (at most 0.25 px and 2 degrees apart), dashed
  for all matched pairs. Same pose should mean nearly the same descriptor,
  so the solid curve should rise steeply near 0.
- Right: pixel distance from every keypoint to the keypoint with the most
  similar descriptor in the other run, solid for a in b, dashed for b in a,
  on a log scale from 0.01 px (smaller distances are drawn at 0.01 px). The
  grey line is the smallest match radius
  (1 px). Every keypoint right of it has its most similar descriptor at a
  different place, which indicates a serious descriptor problem.

## sorted.png, descriptor_bins.png and image_heatmap.png

The plots of the 2021 comparison scripts, redone on the match files.
`popsift_compare.py` writes `sorted.png` and `descriptor_bins.png` into every
image folder and the report folder (pooled over the images of the set), and
`image_heatmap.png` into every image folder. All logarithmic axes are
logarithmic over their whole range; values outside are drawn at the ends.

- `sorted.png`: left, for every keypoint of a, the descriptor distance (L2)
  to the most similar descriptor of b (line) and to the second most similar
  (dots), sorted by the first. The gap between the two is what Lowe's ratio
  test relies on. Middle and right: the pixel distances and the rotation
  differences of the matched pairs, sorted.
- `descriptor_bins.png`: the 128 descriptor values as 8 orientation bins over
  the 4 x 4 cells of the descriptor, in VLFeat's order, with the mean
  difference b - a per value. Rows: mean absolute and mean signed difference
  of the pairs with the same pose, and mean absolute difference of all
  matched pairs. In the signed row, orange means a is larger, blue b. The
  sums come from `desc_bins.csv`, which `popsift-compareSiftFiles` writes.
- `image_heatmap.png`: the matched pairs in cells over the input image, by
  the position of a's keypoint: the share of pairs whose rotation differs by
  more than 2 degrees, and the 90th percentile of the rotation difference and
  of the descriptor distance. Differences that depend on the location, such
  as the handling of the image border, show up here.

## unmatched.png

Written by `popsift_compare.py` into every image folder
(`<set>/<image>/<a>__vs__<b>/unmatched.png`) when matplotlib is available.
It shows the keypoints that only one of the runs found, with their counts in
the top corners.

## inspect_matches.py

```
python3 Testing/analysis/inspect_matches.py \
        build/Testing/compareVlfeat/run/oxford_boat/report_vlfeat-emulation__vs__vlfeat
```

Opens a report folder and shows one image at a time with the matched pairs
whose difference exceeds a threshold, both glyphs joined by a white line.

- The slider sets the threshold.
- The mode buttons switch between pixel distance, rotation difference
  (degrees) and descriptor distance (L2).
- `<` and `>`, or the arrow keys, step through the images.
- A click prints the values of the nearest drawn pair.

## explorer.py

```
python3 Testing/analysis/explorer.py build/Testing/compareVlfeat/run
```

One window for all reports of a run: the image sets and their images on the
left, and for the selected image its key figures, the distributions of
rotation difference and descriptor distance, a scatter plot of rotation
difference against descriptor distance, the set's figures, the glossary,
and `unmatched.png`. A button opens `inspect_matches.py` for the image in its
own window. `--save <png>` (with `--set`, `--image`, `--tab`) writes the
window to a file instead.

## Explorer package for project partners

```
python3 Testing/analysis/make_explorer_package.py build/Testing/compareVlfeat/run \
        -o popsift-explorer.zip
```

Packs the explorer, the match inspector, `glossary.md` and the results of a
run into one zip file that needs only Python with numpy and matplotlib: the
reports, `cdf.png` and `unmatched.png`, the input images, and per image
`matches.npz`, the matched pairs as float32 arrays for the inspector. The CSV
files and the feature files stay out; a run of all Oxford sets gives about
70 MB. After unpacking, `python3 explorer.py` opens it.

`glossary.md` explains the terms of the reports, plots and the explorer.

## sift_plots.py

The drawing functions used by both, for your own scripts:
`read_pgm`, `glyphs`, `show_image` and `render_unmatched`.
