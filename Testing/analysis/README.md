# Analysis of comparison reports

Tools that draw and inspect the results of
`Testing/scripts/popsift_compare.py` and `compare_paramsets.py`. They need numpy and matplotlib:

```shell
python3 -m pip install -r Testing/analysis/requirements.txt
```

The comparison scripts work without them; only the pictures are missing
then.

Keypoints are drawn as SIFT glyphs: a circle of radius 3 sigma and a line
from the centre in the direction of the orientation. The run under test (a)
is orange, the reference (b) blue, a colour-blind safe pair.

## agreement.png

Written by `popsift_compare.py` into every image folder and the report folder,
and by `compare_paramsets.py` into every report folder. It shows the three
agreement levels of the co-located keypoints (see `glossary.md`):

- Left: every co-located pair by its rotation difference (x) and descriptor
  distance (L2, y), both on log scales. The grey lines are the tolerances
  (2 degrees, 12) and split the plot into what a tester can conclude: right
  of the orientation tolerance (vermillion) the orientation computation
  differs and the descriptor cannot be judged; left of it and above the
  descriptor tolerance (reddish purple) the descriptor extraction is
  suspect; below it (green) all agree. The descriptor distance grows with
  the rotation difference, so a cloud that rises along the diagonal points
  to the orientation, a cloud high up on the left to the descriptor.
- Right: the keypoints of a and of b split into unpaired and the three
  levels.

  In both, pairs whose orientation or descriptor differs while another
  keypoint at the same position agrees further are drawn hollow (left) or
  hatched (right). They are a minor problem: one run has an orientation or
  descriptor more or less than the other at a position where both agree
  otherwise. Filled ones are the ones to investigate.

## cdf.png

Written by `popsift_compare.py` into every image folder and the report folder,
and by `compare_paramsets.py` into every report folder plus
`cdf_all_pairs.png` with one colour per pair. Two cumulative distributions:

- Left: descriptor distance (L2) of the co-located keypoints. Solid for the
  pairs with the same orientation (at most 2 degrees apart), dashed for all
  co-located pairs. The same orientation should mean nearly the same
  descriptor, so the solid curve should rise steeply near 0.
- Right: pixel distance from every keypoint to the keypoint with the most
  similar descriptor in the other run, solid for a in b, dashed for b in a,
  on a log scale from 0.01 px (smaller distances are drawn at 0.01 px). The
  grey line is the smallest pairing radius
  (1 px). Every keypoint right of it has its most similar descriptor at a
  different place, which indicates a serious descriptor problem.

## sorted.png, descriptor_bins.png and image_heatmap.png

The plots of the 2021 comparison scripts, redone on the pair files.
`popsift_compare.py` writes `sorted.png` and `descriptor_bins.png` into every
image folder and the report folder (pooled over the images of the set), and
`image_heatmap.png` into every image folder. All logarithmic axes are
logarithmic over their whole range; values outside are drawn at the ends.

- `sorted.png`: left, for every keypoint of a, the descriptor distance (L2)
  to the most similar descriptor of b (line) and to the second most similar
  (dots), sorted by the first. The gap between the two is what Lowe's ratio
  test relies on. Middle and right: the pixel distances and the rotation
  differences of the co-located pairs, sorted.
- `descriptor_bins.png`: the 128 descriptor values as 8 orientation bins over
  the 4 x 4 cells of the descriptor, in VLFeat's order, with the mean
  difference b - a per value. Rows: mean absolute and mean signed difference
  of the pairs with the same orientation, and mean absolute difference of
  all co-located pairs. In the signed row, orange means a is larger, blue b. The
  sums come from `desc_bins.csv`, which `popsift-compareSiftFiles` writes.
- `image_heatmap.png`: the co-located pairs in cells over the input image,
  by the position of a's keypoint: the share of pairs whose orientation
  differs (more than 2 degrees), and the 90th percentile of the rotation difference and
  of the descriptor distance. Differences that depend on the location, such
  as the handling of the image border, show up here.

## unpaired.png

Written by `popsift_compare.py` into every image folder
(`<set>/<image>/<a>__vs__<b>/unpaired.png`) when matplotlib is available.
It shows the keypoints that only one of the runs found, that is, without a
co-located keypoint in the other run, with their counts in the top corners.

## inspect_matches.py

```shell
python3 Testing/analysis/inspect_matches.py \
        build/Testing/run/oxford_boat/report_popsift-vlfeat__vs__vlfeat
```

Opens a report folder and shows one image at a time with the co-located pairs
whose difference exceeds a threshold, both glyphs joined by a white line.

- The slider sets the threshold.
- The mode buttons switch between pixel distance, rotation difference
  (degrees) and descriptor distance (L2).
- `<` and `>`, or the arrow keys, step through the images.
- A click prints the values of the nearest drawn pair.

`glossary.md` explains the terms of the reports and plots.

## sift_plots.py

The drawing functions used by the scripts and the inspector, for your own
scripts:
`read_pgm`, `glyphs`, `show_image`, `read_pairs` and `render_unpaired`.
