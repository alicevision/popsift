# Analysis of comparison reports

Optional tools that draw the results of `Testing/scripts/popsift_compare.py`
and `compare_paramsets.py`. They need numpy and matplotlib:

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

- Left: descriptor distance of the geometrically matched keypoints. Solid
  for pairs with the same pose (at most 0.25 px and 2 degrees apart), dashed
  for all matched pairs. Same pose should mean nearly the same descriptor,
  so the solid curve should rise steeply near 0.
- Right: position distance from every keypoint to the keypoint with the most
  similar descriptor in the other run, solid for a in b, dashed for b in a,
  on a log scale beyond 0.1 px. The grey line is the smallest match radius
  (1 px). Every keypoint right of it has its most similar descriptor at a
  different place, which indicates a serious descriptor problem.

## unmatched.png

Written by `popsift_compare.py` into every image folder
(`<set>/<image>/<a>__vs__<b>/unmatched.png`) when matplotlib is available.
It shows the features that only one of the runs found, with their counts in
the top corners.

## inspect_matches.py

```
python3 Testing/analysis/inspect_matches.py \
        build/Testing/compareVlfeat/run/oxford/boat/report_vlfeat-emulation__vs__vlfeat
```

Opens a report folder and shows one image at a time with the matched pairs
whose difference exceeds a threshold, both glyphs joined by a white line.

- The slider sets the threshold.
- The mode buttons switch between position difference (pixels), rotation
  difference (degrees) and descriptor distance.
- `<` and `>`, or the arrow keys, step through the images.
- A click prints the `matched.csv` row of the nearest drawn pair.

## sift_plots.py

The drawing functions used by both, for your own scripts:
`read_pgm`, `glyphs`, `show_image` and `render_unmatched`.
