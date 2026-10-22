# Glossary

Terms used in the reports, plots and the explorer when PopSift is compared
with VLFeat. a is the run under test (PopSift), b the reference (VLFeat).

**Keypoint** (also feature). A point that SIFT finds in an image, with a
position (x, y), a scale (sigma) and an orientation, plus a descriptor. SIFT
often gives one position several orientations; each is its own keypoint.

**Descriptor.** 128 numbers that describe the image around a keypoint, stored
as bytes (0-255). Keypoints of the same image structure should have similar
descriptors.

**Matched.** A keypoint of a and one of b are paired by pixel distance and
scale, not by descriptor: they are at most max(1 px, 0.5 sigma) apart (the
matching radius), their scales differ by at most a factor 1.5, and among
several candidates the closest in position, then in orientation, wins. Every
keypoint is paired at most once.

**Only PopSift / only VLFeat** (only_a / only_b). Keypoints that one run found
and the other did not, at least not close enough to be matched.

**Recall.** Matched keypoints / keypoints of VLFeat: how much of the reference
PopSift reproduces.

**Precision.** Matched keypoints / keypoints of PopSift: how much of what
PopSift finds is confirmed by VLFeat.

**Pixel distance** (`pos_dist_px`). Distance between the centres of two
keypoints, in pixels of the input image.

**Offset x, y** (`dx_px`, `dy_px`). Position of b minus position of a.

**Scale ratio.** Scale (sigma) of b divided by that of a; 1 is equal.

**Rotation difference** (`ori_diff_deg`). Orientation of b minus orientation
of a in degrees, wrapped to -180..180. `abs_ori_diff_deg` is its size.

**Descriptor distance (L2)** (`desc_dist_l2`). The Euclidean distance between
two descriptors: the square root of the sum of the squared differences of the
128 values, in byte units. 0 means identical; 1.41 means that two values
differ by one. A rotation difference also rotates the descriptor, so larger
rotation differences give larger descriptor distances.

**Same pose.** Matched pairs at most 0.25 px and 2 degrees apart. Their
descriptors should be nearly identical; their descriptor distance shows how
closely PopSift computes VLFeat's descriptor.

**Most similar descriptor** (descriptor nearest neighbour). For a keypoint of
one run, the keypoint of the other run with the smallest descriptor distance,
searched over the whole image.

**Nearest-descriptor agreement** (`nn_agreement_a`, `nn_agreement_b`). Share
of the matched keypoints whose most similar descriptor in the other run is
their matched keypoint. Close to 1 is good. A keypoint whose most similar
descriptor sits somewhere else in the image would be matched wrongly by an
application.

**Nearest descriptor in place** (`nn_in_place_a`, `nn_in_place_b`).
Keypoints whose most similar descriptor in the other run belongs to a
keypoint within the matching radius. An application that pairs keypoints by
their most similar descriptor pairs these correctly.

**Distinctive** (`nn_distinctive_a`, `nn_distinctive_b`). Those keypoints with
the nearest descriptor in place that also pass Lowe's ratio test: the most
similar descriptor distance is less than 0.8 times the second most similar.
An application that applies Lowe's test keeps these pairs; the others are
correct but rejected as ambiguous.

**Descriptor cells and bins.** The descriptor divides the area around the
keypoint into 4 x 4 cells, and the gradient directions in every cell into 8
bins of 45 degrees. Each of the 128 values is one bin of one cell.

**Median, p90, p99.** Half, 90% and 99% of the values are at most this large.

**CDF** (cumulative distribution). For every value on the x axis, the share
of keypoints whose value is at most that large. A curve that rises steeply
near 0 means small differences.
