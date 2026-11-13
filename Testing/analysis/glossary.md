# Glossary

Terms used in the reports and plots when PopSift is compared
with VLFeat. a is the run under test (PopSift), b the reference (VLFeat).

**Keypoint** (also feature). A point that SIFT finds in an image, with a
position (x, y), a scale (sigma) and an orientation, plus a descriptor. SIFT
often gives one position several orientations; each is its own keypoint.

**Descriptor.** 128 numbers that describe the image around a keypoint, stored
as bytes (0-255). Keypoints of the same image structure should have similar
descriptors.

**Co-located.** A keypoint of a and one of b are paired by pixel distance
and scale only, not by orientation or descriptor: they are at most
max(1 px, 0.5 sigma) apart (the pairing radius), their scales differ by at
most a factor 1.5, and among several candidates the closest in position, then
in orientation, wins. Every keypoint is paired at most once. Co-located
keypoints show that the pyramid, the DoG and the extrema detection of the two
runs agree. This is not a match in the sense of SIFT, where keypoints are
matched by their descriptors.

**Agreement level** (`agreement` in `colocated.csv`). How far a co-located
pair agrees beyond position and scale. Each level includes the ones before:

- `position`: the orientations differ by more than 2 degrees (orientation
  differs). The orientation computation differs, and since the descriptor is
  computed relative to the orientation, nothing can be concluded about the
  descriptor extraction.
- `orientation`: the same orientation, at most 2 degrees apart, but a
  descriptor distance (L2) above 12 (descriptor differs). A strong hint at a
  difference in the descriptor extraction.
- `descriptor`: also a similar descriptor, descriptor distance (L2) at most
  12. 12 is a little more than the distance of two descriptors whose 128
  values all differ by 1.

The descriptor is judged in absolute terms against the descriptor of the
co-located keypoint, not by whether it is the most similar one in the image.

**Best at the position** (`best_at_position` in `colocated.csv`). SIFT
gives one position and scale several orientations, each its own keypoint.
This is the highest agreement level among all pairs whose keypoint of a has
the same position and scale. A pair whose orientation or descriptor differs
while another pair at its position agrees further is a minor problem
(`orientation_differs_minor`, `descriptor_differs_minor`): an orientation
or a descriptor that only one run has, while the extremum and at least one
orientation and descriptor agree. The others are the ones to investigate.

**Only PopSift / only VLFeat** (only_a / only_b, unpaired). Keypoints that
one run found and the other did not, at least not close enough to be
co-located.

**Recall.** Co-located keypoints / keypoints of VLFeat: how much of the
reference PopSift reproduces.

**Precision.** Co-located keypoints / keypoints of PopSift: how much of what
PopSift finds is confirmed by VLFeat.

**Orientation agreement** (`ori_agreement`). Pairs with the same orientation
/ co-located pairs.

**Descriptor agreement** (`desc_agreement`). Pairs with a similar descriptor
/ pairs with the same orientation.

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

**Most similar descriptor** (descriptor nearest neighbour). For a keypoint of
one run, the keypoint of the other run with the smallest descriptor distance,
searched over the whole image.

**Nearest-descriptor agreement** (`nn_agreement_a`, `nn_agreement_b`). Share
of the co-located keypoints whose most similar descriptor in the other run is
their co-located keypoint. A keypoint whose most similar descriptor sits
somewhere else in the image would be matched wrongly by an application. In a
repetitive image that can happen to correct descriptors, so this is
secondary to the descriptor agreement.

**Nearest descriptor in place** (`nn_in_place_a`, `nn_in_place_b`).
Keypoints whose most similar descriptor in the other run belongs to a
keypoint within the pairing radius. An application that pairs keypoints by
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
