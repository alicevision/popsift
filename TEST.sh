#!/bin/bash

# IMAGE=../sample/level0even.ppm
# PARAMS="--octaves=3 --levels=3 --sigma=0.82 --threshold=0.01 --edge-threshold=100.0 --vlfeat-mode" # finds the center and 2 out of 4
# PARAMS="--octaves=3 --levels=3 --sigma=1.6  --threshold=0.01 --edge-threshold=100.0 --vlfeat-mode" # finds 4 points
# PARAMS="--octaves=3 --levels=3 --sigma=0.82 --threshold=0.01 --edge-threshold=10.0 --vlfeat-mode" # finds the center
# PARAMS="--octaves=3 --levels=3 --sigma=1.6  --threshold=0.01 --edge-threshold=10.0 --vlfeat-mode" # finds 4 points
# PARAMS="--octaves=3 --levels=3 --sigma=0.82 --threshold=0 --edge-threshold=10.0 --vlfeat-mode" # finds the center
# PARAMS="--octaves=3 --levels=3 --sigma=1.6  --threshold=0 --edge-threshold=10.0 --vlfeat-mode" # finds 4 points

# IMAGE=../sample/level1.ppm
# IMAGE=../sample/lena.ppm

# PARAMS="--octaves=3 --levels=3 --sigma=0.82 --threshold=0.01 --edge-threshold=100.0 --vlfeat-mode"
# PARAMS="--octaves=3 --levels=3 --sigma=1.6  --threshold=0.01 --edge-threshold=100.0 --vlfeat-mode"
# PARAMS="--octaves=3 --levels=3 --sigma=0.82 --threshold=0.01 --edge-threshold=10.0 --vlfeat-mode" # lots of points
# PARAMS="--octaves=3 --levels=3 --sigma=0.82 --threshold=0.05 --edge-threshold=10.0 --vlfeat-mode" # lots of points
# PARAMS="--octaves=3 --levels=3 --sigma=1.6  --threshold=0.01 --edge-threshold=10.0 --vlfeat-mode" # 4 points, not good
# PARAMS="--octaves=3 --levels=3 --sigma=0.82 --threshold=0 --edge-threshold=10.0 --vlfeat-mode"
# PARAMS="--octaves=3 --levels=3 --sigma=1.6  --threshold=0 --edge-threshold=10.0 --vlfeat-mode"

# IMAGE=../sample/level0.ppm
# PARAMS="--octaves=3 --levels=3 --sigma=0.82  --threshold=0.01 --edge-threshold=10.0 --vlfeat-mode" # finds 4 points

# PARAMS="--octaves=3 --levels=3 --sigma=0.82 --threshold=0.00001 --edge-threshold=10.0 --vlfeat-mode"
# PARAMS="--octaves=3 --levels=3 --sigma=1.6 --threshold=0.00001 --edge-threshold=10.0"
# PARAMS="--octaves=3 --levels=3 --sigma=1.6 --threshold=0.00001 --edge-threshold=5.0"
# PARAMS=

IMAGE1=../sample/img1.ppm
IMAGE2=../sample/img2.ppm
PARAMS="--sigma=0.82  --threshold=0.1 --edge-threshold=50.0 --vlfeat-mode" # finds 4 points
rm -rf 1 2 dir-*

./sift_v4 \
	$PARAMS --log \
	$IMAGE1
mkdir 1
mv dir-* 1/

./sift_v4 \
	$PARAMS --log \
	$IMAGE2
mkdir 2
mv dir-* 2/

