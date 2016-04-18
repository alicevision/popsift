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

IMAGE1=../sample/boat/img1.ppm
IMAGE2=../sample/boat/img2.ppm
IMAGE3=../sample/boat/img3.ppm
IMAGE4=../sample/boat/img4.ppm
IMAGE5=../sample/boat/img5.ppm
IMAGE6=../sample/boat/img6.ppm
PARAMS="--downsampling=0 --sigma=0.82 --octaves=3 --threshold=0.1 --edge-threshold=50.0 --vlfeat-mode" # finds 4 points
# PARAMS="--downsampling=0 --octaves=3 --sigma=0.82  --threshold=0.1 --edge-threshold=50.0 --vlfeat-mode" # finds 4 points
# PARAMS="--downsampling=0 --threshold=0.1 --edge-threshold=50.0" # finds 4 points
# LOG=--log
LOG=
rm -rf 1 2 3 4 5 6 dir-*

echo ./sift_v4 $PARAMS --dd $LGO $IMAGE1
./sift_v4 $PARAMS --dd $LGO $IMAGE1
mkdir 1
mv dir-* 1/

echo ./sift_v4 $PARAMS --dd $LGO $IMAGE2
./sift_v4 $PARAMS --dd $LGO $IMAGE2
mkdir 2
mv dir-* 2/

echo ./sift_v4 $PARAMS --dd $LGO $IMAGE3
./sift_v4 $PARAMS --dd $LGO $IMAGE3
mkdir 3
mv dir-* 3/

echo ./sift_v4 $PARAMS --dd $LGO $IMAGE4
./sift_v4 $PARAMS --dd $LGO $IMAGE4
mkdir 4
mv dir-* 4/

echo ./sift_v4 $PARAMS --dd $LGO $IMAGE5
./sift_v4 $PARAMS --dd $LGO $IMAGE5
mkdir 5
mv dir-* 5/

echo ./sift_v4 $PARAMS --dd $LGO $IMAGE6
./sift_v4 $PARAMS --dd $LGO $IMAGE6
mkdir 6
mv dir-* 6/

cat 1/dir-desc/* > img1_popSIFT.txt
cat 2/dir-desc/* > img2_popSIFT.txt
cat 3/dir-desc/* > img3_popSIFT.txt
cat 4/dir-desc/* > img4_popSIFT.txt
cat 5/dir-desc/* > img5_popSIFT.txt
cat 6/dir-desc/* > img6_popSIFT.txt
zip -r descriptors.zip img* ?/dir-desc/desc-pyramid-o-*

# output names: <img_name>_popSIFT.txt
# run VLFeat mode and OpenCV mode
# make a plot of nearest neighbours between 2 images
# distance must be less than 0.8

