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

# IMAGE1=/local/home/griff/GIT/github/openmvg/src/openMVG_Samples/imageData/StanfordMobileVisualSearch/Ace_40p_gray.pgm
# IMAGE1=/local/home/griff/GIT/github/openmvg/build/gray.pgm

IMAGE1=../sample/boat/img1.ppm
IMAGE2=../sample/boat/img2.ppm
IMAGE3=../sample/boat/img3.ppm
IMAGE4=../sample/boat/img4.ppm
IMAGE5=../sample/boat/img5.ppm
IMAGE6=../sample/boat/img6.ppm
# IMAGE7=../sample/level1.ppm
# PARAMS="--sigma=0.82 --octaves=3 --threshold=0.1 --edge-threshold=10.0 --vlfeat-mode" # finds 4 points
# PARAMS="--downsampling=0 --octaves=3 --sigma=0.82  --threshold=0.1 --edge-threshold=10.0 --vlfeat-mode" # finds 4 points
# PARAMS="--indirect-unfiltered --octaves=4 --threshold=0.1 --edge-threshold=10.0" # finds 4 points
# PARAMS="--indirect-unfiltered --threshold=0.04 --edge-threshold=10.0" # finds 4 points
# PARAMS="--downsampling=0 --octaves=4 --indirect-unfiltered --threshold=0.04 --edge-threshold=10.0"
PARAMS="--octaves=8 --indirect-unfiltered --threshold=0.04 --edge-threshold=10.0 --initial-blur=0.5"
# PARAMS="--octaves=4 --indirect-unfiltered --threshold=0.04 --edge-threshold=10.0 --initial-blur=0.5 --bemap-orientation"
# PARAMS="--octaves=4 --indirect-unfiltered --threshold=0.04 --edge-threshold=10.0"
# PARAMS="--vlfeat-mode --sigma=0.82 --octaves=1 --levels=1 --downsampling=0 --indirect-unfiltered --threshold=0.04 --edge-threshold=10.0 --initial-blur=0.5"
# PARAMS="--sigma=1.6 --octaves=4 --levels=3 --downsampling=0 --indirect-unfiltered --threshold=0.0 --edge-threshold=10.0"
LOG=--log
# LOG=
rm -rf 1 2 3 4 5 6 7 dir-* popsift popsift.zip

mkdir popsift

echo ./sift_v4 $PARAMS $LOG $IMAGE1
./sift_v4 $PARAMS $LOG $IMAGE1
mkdir 1
mv dir-* 1/
echo "128" > popsift/img1.sift
wc -l 1/dir-desc/* | head -1 | awk '{print $1;}' >> popsift/img1.sift
cat 1/dir-desc/* >> popsift/img1.sift

echo ./sift_v4 $PARAMS $LOG $IMAGE2
./sift_v4 $PARAMS $LOG $IMAGE2
mkdir 2
mv dir-* 2/
echo "128" > popsift/img2.sift
wc -l 2/dir-desc/* | head -1 | awk '{print $1;}' >> popsift/img2.sift
cat 2/dir-desc/* >> popsift/img2.sift

echo ./sift_v4 $PARAMS $LOG $IMAGE3
./sift_v4 $PARAMS $LOG $IMAGE3
mkdir 3
mv dir-* 3/
echo "128" > popsift/img3.sift
wc -l 3/dir-desc/* | head -1 | awk '{print $1;}' >> popsift/img3.sift
cat 3/dir-desc/* >> popsift/img3.sift

echo ./sift_v4 $PARAMS $LOG $IMAGE4
./sift_v4 $PARAMS $LOG $IMAGE4
mkdir 4
mv dir-* 4/
echo "128" > popsift/img4.sift
wc -l 4/dir-desc/* | head -1 | awk '{print $1;}' >> popsift/img4.sift
cat 4/dir-desc/* >> popsift/img4.sift

echo ./sift_v4 $PARAMS $LOG $IMAGE5
./sift_v4 $PARAMS $LOG $IMAGE5
mkdir 5
mv dir-* 5/
echo "128" > popsift/img5.sift
wc -l 5/dir-desc/* | head -1 | awk '{print $1;}' >> popsift/img5.sift
cat 5/dir-desc/* >> popsift/img5.sift

echo ./sift_v4 $PARAMS $LOG $IMAGE6
./sift_v4 $PARAMS $LOG $IMAGE6
mkdir 6
mv dir-* 6/
echo "128" > popsift/img6.sift
wc -l 6/dir-desc/* | head -1 | awk '{print $1;}' >> popsift/img6.sift
cat 6/dir-desc/* >> popsift/img6.sift

zip -r popsift.zip popsift
# zip -r descriptors.zip popsift/img* ?/dir-desc/desc-pyramid-o-*

# output names: <img_name>_popSIFT.txt
# run VLFeat mode and OpenCV mode
# make a plot of nearest neighbours between 2 images
# distance must be less than 0.8

