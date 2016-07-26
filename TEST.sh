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

IMAGE1=/local/home/griff/GIT/github/openmvg/src/openMVG_Samples/imageData/StanfordMobileVisualSearch/Ace_40p_gray.pgm
# IMAGE1=/local/home/griff/GIT/github/openmvg/build/gray.pgm

# IMAGE1=../sample/boat/img1.ppm
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
# PARAMS="--octaves=8 --indirect-unfiltered --threshold=0.04 --edge-threshold=10.0 --initial-blur=0.5 --print-gauss-tables"
PARAMS="--octaves=8 --indirect-unfiltered --threshold=0.04 --edge-threshold=10.0 --initial-blur=0.5"
# PARAMS="--octaves=4 --indirect-unfiltered --threshold=0.04 --edge-threshold=10.0 --initial-blur=0.5 --bemap-orientation"
# PARAMS="--octaves=4 --indirect-unfiltered --threshold=0.04 --edge-threshold=10.0"
# PARAMS="--vlfeat-mode --sigma=0.82 --octaves=1 --levels=1 --downsampling=0 --indirect-unfiltered --threshold=0.04 --edge-threshold=10.0 --initial-blur=0.5"
# PARAMS="--sigma=1.6 --octaves=4 --levels=3 --downsampling=0 --indirect-unfiltered --threshold=0.0 --edge-threshold=10.0"
LOG=--log
# LOG=
rm -rf outputs

mkdir -p outputs/popsift

# for i in $IMAGE1 $IMAGE2 $IMAGE3 $IMAGE4 $IMAGE5 $IMAGE6 ; do
for i in $IMAGE1 ; do
  if [ -f "$i" ] ; then
    outname=`basename --suffix=.ppm $i`
    echo ./sift_v4 $PARAMS $LOG $i
    ./sift_v4 $PARAMS $LOG $i
    mkdir -p outputs/${outname}
    mv dir-* outputs/${outname}/
    echo "128" > outputs/popsift/${outname}.sift
    wc -l  outputs/${outname}/dir-desc/* | tail -1 | awk '{print $1;}' >> outputs/popsift/${outname}.sift
    cat  outputs/${outname}/dir-desc/* >> outputs/popsift/${outname}.sift
    echo " "

  fi
done

( cd outputs; zip -r popsift.zip popsift )
# zip -r descriptors.zip popsift/img* ?/dir-desc/desc-pyramid-o-*

# output names: <img_name>_popSIFT.txt
# run VLFeat mode and OpenCV mode
# make a plot of nearest neighbours between 2 images
# distance must be less than 0.8

