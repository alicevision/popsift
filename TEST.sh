#!/bin/bash

# IMAGE1=/local/home/griff/GIT/github/openmvg/src/openMVG_Samples/imageData/StanfordMobileVisualSearch/Ace_40p_gray.pgm
# IMAGE1=/local/home/griff/GIT/github/openmvg/build/gray.pgm

# For a package to compare with OpenCV and VLFeat
IMAGE1=../sample/boat/img1.ppm
IMAGE2=../sample/boat/img2.ppm
IMAGE3=../sample/boat/img3.ppm
IMAGE4=../sample/boat/img4.ppm
IMAGE5=../sample/boat/img5.ppm
IMAGE6=../sample/boat/img6.ppm

# For testing edge effects on a tiny hand-crafted image
# IMAGE1=../sample/box-6x6.pgm

# For running with default parameters, assuming an unblurred input image
# PARAMS=

# PARAMS="--indirect-unfiltered --octaves=4 --threshold=0.1 --edge-threshold=10.0"
# PARAMS="--indirect-unfiltered --threshold=0.04 --edge-threshold=10.0" # finds 4 points
# PARAMS="--downsampling=0 --octaves=4 --indirect-unfiltered --threshold=0.04 --edge-threshold=10.0"
# PARAMS="--octaves=8 --indirect-unfiltered --threshold=0.04 --edge-threshold=10.0 --initial-blur=0.5 --print-gauss-tables"
# PARAMS="--popsift-mode --octaves=8 --indirect-unfiltered --threshold=0.04 --edge-threshold=10.0 --initial-blur=0.5"
# PARAMS="--vlfeat-mode --octaves=8 --indirect-unfiltered --threshold=0.04 --edge-threshold=10.0 --initial-blur=0.5"
PARAMS="--opencv-mode --octaves=8 --indirect-unfiltered --threshold=0.04 --edge-threshold=10.0 --initial-blur=0.5"
# PARAMS="--popsift-mode --octaves=8 --indirect-unfiltered --threshold=0.04 --edge-threshold=10.0 --initial-blur=0.5 --downsampling=0"
# PARAMS="--octaves=4 --indirect-unfiltered --threshold=0.04 --edge-threshold=10.0 --initial-blur=0.5 --bemap-orientation"
# PARAMS="--octaves=4 --indirect-unfiltered --threshold=0.04 --edge-threshold=10.0"
# PARAMS="--vlfeat-mode --sigma=0.82 --octaves=1 --levels=1 --downsampling=0 --indirect-unfiltered --threshold=0.04 --edge-threshold=10.0 --initial-blur=0.5"
# PARAMS="--sigma=1.6 --octaves=4 --levels=3 --downsampling=0 --indirect-unfiltered --threshold=0.0 --edge-threshold=10.0"
# PARAMS="--popsift-mode --downsampling=0 --octaves=8 --indirect-unfiltered --threshold=0.04 --edge-threshold=10.0 --initial-blur=0.5"

LOG=--log
# LOG=

rm -rf outputs

mkdir -p outputs/popsift

# for i in $IMAGE1 ; do
for i in $IMAGE1 $IMAGE2 $IMAGE3 $IMAGE4 $IMAGE5 $IMAGE6 ; do
  if [ -f "$i" ] ; then
    outname=`basename --suffix=.ppm $i`
    # outname=`basename --suffix=.pgm $i`
    echo ./sift_v4 $PARAMS $LOG $i
    ./sift_v4 $PARAMS $LOG $i
    if [ ! -z "$LOG" ] ; then
      mkdir -p outputs/${outname}
      mv dir-* outputs/${outname}/
      echo "128" > outputs/popsift/${outname}.sift
      wc -l  outputs/${outname}/dir-desc/* | tail -1 | awk '{print $1;}' >> outputs/popsift/${outname}.sift
      cat  outputs/${outname}/dir-desc/* >> outputs/popsift/${outname}.sift
      echo " "
    fi
  fi
done

if [ ! -z "$LOG" ] ; then
  ( cd outputs; zip -r popsift.zip popsift )
fi
# zip -r descriptors.zip popsift/img* ?/dir-desc/desc-pyramid-o-*

# output names: <img_name>_popSIFT.txt
# run VLFeat mode and OpenCV mode
# make a plot of nearest neighbours between 2 images
# distance must be less than 0.8

