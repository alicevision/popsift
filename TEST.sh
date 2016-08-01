#!/bin/bash

# IMAGE1=/local/home/griff/GIT/github/openmvg/src/openMVG_Samples/imageData/StanfordMobileVisualSearch/Ace_40p_gray.pgm
# IMAGE1=/local/home/griff/GIT/github/openmvg/build/gray.pgm

# For a package to compare with OpenCV and VLFeat
# IMAGE1=../sample/boat/img1.ppm
# IMAGE2=../sample/boat/img2.ppm
# IMAGE3=../sample/boat/img3.ppm
# IMAGE4=../sample/boat/img4.ppm
# IMAGE5=../sample/boat/img5.ppm
# IMAGE6=../sample/boat/img6.ppm

IMAGE1=../sample/big_set/boat/img3.ppm
IMAGE2=../sample/big_set/boat/img6.ppm
IMAGE3=../sample/big_set/boat/img2.ppm
IMAGE4=../sample/big_set/boat/img1.ppm
IMAGE5=../sample/big_set/boat/img5.ppm
IMAGE6=../sample/big_set/boat/img4.ppm
IMAGE7=../sample/big_set/bark/img3.ppm
IMAGE8=../sample/big_set/bark/img6.ppm
IMAGE9=../sample/big_set/bark/img2.ppm
IMAGE10=../sample/big_set/bark/img1.ppm
IMAGE11=../sample/big_set/bark/img5.ppm
IMAGE12=../sample/big_set/bark/img4.ppm
IMAGE13=../sample/big_set/bikes/img3.ppm
IMAGE14=../sample/big_set/bikes/img6.ppm
IMAGE15=../sample/big_set/bikes/img2.ppm
IMAGE16=../sample/big_set/bikes/img1.ppm
IMAGE17=../sample/big_set/bikes/img5.ppm
IMAGE18=../sample/big_set/bikes/img4.ppm
IMAGE19=../sample/big_set/wall/img3.ppm
IMAGE20=../sample/big_set/wall/img6.ppm
IMAGE21=../sample/big_set/wall/img2.ppm
IMAGE22=../sample/big_set/wall/img1.ppm
IMAGE23=../sample/big_set/wall/img5.ppm
IMAGE24=../sample/big_set/wall/img4.ppm
IMAGE25=../sample/big_set/leuven/img3.ppm
IMAGE26=../sample/big_set/leuven/img6.ppm
IMAGE27=../sample/big_set/leuven/img2.ppm
IMAGE28=../sample/big_set/leuven/img1.ppm
IMAGE29=../sample/big_set/leuven/img5.ppm
IMAGE30=../sample/big_set/leuven/img4.ppm
IMAGE31=../sample/big_set/ubc/img3.ppm
IMAGE32=../sample/big_set/ubc/img6.ppm
IMAGE33=../sample/big_set/ubc/img2.ppm
IMAGE34=../sample/big_set/ubc/img1.ppm
IMAGE35=../sample/big_set/ubc/img5.ppm
IMAGE36=../sample/big_set/ubc/img4.ppm
IMAGE37=../sample/big_set/graf/img3.ppm
IMAGE38=../sample/big_set/graf/img6.ppm
IMAGE39=../sample/big_set/graf/img2.ppm
IMAGE40=../sample/big_set/graf/img1.ppm
IMAGE41=../sample/big_set/graf/img5.ppm
IMAGE42=../sample/big_set/graf/img4.ppm
IMAGE43=../sample/big_set/trees/img3.ppm
IMAGE44=../sample/big_set/trees/img6.ppm
IMAGE45=../sample/big_set/trees/img2.ppm
IMAGE46=../sample/big_set/trees/img1.ppm
IMAGE47=../sample/big_set/trees/img5.ppm
IMAGE48=../sample/big_set/trees/img4.ppm

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
# PARAMS="--opencv-mode --octaves=8 --indirect-unfiltered --threshold=0.04 --edge-threshold=10.0 --initial-blur=0.5"
PARAMS="--popsift-mode --octaves=8 --indirect-unfiltered --threshold=0.04 --edge-threshold=10.0 --initial-blur=0.5 --downsampling=0"
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
# for i in $IMAGE1 $IMAGE2 $IMAGE3 $IMAGE4 $IMAGE5 $IMAGE6 ; do
for i in $IMAGE1 $IMAGE2 $IMAGE3 $IMAGE4 $IMAGE5 $IMAGE6 $IMAGE7 $IMAGE8 $IMAGE9 $IMAGE10 $IMAGE11 $IMAGE12 $IMAGE13 $IMAGE14 $IMAGE15 $IMAGE16 $IMAGE17 $IMAGE18 $IMAGE19 $IMAGE20 $IMAGE21 $IMAGE22 $IMAGE23 $IMAGE24 $IMAGE25 $IMAGE22 $IMAGE27 $IMAGE28 $IMAGE29 $IMAGE30 $IMAGE31 $IMAGE32 $IMAGE33 $IMAGE34 $IMAGE35 $IMAGE36 $IMAGE37 $IMAGE38 $IMAGE39 $IMAGE40 $IMAGE41 $IMAGE42 $IMAGE43 $IMAGE44 $IMAGE45 $IMAGE46 $IMAGE47 $IMAGE48 ; do
  if [ -f "$i" ] ; then
    Dirname=`dirname $i`
    Dirname=`basename ${Dirname}`
    outname=`basename --suffix=.ppm $i`
    # outname=`basename --suffix=.pgm $i`
    outname=$Dirname/$outname
    echo ./sift_v4 $PARAMS $LOG $i
    ./sift_v4 $PARAMS $LOG $i
    if [ ! -z "$LOG" ] ; then
      echo "mkdir -p outputs/${outname}"
      mkdir -p outputs/${outname}
      mv dir-* outputs/${outname}/
      mkdir -p outputs/popsift/$Dirname
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

