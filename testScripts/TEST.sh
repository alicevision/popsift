# IMAGE=../../popsift-samples/sample/big_set/boat/img2.ppm
IMAGE=../../popsift-samples/sample/big_set/boat/img3.ppm
# IMAGE=./test-17x17.pgm

# LOG=--log
LOG=
# GAUSS_MODE="--gauss-mode=vlfeat"
GAUSS_MODE="--gauss-mode=relative"
# GAUSS_MODE="--gauss-mode=fixed15"
SCALING=
# SCALING=--direct-scaling

# for mode in loop ; do
for mode in loop grid igrid notile ; do
# for mode in igrid notile ; do
  echo "MODE: $mode"
  echo "./popsift-demo $LOG $GAUSS_MODE $SCALING --popsift-mode --desc-mode=$mode --octaves=8 --threshold=0.04 --edge-threshold=10.0 --initial-blur=0.5 --write-as-uchar --norm-multi=9 -i $IMAGE"
  ./popsift-demo $LOG $GAUSS_MODE $SCALING --popsift-mode --desc-mode=$mode --octaves=8 --threshold=0.04 --edge-threshold=10.0 --initial-blur=0.5 --write-as-uchar --norm-multi=9 -i $IMAGE
  # ./popsift-demo $LOG --popsift-mode --desc-mode=$mode --octaves=8 --threshold=0.04 --edge-threshold=10.0 --initial-blur=0.5 --write-as-uchar --norm-multi=9 -i $IMAGE
  sort -n output-features.txt > UML
  echo 128        >  output-features-$mode.txt
  cat UML | wc -l >> output-features-$mode.txt
  cat UML         >> output-features-$mode.txt
  rm output-features.txt
done

echo -n "loop vs grid:   "
~/GIT/github/popsift-samples/playground/build/compare-descfiles \
	-q output-features-loop.txt output-features-grid.txt

#echo -n "loop vs iloop:  "
#~/GIT/github/popsift-samples/playground/build/compare-descfiles \
#	-q output-features-loop.txt output-features-iloop.txt

echo -n "loop vs igrid:  "
~/GIT/github/popsift-samples/playground/build/compare-descfiles \
	-q output-features-loop.txt output-features-igrid.txt

echo -n "loop vs notile:  "
~/GIT/github/popsift-samples/playground/build/compare-descfiles \
	-q output-features-loop.txt output-features-notile.txt

#echo -n "grid vs iloop:  "
#~/GIT/github/popsift-samples/playground/build/compare-descfiles \
#	-q output-features-grid.txt output-features-iloop.txt

echo -n "grid vs igrid:  "
~/GIT/github/popsift-samples/playground/build/compare-descfiles \
	-q output-features-grid.txt output-features-igrid.txt

echo -n "grid vs notile:  "
~/GIT/github/popsift-samples/playground/build/compare-descfiles \
	-q output-features-grid.txt output-features-notile.txt

#echo -n "iloop vs igrid: "
#~/GIT/github/popsift-samples/playground/build/compare-descfiles \
#	-q output-features-iloop.txt output-features-igrid.txt

#echo -n "iloop vs notile: "
#~/GIT/github/popsift-samples/playground/build/compare-descfiles \
#	-q output-features-iloop.txt output-features-notile.txt

echo -n "igrid vs notile: "
~/GIT/github/popsift-samples/playground/build/compare-descfiles \
	-q output-features-igrid.txt output-features-notile.txt

