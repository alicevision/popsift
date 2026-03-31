To compare Popsift results with vlfeat results:

Compile vlfeat from source. Fix the following bug in the C code of VLFeat:
in vl/pgm.c
change the if-test
'if(! (max_value >= 65536)) {'
to
'if( max_value >= 65536 ) {'
After this, the VLfeat command line version works again.

Run vlfeat on your image like this;:
./bin/arm64/sift boat-img1.pgm

Run popsift-demo on your image like this:
./Darwin-arm64/popsift-demo --root-sift --write-as-uchar --write-as-ori --norm-multi 9 --threshold 0 -i /Users/griff/boat-img1.pgm
--write-as-ori is required to have the same header information for every descriptor.
--write-as-uchar --norm-multi 9 is required to create descriptor consisting of bytes like VLFeat.
--threshold 0 is required to get (nearly) the same extrema points as VLFeat.
--root-sift: It is possible that L2 norm instead of root sift would have worked better. I didn't try that yet.

Note that these were compiles for M3 Mac. That required a few additional modifications in the vlfeat Makefiles (introduce arm64 in several places).

