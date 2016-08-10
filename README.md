PopSift
=======

PopSift is an implementation of the SIFT algorithm in CUDA.
PopSift tries to stick as closely as possible to David Lowe's famous paper (Lowe, D. G. (2004). Distinctive Image Features from Scale-Invariant Keypoints. International Journal of Computer Vision, 60(2), 91–110. doi:10.1023/B:VISI.0000029664.99615.94), while extracting features from an image in real-time at least on an NVidia GTX 980 Ti GPU.

Build
-----

PopSift has been developed and tested on Linux machines, mostly a variant of Ubuntu. It comes as a CMake project and requires at least CUDA 7.0 and a device with compute capability 3.5 or later. It requires also boost to compile the test application.

```
mkdir build && cd build
cmake ..
make
make install
```

Usage
-----

Two artifacts are made: libpopsift and the test application popsift-demo. Calling popsift-demo without parameters shows an option. To integrate PopSift into other software, link with libpopsift.
The caller must create a popart::Config struct (documented in src/sift/sift_conf.h) to control the behaviour of the PopSift, and instantiate an object of class PopSift (found in src/sift/popsift.h). After creating an instance, it must be configured to the constant width and height of the input image (init()). After that, it can be fed a one image at a time (execute()). The only valid input format is a single plane of grayscale unsigned characters. The caller can pass two pointers to one vector of vectors each. They should be empty on entry. The first one will on return hold the extrema (several copies for several orientations) sorted by level and the second the descriptors belonging to each of them.

As far as we know, no implementation that is faster than PopSift at the time of PopSift's release comes under a license that allows commercial use and sticks close to the original paper at the same time as well. PopSift can be configured at runtime to use constants that affect it behaviours. In particular, users can choose to generate results very similar to VLFeat or results that are closer (but not as close) to the SIFT implementation of the OpenCV extras. We acknowledge that there is at least one SIFT implementation that is vastly faster, but it makes considerable sacifices in terms of accuracy and compatibility.

License
-------

PopSift is licensed under [MPL v2 license](LICENSE.md).
However, SIFT is patented in the US and perhaps other countries, and this license does not release users of this code from any requirements that may arise from such patents.

Authors
-------

It was developed within the project [POPART](http://www.popartproject.eu), which has been funded by the European Commission in the Horizon 2020 framework.
