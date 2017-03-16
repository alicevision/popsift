
PopSift
=======

PopSift is an implementation of the SIFT algorithm in CUDA.
PopSift tries to stick as closely as possible to David Lowe's famous paper (Lowe, D. G. (2004). Distinctive Image Features from Scale-Invariant Keypoints. International Journal of Computer Vision, 60(2), 91–110. doi:10.1023/B:VISI.0000029664.99615.94), while extracting features from an image in real-time at least on an NVidia GTX 980 Ti GPU.


Build
-----

PopSift has been developed and tested on Linux machines, mostly a variant of Ubuntu, but compiles on MacOSX as well. It comes as a CMake project and requires at least CUDA 7.0. It is known to compile and work with NVidia cards of compute capability 3.0 (including the GT 650M), but the code is developed with the compute capability 5.2 card GTX 980 Ti in mind.

If you want to avoid building the application you can run cmake with the option `-DPopSift_BUILD_EXAMPLES:BOOL=OFF`.

In order to build the library you can run:

```
mkdir build && cd build
cmake ..
make
make install
```

Continuous integration: 
- [![Build Status](https://travis-ci.org/alicevision/popsift.svg?branch=master)](https://travis-ci.org/alicevision/popsift) master branch.
- [![Build Status](https://travis-ci.org/alicevision/popsift.svg?branch=develop)](https://travis-ci.org/alicevision/popsift) develop branch.



Usage
-----

Two artifacts are made: `libpopsift` and the test application `popsift-demo`. Calling popsift-demo without parameters shows the options.

### Using PopSift as third party

To integrate PopSift into other software, link with `libpopsift`.  If your are using CMake for building your project you can easily add PopSift to your project. Once you have built and installed PopSift in a directory (say, `<prefix>`), in your `CMakeLists.txt` file just add the dependency

```cmake
# Find the package from the PopSiftConfig.cmake 
# in <prefix>/lib/cmake/PopSift/. Under the namespace PopSift::
# it exposes the target popsift that allows you to compile
# and link with the library
find_package(PopSift CONFIG REQUIRED)
...
# suppose you want to try it out in a executable
add_executable(poptest yourfile.cpp)
# add link to the library
target_link_libraries(poptest PUBLIC PopSift::popsift)
```

Then, in order to build just pass the location of `PopSiftConfig.cmake` from the cmake command line:

```bash
cmake .. -DPopSift_DIR=<prefix>/lib/cmake/PopSift/
```



### Calling the API

The caller must create a `popart::Config` struct (documented in `src/sift/sift_conf.h`) to control the behaviour of the PopSift, and instantiate an object of class `PopSift` (found in `src/sift/popsift.h`).  After creating an instance, it must be configured to the constant width and height of the input image (`init()`).  After that, it can be fed a one image at a time (`execute()`). The only valid input format is a single plane of grayscale unsigned characters. The `execute()` function returns a pointer to an object of type `Features` that must be deleted by the caller after use. Features offer iterators that iterate over objects of type `Feature`. Both classes are documented in `sift_extremum.h`. Each feature represents a feature point in the coordinate system of the input image, providing X and Y coordinates and scale (sigma), as well as several alternative descriptors for the feature point (according to Lowe, 15% of the feature points should be expected to have 2 or more descriptors).

As far as we know, no implementation that is faster than PopSift at the time of PopSift's release comes under a license that allows commercial use and sticks close to the original paper at the same time as well. PopSift can be configured at runtime to use constants that affect it behaviours. In particular, users can choose to generate results very similar to VLFeat or results that are closer (but not as close) to the SIFT implementation of the OpenCV extras. We acknowledge that there is at least one SIFT implementation that is vastly faster, but it makes considerable sacifices in terms of accuracy and compatibility.


License
-------

PopSift is licensed under [MPL v2 license](LICENSE.md).
However, SIFT is patented in the US and perhaps other countries, and this license does not release users of this code from any requirements that may arise from such patents.


Authors
-------

It was developed within the project [POPART](http://www.popartproject.eu), which has been funded by the European Commission in the Horizon 2020 framework.
