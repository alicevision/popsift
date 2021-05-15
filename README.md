
# PopSift

[![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/3728/badge)](https://bestpractices.coreinfrastructure.org/projects/3728)  [![Codacy Badge](https://api.codacy.com/project/badge/Grade/8b0f7a68bc0d4df2ac89c6e732917caa)](https://app.codacy.com/manual/alicevision/popsift?utm_source=github.com&utm_medium=referral&utm_content=alicevision/popsift&utm_campaign=Badge_Grade_Settings)

PopSift is an open-source implementation of the SIFT algorithm in CUDA.
PopSift tries to stick as closely as possible to David Lowe's famous paper [1], while extracting features from an image in real-time at least on an NVidia GTX 980 Ti GPU.

Check out the [documentation](https://popsift.readthedocs.io/) for more info.

## HW requirements

PopSift compiles and works with NVidia cards of compute capability >= 3.0 (including the GT 650M), but the code is developed with the compute capability 5.2 card GTX 980 Ti in mind.

CUDA SDK 11 does no longer support compute capability 3.0. 3.5 is still supported with deprecation warning.

## Dependencies

PopSift depends on:

* Host compiler that supports C++14 for CUDA SDK >= 9.0 and C++11 for CUDA SDK 8

* CUDA >= 8.0

Optionally, for the provided applications:

* Boost >= 1.55 (required components {atomic, chrono, date-time, system, thread}-dev)

* DevIL (libdevil-dev) can be used to load a broader range of image formats, otherwise only pgm is supported.

## Build

In order to build the library you can run:

```
mkdir build && cd build
cmake ..
make
make install
```

Some build options are available:

* `PopSift_BUILD_EXAMPLES` (default: `ON`) enable building the applications that showcase the use of the library.

* `BUILD_SHARED_LIBS` (default: `ON`) controls the type of library to build (`ON` for shared libraries, `OFF` for static)

## Usage

The main artifact created is `libpopsift`.
If enabled, the test application `popsift-demo` is created as well.
Calling `popsift-demo` without parameters shows the options.

### Using PopSift as third party

To integrate PopSift into other software, link with `libpopsift`.
If your are using CMake for building your project you can easily add PopSift to your project.
Once you have built and installed PopSift in a directory (say, `<prefix>`), in your `CMakeLists.txt` file just add the dependency

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

The caller must create a `popart::Config` struct (documented in `src/sift/sift_conf.h`) to control the behaviour of the PopSift, and instantiate an object of class `PopSift` (found in `src/sift/popsift.h`).

After this, images can be enqueued for SIFT extraction using (`enqueue()`).
A valid input is a single plane of grayscale values located in host memory.
They can passed as a pointer to unsigned char, with a value range from 0 to 255, or as a pointer to float, with a value range from 0.0f to 1.0f.

Only host memory limits the number of images that can be enqueued.
The `enqueue` function returns a pointer to a `SiftJob` immediately and performs the feature extraction asynchronously.
The memory of the image passed to enqueue remains the caller's responsibility. Calling `SiftJob::get` on the returned job blocks until features are extracted, and returns them.

Features offer iterators that iterate over objects of type `Feature`.
Both classes are documented in `sift_extremum.h`.
Each feature represents a feature point in the coordinate system of the input image, providing X and Y coordinates and scale (sigma), as well as several alternative descriptors for the feature point (according to Lowe, 15% of the feature points should be expected to have 2 or more descriptors).

In an alternate, deprecated, blocking API, `init()` must be called to pass image width and height to PopSift, followed by a call to `executed()` that takes image data and returns the extracted features. `execute()` is synchronous and blocking.

As far as we know, no implementation that is faster than PopSift at the time of PopSift's release comes under a license that allows commercial use and sticks close to the original paper at the same time as well.
PopSift can be configured at runtime to use constants that affect it behaviours.
In particular, users can choose to generate results very similar to VLFeat or results that are closer (but not as close) to the SIFT implementation of the OpenCV extras.
We acknowledge that there is at least one SIFT implementation that is vastly faster, but it makes considerable sacrifices in terms of accuracy and compatibility.

## Continuous integration:
- [![Build Status](https://travis-ci.org/alicevision/popsift.svg?branch=master)](https://travis-ci.org/alicevision/popsift) master branch.
- [![Build Status](https://travis-ci.org/alicevision/popsift.svg?branch=develop)](https://travis-ci.org/alicevision/popsift) develop branch.
- [![Build status](https://ci.appveyor.com/api/projects/status/rsm5269hs288c2ji/branch/develop?svg=true)](https://ci.appveyor.com/project/AliceVision/popsift/branch/develop)
  develop branch.

## License

PopSift is licensed under [MPL v2 license](COPYING.md).
SIFT was patented in the United States from 1999-03-08 to 2020-03-28. See the [patent link](https://patents.google.com/patent/US6711293B1/en) for more information.
PopSift license only concerns the PopSift source code and does not release users of this code from any requirements that may arise from patents.


## Cite Us

If you use PopSift for your publication, please cite us as:
```bibtex
@inproceedings{Griwodz2018Popsift,
	 author = {Griwodz, Carsten and Calvet, Lilian and Halvorsen, P{\aa}l},
	 title = {Popsift: A Faithful SIFT Implementation for Real-time Applications},
	 booktitle = {Proceedings of the 9th {ACM} Multimedia Systems Conference},
	 series = {MMSys '18},
	 year = {2018},
	 isbn = {978-1-4503-5192-8},
	 location = {Amsterdam, Netherlands},
	 pages = {415--420},
	 numpages = {6},
	 doi = {10.1145/3204949.3208136},
	 acmid = {3208136},
	 publisher = {ACM},
	 address = {New York, NY, USA},
}
```


## Acknowledgements

PopSift was developed within the project [POPART](https://alicevision.org/popart), which has been funded by the [European Commission in the Horizon 2020](https://cordis.europa.eu/project/id/644874) framework.

___

[1]: Lowe, D. G. (2004). Distinctive Image Features from Scale-Invariant Keypoints. International Journal of Computer Vision, 60(2), 91–110. doi:10.1023/B:VISI.0000029664.99615.94
