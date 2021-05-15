Requirements
============

Hardware
~~~~~~~~

PopSift is a GPU implementation that requires an NVIDIA GPU card with a CUDA compute capability >= 3.0 (including, e.g. the GT 650M).
The code is originally developed with the compute capability 5.2 card GTX 980 Ti in mind.

You can check your `NVIDIA GPU card CC support here <https://github.com/tpruvot/ccminer/wiki/Compatibility>`_ or on the `NVIDIA dev page <https://developer.nvidia.com/cuda-gpus>`_.
If you do not have a NVIDIA card you will still able to compile and use the CPU version.

Here are the minimum hardware requirements for PopSift:

+--------------------------------------------------------------------------+
| Minimum requirements                                                     |
+===================+======================================================+
| Operating systems | Windows x64, Linux, macOS                            |
+-------------------+------------------------------------------------------+
| CPU               | Recent Intel or AMD cpus                             |
+-------------------+------------------------------------------------------+
| RAM Memory        | 8 GB                                                 |
+-------------------+------------------------------------------------------+
| Hard Drive        | No particular requirements                           |
+-------------------+------------------------------------------------------+
| GPU               | NVIDIA CUDA-enabled GPU (compute capability >= 3.5)  |
+-------------------+------------------------------------------------------+



Software
~~~~~~~~

The core library depends only on Cuda >= 7.0

The library includes a few sample applications that show how to use the library.
They require

* Boost >= 1.55 (required components atomic, chrono, date-time, system, thread)

* [optionally] DevIL (libdevil-dev) can be used to load a broader range of image formats, otherwise only pgm is supported.



------------


vcpkg
=====

`vcpkg <https://github.com/microsoft/vcpkg>`_ is a cross-platform (Windows, Linux and MacOS), open-source package manager created by Microsoft.

Starting from v0.9, PopSift package can be installed on each platform via vcpkg.
To install the library:

.. code:: shell

  vcpkg install popsift --triplet <arch>

where :code:`<arch>` is the architecture to build for (e.g. :code:`x64-windows`, :code:`x64-linux-dynamic` etc.)

If you want to install the demo applications that come with the library you can add the option :code:`apps`:

.. code:: shell

  vcpkg install popsift[apps] --triplet <arch>

------------

Building the library
====================

Building tools
~~~~~~~~~~~~~~

Required tools:

* CMake >= 3.14 to build the code
* Git
* C/C++ compiler supporting the C++11 standard (gcc >= 4.6 or visual studio or clang)
* CUDA >= 7.0



Dependencies
~~~~~~~~~~~~

vcpkg
+++++

vcpkg can be used to install all the dependencies on all the supported platforms.
This is particularly useful on Windows.
To install the dependencies:

.. code:: shell

  vcpkg install cuda devil boost-system boost-program-options boost-thread boost-filesystem

You can add the flag :code:`--triplet` to specify the architecture and the version you want to build.
For example:

* :code:`--triplet x64-windows` will build the dynamic version for Windows 64 bit

* :code:`--triplet x64-windows-static` will build the static version for Windows 64 bit

* :code:`--triplet x64-linux-dynamic` will build the dynamic version for Linux 64 bit

and so on.
More information can be found `here <https://vcpkg.readthedocs.io/en/latest/examples/overlay-triplets-linux-dynamic>`_

Linux
+++++

On Linux you can install from the package manager:

For Ubuntu/Debian package system:

.. code:: shell

    sudo apt-get install g++ git-all libboost-all-dev libdevil-dev


For CentOS package system:

.. code:: shell

    sudo yum install gcc-c++ git boost-devel devil


MacOS
+++++

On MacOs using `Homebrew <https://brew.sh/>`_ install the following packages:

.. code:: shell

    brew install git boost devil


Getting the sources
~~~~~~~~~~~~~~~~~~~~

.. code:: shell

   git clone https://github.com/alicevision/PopSift.git


CMake configuration
~~~~~~~~~~~~~~~~~~~

From PopSift root folder you can run cmake:

.. code:: shell

    mkdir build && cd build
    cmake ..
    make -j `nproc`

On Windows add :code:`-G "Visual Studio 16 2019" -A x64` to generate the Visual Studio solution according to your VS version (`see CMake documentation <https://cmake.org/cmake/help/latest/manual/cmake-generators.7.html#ide-build-tool-generators>`_).

If you are using the dependencies built with VCPKG you need to pass :code:`-DCMAKE_TOOLCHAIN_FILE=path/to/vcpkg/scripts/buildsystems/vcpkg.cmake` at cmake step to let it know where to find the dependencies.


CMake options
+++++++++++++

CMake configuration can be controlled by changing the values of the following variables (here with their default value)


* :code:`BUILD_SHARED_LIBS:BOOL=ON` to enable/disable the building shared libraries

* :code:`PopSift_BUILD_EXAMPLES:BOOL=ON` to enable/disable the building of applications

* :code:`PopSift_BUILD_DOC:BOOL=OFF` to enable/disable building this documentation and the Doxygen one.

For example, if you do not want to build the applications, you have to pass :code:`-DPopSift_BUILD_EXAMPLES:BOOL=OFF` and so on.


------------


PopSift as third party
====================

When you install PopSift a file :code:`PopSiftConfig.cmake` is installed in :code:`<install_prefix>/lib/cmake/PopSift/` that allows you to import the library in your CMake project.
In your :code:`CMakeLists.txt` file you can add the dependency in this way:

.. code-block::
  :linenos:

  # Find the package from the PopSiftConfig.cmake
  # in <prefix>/lib/cmake/PopSift/. Under the namespace PopSift::
  # it exposes the target PopSift that allows you to compile
  # and link with the library
  find_package(PopSift CONFIG REQUIRED)
  ...
  # suppose you want to try it out in a executable
  add_executable(popsiftTest yourfile.cpp)
  # add link to the library
  target_link_libraries(popsiftTest PUBLIC PopSift::PopSift)

Then, in order to build just pass the location of :code:`PopSiftConfig.cmake` from the cmake command line:

.. code:: shell

    cmake .. -DPopSift_DIR=<install_prefix>/lib/cmake/PopSift/


------------



Docker image
============

A docker image can be built using the Ubuntu based :code:`Dockerfile`, which is based on nvidia/cuda image (https://hub.docker.com/r/nvidia/cuda/ )


Building the dependency image
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We provide a :code:`Dockerfile_deps` containing a cuda image with all the necessary PopSift dependencies installed.

A parameter :code:`CUDA_TAG` can be passed when building the image to select the cuda version.
Similarly, :code:`OS_TAG` can be passed to select the Ubuntu version.
By default, :code:`CUDA_TAG=10.2` and :code:`OS_TAG=18.04`

For example to create the dependency image based on ubuntu 18.04 with cuda 8.0 for development, use

.. code:: shell

    docker build --build-arg CUDA_TAG=8.0 --tag alicevision/popsift-deps:cuda8.0-ubuntu18.04 -f Dockerfile_deps .

The complete list of available tags can be found on the nvidia [dockerhub page](https://hub.docker.com/r/nvidia/cuda/)


Building the PopSift image
~~~~~~~~~~~~~~~~~~~~~~~~

Once you built the dependency image, you can build the popsift image in the same manner using :code:`Dockerfile`:

.. code:: shell

    docker build --tag alicevision/popsift:cuda8.0-ubuntu18.04 .


Running the PopSift image
~~~~~~~~~~~~~~~~~~~~~~~

In order to run the image nvidia docker is needed: see the `installation instruction <https://github.com/nvidia/nvidia-docker/wiki/Installation-(version-2.0)>`_.
Once installed, the docker can be run, e.g., in interactive mode with

.. code:: shell

    docker run -it --runtime=nvidia alicevision/popsift:cuda8.0-ubuntu18.04


Official images on DockeHub
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Check the docker hub `PopSift repository <https://hub.docker.com/repository/docker/alicevision/popsift>`_ for the available images.