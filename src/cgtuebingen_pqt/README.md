# Efficient Large-scale Approximate Nearest Neighbor Search on the GPU

This repositoy contains the implementation of **Product Quantization Trees** (PQT) for large scale nearest neighbor search.

    @inproceedings{PQT,
        author = "Patrick Wieschollek and Oliver Wang and Alexander Sorkine-Hornung and Hendrik P.A. Lensch",
        title = "Efficient Large-scale Approximate Nearest Neighbor Search on the GPU",
        booktitle = "IEEE Conference on Computer Vision and Pattern Recognition (CVPR)",
        pages = "",
        month = "June",
        year = "2016",
        url = "goo.gl/4Zl5xB"
    }

This experimental implementation is the first running on the GPU outperforming previous CPU approaches. 


## Usage

### Download

Just by

    git clone --recursive https://github.com/cgtuebingen/Product-Quantization-Tree.git

### Requirements
To build this project you need

* [cmake](http://www.cmake.org)
* A C++11 capable compiler
* The CUDA toolkit and the CUDA samples.
* [GFlags](https://github.com/gflags/gflags) and [GoogleTesting](https://github.com/google/googletest) (as git submodules)

We use Toolkit v7.5, gcc 4.8.4 on Ubuntu 14.04 for Nvidia Titan X.

### Preprocessing

To handle datasets and efficiently read them from (ram-)disk we converted the official datasets from [SIFT1M, SIFT1B](http://corpus-texmex.irisa.fr/) in our own format:


Each `*.umem`, `*.imem`, `*.fmem` has the following layout (header and content)

    uint number_of_vectors
    uint dimen_of_vector
    ... header are 20 bytes, next data start at byte 20:
    T consecutive array of data, each entry is a T

Examples

    * .umem: `uint  uint 0 0 ... 0 0 uint8_t uint8_t uint8_t uint8_t uint8_t uint8_t ...  uint8_t`
    * .imem: `uint  uint 0 0 ... 0 0 int int int int int int ...  int`
    * .fmem: `uint  uint 0 0 ... 0 0 float float float float float float ...  float`

We provide script to convert these datasets

    ./convert_fvecs --fvecs src/path/to/db.fvecs --umem dst/path/to/db.umem
    ./convert_bvecs --bvecs src/path/to/db.bvecs --umem dst/path/to/db.umem
    ./convert_ivecs --ivecs src/path/to/db.ivecs --imem dst/path/to/db.imem


### Query (offline phase)

Building the index structure (Product-Quantization-Tree) is done by `tool_createdb`. This creates the index structure, prepares the database and dump all intermediate values into binary files.

Possible Flags (see `./tool_createdb -h`):

    - device    "selected cuda device"
    - c1        "number of clusters in first level"
    - c2        "number of refinements in second level"
    - p         "parts per vector"
    - dim       "expected dimension for each vector"
    - lineparts "vectorparts for reranking informations"
    - chunksize "number of vectors per chunk"
    - hashsize  "maximal number of bins"
    - basename  "prefix for generated data"
    - dataset   "patch to vector dataset"

### Query (online phase)

The query process is done batchwise using `tool_query`. The accompanying example return the best and second best found vector. For possible Flags see `./tool_query -h`.
