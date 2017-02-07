#include "dataio.h"
#include <iostream>
#include "SiftBench.h"
#include "KDTree.h"


// TODO: cmdline parameters!
static constexpr unsigned LEAF_SIZE = 50;
static constexpr unsigned TREE_COUNT = 10;
static constexpr unsigned QUERY_DESCRIPTOR_LIMIT = 2000;

int main()
{
    popsift::kdtree::VerifyL2DistanceAVX();

    std::vector<popsift::kdtree::U8Descriptor> dbv, qv;
    std::vector<std::pair<unsigned, unsigned>> gt;
    ReadTexMex(dbv, qv, gt);

    popsift::kdtree::SiftBench bench(gt, dbv, qv);
    bench.BuildKDTree(LEAF_SIZE, TREE_COUNT);
    bench.Bench(QUERY_DESCRIPTOR_LIMIT);

    return 0;
}
