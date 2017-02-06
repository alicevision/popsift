#include "dataio.h"
#include <iostream>
#include "SiftBench.h"

void TexMexBench();

int main()
{
    auto dbv = ReadCDCoversVectors("C:/LOCAL/sift_cdcovers/sift_dump_1M_3319.txt");
    auto qv = ReadCDCoversVectors("C:/LOCAL/sift_cdcovers/sift_dump_ukbench_normal_10K_3319.txt");
    auto gt = ReadCDCoversGT("C:/LOCAL/sift_cdcovers/GT_3319.txt");

    using namespace popsift::kdtree;
    SiftBench bench(gt, dbv, qv);
    bench.BuildKDTree(50, 10);
    bench.Bench(1000);
    return 0;
}
