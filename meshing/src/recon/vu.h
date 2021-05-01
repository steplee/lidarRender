#include <Eigen/StdVector>
#include <stdint.h>

struct CompressedOctree;

void labelCells(CompressedOctree& so, const std::vector<uint8_t>& vis, int lvl=0);
