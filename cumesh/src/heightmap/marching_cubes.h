#pragma once


#include <vector>

#include "octree.h"


void run_mc(SortedOctree<float>& tree, int lvl, float isovalue, std::vector<float>& outTris);
