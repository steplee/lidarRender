#pragma once

#include "types.h"

struct HeightMapRasterizer {
  int resolution = 4096;
  void run(const std::vector<LasPoint>& pts);

  std::vector<float> inlyingPoints;
};

