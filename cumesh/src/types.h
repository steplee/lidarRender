#pragma once

#include <ostream>

struct LasPoint {
  float x,y,z;
  inline friend std::ostream& operator <<(std::ostream& o, const LasPoint& p) {
    o << p.x << " " << p.y << " " << p.z;
    return o;
  }
  inline LasPoint& operator*=(const float& m) {
    x *= m; y *= m; z *= m;
    return *this;
  }
};
