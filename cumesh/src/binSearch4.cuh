#pragma once


// See binSearch.cuh, this is the same, except it add one dim for depth,
// which is exactly the same as a spatial dim.

__device__ inline int64_t binSearch4(int d, int x, int y, int z, const int32_t* inds, int N) {
  int lastLo = 0, lastHi = N;
  int lo=0, hi=N;
  int mid = (lo + hi) / 2;

  // D
  while (lo < hi and inds[mid] != d) {
    if (inds[mid] > d) { hi = mid; mid = (lo+hi) / 2; }
    else if (inds[mid] < d) { lo = mid+1; mid = (lo+hi) / 2; }
  }
  //lo=hi=mid;
  if (inds[lo] != d) lo = mid;
  if (inds[hi] != d) hi = mid;
  int step = 1024;
  while (lo>lastLo+1 and inds[lo-1] == d) {
    if (lo-step > lastLo and inds[lo-step] == d) lo -= step;
    else if (step>1) step /= 2;
    //printf(" - stuck %d %d %d | X1 %d %d step %d\n",x,y,z, lo,hi, step);
  }
  step = 1024;
  while (hi<lastHi and inds[hi] == d) {
    if (hi+step-1 < lastHi and inds[hi+step-1] == d) hi += step;
    else if (step>1) step /= 2;
    //printf(" - stuck %d %d %d | X2 %d %d step %d\n",x,y,z, lo,hi, step);
  }
  //while (hi<lastHi   and inds[N+hi] == y) hi++;
  if (inds[lo] != d) return -1;
  lastLo = lo, lastHi = hi;
  mid = (lo+hi) / 2;

  // X
  while (lo < hi and inds[N+mid] != x) {
    if (inds[N+mid] > x) { hi = mid; mid = (lo+hi) / 2; }
    else if (inds[N+mid] < x) { lo = mid+1; mid = (lo+hi) / 2; }
  }
  //lo=hi=mid;
  if (inds[N+lo] != x) lo = mid;
  if (inds[N+hi] != x) hi = mid;
  step = 1024;
  while (lo>lastLo+1 and inds[N+lo-1] == x) {
    if (lo-step > lastLo and inds[N+lo-step] == x) lo -= step;
    else if (step>1) step /= 2;
    //printf(" - stuck %d %d %d | x1 %d %d step %d\n",x,x,z, lo,hi, step);
  }
  step = 1024;
  while (hi<lastHi   and inds[N+hi] == x) {
    if (hi+step-1 < lastHi and inds[N+hi+step-1] == x) hi += step;
    else if (step>1) step /= 2;
    //printf(" - stuck %d %d %d | x2 %d %d step %d\n",x,x,z, lo,hi, step);
  }
  //while (hi<lastHi   and inds[N+hi] == x) hi++;
  if (inds[N+lo] != x) return -1;
  lastLo = lo, lastHi = hi;
  mid = (lo+hi) / 2;

  // Y
  while (lo < hi and inds[N+N+mid] != y) {
    if (inds[N+N+mid] > y) { hi = mid; mid = (lo+hi) / 2; }
    else if (inds[N+N+mid] < y) { lo = mid+1; mid = (lo+hi) / 2; }
  }
  //lo=hi=mid;
  if (inds[N+N+lo] != y) lo = mid;
  if (inds[N+N+hi] != y) hi = mid;
  step = 1024;
  while (lo>lastLo+1 and inds[N+N+lo-1] == y) {
    if (lo-step > lastLo and inds[N+N+lo-step] == y) lo -= step;
    else if (step>1) step /= 2;
    //printf(" - stuck %d %d %d | Y1 %d %d step %d\n",x,y,z, lo,hi, step);
  }
  step = 1024;
  while (hi<lastHi   and inds[N+N+hi] == y) {
    if (hi+step-1 < lastHi and inds[N+N+hi+step-1] == y) hi += step;
    else if (step>1) step /= 2;
    //printf(" - stuck %d %d %d | Y2 %d %d step %d\n",x,y,z, lo,hi, step);
  }
  //while (hi<lastHi   and inds[N+N+hi] == y) hi++;
  if (inds[N+N+lo] != y) return -1;
  lastLo = lo, lastHi = hi;
  mid = (lo+hi) / 2;

  // Z
  while (lo < hi and inds[N+N+N+mid] != z) {
    if (inds[N+N+N+mid] > z) { hi = mid; mid = (lo+hi) / 2; }
    else if (inds[N+N+N+mid] < z) { lo = mid+1; mid = (lo+hi) / 2; }
  }
  //lo=hi=mid;
  if (inds[N+N+N+lo] != z) lo = mid;
  if (inds[N+N+N+hi] != z) hi = mid;
  while (lo>lastLo+1 and inds[N+N+N+lo-1] == z) lo--;
  while (hi<lastHi   and inds[N+N+N+hi] == z) hi++;
  if (inds[N+N+N+lo] != z) return -1;

  // We know that after coalesce() indices are unique, so either 1 or 0 correct indices.
  return hi==lo+1 ? lo : -1;
}

//__device__ inline int64_t decode_d(const int64_t& i) { return i >> 59ul; }
//__device__ inline int64_t decode_x(const int64_t& i) { return i >> 38ul; }
//__device__ inline int64_t decode_y(const int64_t& i) { return i >> 19ul; }
//__device__ inline int64_t decode_z(const int64_t& i) { return i &  0b1111111111111111111; }
__device__ inline int64_t decode_d(const int64_t& i) { return (i >> 59ul) & 0b1111111111111111111; }
__device__ inline int64_t decode_x(const int64_t& i) { return (i >> 38ul) & 0b1111111111111111111; }
__device__ inline int64_t decode_y(const int64_t& i) { return (i >> 19ul) & 0b1111111111111111111; }
__device__ inline int64_t decode_z(const int64_t& i) { return  i          & 0b1111111111111111111; }
__device__ inline int     binSearch4_encoded(int d, int x, int y, int z, const int64_t* inds, int N) {
  int lastLo = 0, lastHi = N;
  int lo=0, hi=N;
  int mid = (lo + hi) / 2;

  // D
  auto md = decode_d(inds[mid]);
  while (lo < hi and md != d) {
    if (md > d) { hi = mid; mid = (lo+hi) / 2; }
    else if (md < d) { lo = mid+1; mid = (lo+hi) / 2; }
  }
  //lo=hi=mid;
  if (decode_d(inds[lo]) != d) lo = mid;
  if (decode_d(inds[hi]) != d) hi = mid;
  int step = 1024;
  while (lo>lastLo+1 and decode_d(inds[lo-1]) == d) {
    if (lo-step > lastLo and decode_d(inds[lo-step]) == d) lo -= step;
    else if (step>1) step /= 2;
    //printf(" - stuck %d %d %d | X1 %d %d step %d\n",x,y,z, lo,hi, step);
  }
  step = 1024;
  while (hi<lastHi and decode_d(inds[hi]) == d) {
    if (hi+step-1 < lastHi and decode_d(inds[hi+step-1]) == d) hi += step;
    else if (step>1) step /= 2;
    //printf(" - stuck %d %d %d | X2 %d %d step %d\n",x,y,z, lo,hi, step);
  }
  //while (hi<lastHi   and inds[N+hi] == y) hi++;
  if (decode_d(inds[lo]) != d) return -1;
  lastLo = lo, lastHi = hi;
  mid = (lo+hi) / 2;

  // X
  while (lo < hi and decode_x(inds[mid]) != x) {
    if (decode_x(inds[mid]) > x) { hi = mid; mid = (lo+hi) / 2; }
    else if (decode_x(inds[mid]) < x) { lo = mid+1; mid = (lo+hi) / 2; }
  }
  //lo=hi=mid;
  if (decode_x(inds[lo]) != x) lo = mid;
  if (decode_x(inds[hi]) != x) hi = mid;
  step = 1024;
  while (lo>lastLo+1 and decode_x(inds[lo-1]) == x) {
    if (lo-step > lastLo and decode_x(inds[lo-step]) == x) lo -= step;
    else if (step>1) step /= 2;
    //printf(" - stuck %d %d %d | x1 %d %d step %d\n",x,x,z, lo,hi, step);
  }
  step = 1024;
  while (hi<lastHi   and decode_x(inds[hi]) == x) {
    if (hi+step-1 < lastHi and decode_x(inds[hi+step-1]) == x) hi += step;
    else if (step>1) step /= 2;
    //printf(" - stuck %d %d %d | x2 %d %d step %d\n",x,x,z, lo,hi, step);
  }
  //while (hi<lastHi   and inds[N+hi] == x) hi++;
  if (decode_x(inds[lo]) != x) return -1;
  lastLo = lo, lastHi = hi;
  mid = (lo+hi) / 2;

  // Y
  while (lo < hi and decode_y(inds[mid]) != y) {
    if (decode_y(inds[mid]) > y) { hi = mid; mid = (lo+hi) / 2; }
    else if (decode_y(inds[mid]) < y) { lo = mid+1; mid = (lo+hi) / 2; }
  }
  //lo=hi=mid;
  if (decode_y(inds[lo]) != y) lo = mid;
  if (decode_y(inds[hi]) != y) hi = mid;
  step = 1024;
  while (lo>lastLo+1 and decode_y(inds[lo-1]) == y) {
    if (lo-step > lastLo and decode_y(inds[lo-step]) == y) lo -= step;
    else if (step>1) step /= 2;
    //printf(" - stuck %d %d %d | Y1 %d %d step %d\n",x,y,z, lo,hi, step);
  }
  step = 1024;
  while (hi<lastHi   and decode_y(inds[hi]) == y) {
    if (hi+step-1 < lastHi and decode_y(inds[hi+step-1]) == y) hi += step;
    else if (step>1) step /= 2;
    //printf(" - stuck %d %d %d | Y2 %d %d step %d\n",x,y,z, lo,hi, step);
  }
  //while (hi<lastHi   and inds[N+N+hi] == y) hi++;
  if (decode_y(inds[lo]) != y) return -1;
  lastLo = lo, lastHi = hi;
  mid = (lo+hi) / 2;

  // Z
  while (lo < hi and decode_z(inds[mid]) != z) {
    if (decode_z(inds[mid]) > z) { hi = mid; mid = (lo+hi) / 2; }
    else if (decode_z(inds[mid]) < z) { lo = mid+1; mid = (lo+hi) / 2; }
  }
  //lo=hi=mid;
  if (decode_z(inds[lo]) != z) lo = mid;
  if (decode_z(inds[hi]) != z) hi = mid;
  while (lo>lastLo+1 and decode_z(inds[lo-1]) == z) lo--;
  while (hi<lastHi   and decode_z(inds[hi]) == z) hi++;
  if (decode_z(inds[lo]) != z) return -1;

  // We know that after coalesce() indices are unique, so either 1 or 0 correct indices.
  return hi==lo+1 ? lo : -1;
}
