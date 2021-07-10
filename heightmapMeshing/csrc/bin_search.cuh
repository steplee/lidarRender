#pragma once

// FASTER VERSION
__device__ inline int64_t binSearch(int x, int y, int z, const int64_t* inds, int N) {
  int lastLo = 0, lastHi = N;
  int lo=0, hi=N;
  int mid = (lo + hi) / 2;

  // X
  while (lo < hi and inds[mid] != x) {
    if (inds[mid] > x) { hi = mid; mid = (lo+hi) / 2; }
    else if (inds[mid] < x) { lo = mid+1; mid = (lo+hi) / 2; }
  }
  //lo=hi=mid;
  if (inds[lo] != x) lo = mid;
  if (inds[hi] != x) hi = mid;
  int step = 1024;
  while (lo>lastLo+1 and inds[lo-1] == x) {
    if (lo-step > lastLo and inds[lo-step] == x) lo -= step;
    else if (step>1) step /= 2;
    //printf(" - stuck %d %d %d | X1 %d %d step %d\n",x,y,z, lo,hi, step);
  }
  step = 1024;
  while (hi<lastHi and inds[hi] == x) {
    if (hi+step-1 < lastHi and inds[hi+step-1] == x) hi += step;
    else if (step>1) step /= 2;
    //printf(" - stuck %d %d %d | X2 %d %d step %d\n",x,y,z, lo,hi, step);
  }
  //while (hi<lastHi   and inds[N+hi] == y) hi++;
  if (inds[lo] != x) return -1;
  lastLo = lo, lastHi = hi;
  mid = (lo+hi) / 2;

  // Y
  while (lo < hi and inds[N+mid] != y) {
    if (inds[N+mid] > y) { hi = mid; mid = (lo+hi) / 2; }
    else if (inds[N+mid] < y) { lo = mid+1; mid = (lo+hi) / 2; }
  }
  //lo=hi=mid;
  if (inds[N+lo] != y) lo = mid;
  if (inds[N+hi] != y) hi = mid;
  step = 1024;
  while (lo>lastLo+1 and inds[N+lo-1] == y) {
    if (lo-step > lastLo and inds[N+lo-step] == y) lo -= step;
    else if (step>1) step /= 2;
    //printf(" - stuck %d %d %d | Y1 %d %d step %d\n",x,y,z, lo,hi, step);
  }
  step = 1024;
  while (hi<lastHi   and inds[N+hi] == y) {
    if (hi+step-1 < lastHi and inds[N+hi+step-1] == y) hi += step;
    else if (step>1) step /= 2;
    //printf(" - stuck %d %d %d | Y2 %d %d step %d\n",x,y,z, lo,hi, step);
  }
  //while (hi<lastHi   and inds[N+hi] == y) hi++;
  if (inds[N+lo] != y) return -1;
  lastLo = lo, lastHi = hi;
  mid = (lo+hi) / 2;

  // Z
  while (lo < hi and inds[N+N+mid] != z) {
    if (inds[N+N+mid] > z) { hi = mid; mid = (lo+hi) / 2; }
    else if (inds[N+N+mid] < z) { lo = mid+1; mid = (lo+hi) / 2; }
  }
  //lo=hi=mid;
  if (inds[N+N+lo] != z) lo = mid;
  if (inds[N+N+hi] != z) hi = mid;
  while (lo>lastLo+1 and inds[N+N+lo-1] == z) lo--;
  while (hi<lastHi   and inds[N+N+hi] == z) hi++;
  if (inds[N+N+lo] != z) return -1;

  // We know that after coalesce() indices are unique, so either 1 or 0 correct indices.
  return hi==lo+1 ? lo : -1;
}
