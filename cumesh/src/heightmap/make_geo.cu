#include "make_geo.h"

#include <unordered_map>
#include <iostream>

static int log2(int x) {
  int i = 0;
  while (x>>=1) i++;
  return i;
}

const int8_t PLUS_Z = 1;
const int8_t MNUS_Z = 2;
const int8_t PLUS_X = 4;
const int8_t MNUS_X = 8;
const int8_t PLUS_Y = 16;
const int8_t MNUS_Y = 32;

// Originally the first zeros were x/y/z, but now I just add those in the loop
constexpr int xyzss[6*4*3] = {
  // Z+
  0     , 0   , 0+1 ,
  0+1   , 0   , 0+1 ,
  0+1   , 0+1 , 0+1 ,
  0     , 0+1 , 0+1 ,
  // Z-
  0     , 0   , 0+0 ,
  0+1   , 0   , 0+0 ,
  0+1   , 0+1 , 0+0 ,
  0     , 0+1 , 0+0 ,
  // X+
  0+1   , 0   , 0   ,
  0+1   , 0+1 , 0   ,
  0+1   , 0+1 , 0+1 ,
  0+1   , 0   , 0+1 ,
  // X-
  0     , 0   , 0   ,
  0     , 0+1 , 0   ,
  0     , 0+1 , 0+1 ,
  0     , 0   , 0+1 ,
  // Y+
  0     , 0+1 , 0   ,
  0+1   , 0+1 , 0   ,
  0+1   , 0+1 , 0+1 ,
  0     , 0+1 , 0+1 ,
  // Y+
  0     , 0   , 0   ,
  0+1   , 0   , 0   ,
  0+1   , 0   , 0+1 ,
  0     , 0   , 0+1
};

void makeGeo(
    std::vector<float>& verts,
    std::vector<int>& tris,
    //GpuBuffer<int4>& outVerts,
    GpuBuffer<int3>& outTris,
    GpuBuffer<int4>& outQuads,
    GpuBuffer<int>& outVert2tris,
    const int4* nodesGpu,
    int N,
    int G) {
  float Gf = G;

  int4* nodes = (int4*) malloc(sizeof(int4)*N);
  cudaMemcpy(nodes, nodesGpu, sizeof(int4)*N, cudaMemcpyDeviceToHost);


  // Create one quad or two tris per node.
  bool needTris = true;
  bool needQuads = false;

  std::unordered_map<int64_t, int32_t> seen;
  //std::vector<float> verts;
  std::vector<int32_t> quads;
  //std::vector<int32_t> tris;
  std::vector<int32_t> vert2tris(N*12, -1);

  // Most likely need much more, but a good start
  if (needTris) tris.reserve(N * 6);
  if (needQuads) quads.reserve(N * 4);

  for (int i=0; i<N; i++) {
    int4 xyzu = nodes[i];
    int x = xyzu.x, y = xyzu.y, z = xyzu.z, u = xyzu.w;

    // For the X/Y facing faces, the correct order must be used when GL_CULL_FACE is enabled
    int flip[6] = {0,1, 0,1, 1,0};

    for (int k=0; k<6; k++) {
      if ((1<<k) & u) {
        //if (u & PLUS_Z) {
        //int* xyzs = xyzss + 0*12;
        const int* xyzs = xyzss + k*12;
        int32_t inds[4];

        for (int j=0; j<4; j++) {
          int xx = x+xyzs[j*3+0], yy = y+xyzs[j*3+1], zz = z+xyzs[j*3+2];
          int32_t ind_j;
          //int64_t signature = (((int64_t)zz)<<16ull) | (((int64_t)yy)<<16ull) | (((int64_t)xx)<<16ull);
          int64_t signature  = static_cast<int64_t>(zz); signature <<= 16;
          signature |= static_cast<int64_t>(yy); signature <<= 16;
          signature |= static_cast<int64_t>(xx);
          if (seen.find(signature) == seen.end()) {
            float xf = static_cast<float>(xx) / Gf,
                  yf = static_cast<float>(yy) / Gf,
                  zf = static_cast<float>(zz) / Gf;
            seen[signature] = verts.size()/3;
            ind_j = verts.size()/3;
            verts.push_back(xf); verts.push_back(yf); verts.push_back(zf);

            if (verts.size() >= vert2tris.size() / 12 / 3) {
              std::cout << " - resizing vert2tris " << vert2tris.size() << " -> " << vert2tris.size() * 2 * 12 * 3 << " (verts " << verts.size() << ")\n";
              //vert2tris.resize(vert2tris.size() * 2 * 12 * 3, -1);
              vert2tris.resize(vert2tris.size() * 2, -1);
            }
          } else
            ind_j = seen[signature];

          inds[j] = ind_j;
        }

        if (needQuads) {
          quads.push_back(inds[0]); quads.push_back(inds[1]);
          quads.push_back(inds[2]); quads.push_back(inds[3]);
        }
        if (needTris) {
          if (flip[k] == 0) {
            tris.push_back(inds[0]); tris.push_back(inds[1]); tris.push_back(inds[2]);
            tris.push_back(inds[2]); tris.push_back(inds[3]); tris.push_back(inds[0]);
            vert2tris[inds[0]*12+k*2+0] = tris.size()/3-2; vert2tris[inds[1]*12+k*2+0] = tris.size()/3-2; vert2tris[inds[2]*12+k*2+0] = tris.size()/3-2;
            vert2tris[inds[2]*12+k*2+1] = tris.size()/3-1; vert2tris[inds[3]*12+k*2+1] = tris.size()/3-1; vert2tris[inds[0]*12+k*2+1] = tris.size()/3-1;
          } else {
            tris.push_back(inds[1]); tris.push_back(inds[0]); tris.push_back(inds[2]);
            tris.push_back(inds[3]); tris.push_back(inds[2]); tris.push_back(inds[0]);
            vert2tris[inds[1]*12+k*2+0] = tris.size()/3-2; vert2tris[inds[0]*12+k*2+0] = tris.size()/3-2; vert2tris[inds[2]*12+k*2+0] = tris.size()/3-2;
            vert2tris[inds[3]*12+k*2+1] = tris.size()/3-1; vert2tris[inds[2]*12+k*2+1] = tris.size()/3-1; vert2tris[inds[0]*12+k*2+1] = tris.size()/3-1;
          }
        }
      }
      }

    }


    /*
  torch::Tensor outVerts, outTris, outQuads;

  outVerts = torch::empty({(int)verts.size()/3,3}, TensorOptions().dtype(kFloat));
  memcpy(outVerts.data_ptr<float>(), verts.data(), sizeof(float)*verts.size());

  if (needTris) {
    outTris = torch::empty({(int)tris.size()}, TensorOptions().dtype(kInt32));
    memcpy(outTris.data_ptr<int32_t>(), tris.data(), sizeof(int32_t)*tris.size());
  }
  if (needQuads) {
    outQuads = torch::empty({(int)quads.size()}, TensorOptions().dtype(kInt32));
    memcpy(outQuads.data_ptr<int32_t>(), quads.data(), sizeof(int32_t)*quads.size());
  }

  torch::Tensor vert2tris_t = torch::from_blob((void*)vert2tris.data(), {(int)verts.size()*3,12}, TensorOptions().dtype(kInt32));

  std::vector<torch::Tensor> out;
  out.push_back(outVerts.cpu());
  out.push_back(outTris.cpu());
  out.push_back(outQuads);
  return out;
  */
  //outVerts.allocate(verts.size() / 3);
  outTris.allocate(tris.size() / 3);
  outQuads.allocate(quads.size() / 4);
  outVert2tris.allocate(vert2tris.size());
  cudaMemcpy(outTris.buf, tris.data(), sizeof(int32_t) * tris.size() / 3, cudaMemcpyHostToDevice);
  cudaMemcpy(outQuads.buf, quads.data(), sizeof(int32_t) * quads.size() / 4, cudaMemcpyHostToDevice);
  cudaMemcpy(outVert2tris.buf, vert2tris.data(), sizeof(int32_t) * vert2tris.size(), cudaMemcpyHostToDevice);
}
