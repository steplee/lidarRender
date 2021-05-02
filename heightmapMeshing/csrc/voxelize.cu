#include <torch/extension.h>

using namespace torch;
using namespace torch::indexing;

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

static __global__ void fillLvl(
    int64_t* inds, int G, int lvl,
    float* surface) {
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int x = blockIdx.x * blockDim.x + threadIdx.x;

  if (y<0 or y>=G or x<0 or x>=G) return;

  float factor = 1. / ((float)G);
  float myZ = ((float)lvl) * factor;
  float mySurfaceZ = surface[y*G+x];

  int8_t dirMask = 0;

  float myLastZ = ((float)(lvl-1)) * factor;
  // nope
  if (myLastZ > mySurfaceZ) return;
  if (myZ > mySurfaceZ) dirMask = PLUS_Z;

  for (int dd=0; dd<4; dd++) {
    //int dx = (dd<2) * (dd%2?1:-1);
    //int dy = (dd>2) * (dd%2?1:-1);
    int dx = dd == 0 ? 1 : dd == 1 ? -1 : 0;
    int dy = dd == 2 ? 1 : dd == 3 ? -1 : 0;

    int yy = y+dy;
    int xx = x+dx;
    if (yy>=0 && yy < G && xx>=0 && xx < G) {
      float neighborSurfaceZ = surface[yy*G+xx];
      //if (myZ > neighborSurfaceZ) dirMask |= ((i!=0)*4*(i+3)) | ((j!=0)*2*(j+3));
      if (myZ > neighborSurfaceZ)
        if (dx ==  1) dirMask |= PLUS_X;
        else if (dx == -1) dirMask |= MNUS_X;
        else if (dy ==  1) dirMask |= PLUS_Y;
        else dirMask |= MNUS_Y;
    }
  }

  if (dirMask) {
    int i = y*G+x;
    inds[i*4+0] = x;
    inds[i*4+1] = y;
    inds[i*4+2] = lvl;
    inds[i*4+3] = dirMask;
  }

}

torch::Tensor forward(
    torch::Tensor& surface,
    float meterScale
    ) {
  int G = surface.size(0);
  int L = log2(G);

  float max_z = surface.max().item().to<float>();
  int max_z_lvl = G * max_z;
  float min_z = surface.min().item().to<float>();
  int min_z_lvl = G * min_z;
  std::cout << " - max z " << max_z << " " << max_z_lvl << std::endl;
  std::cout << " - min z " << min_z << " " << min_z_lvl << std::endl;

  Tensor inds = torch::zeros({0,0});

  for (int l=0; l<max_z_lvl; l++) {
    Tensor newInds = torch::full({G*G,4}, -1, TensorOptions().dtype(kLong).device(kCUDA));
    dim3 blk((G+15)/16, (G+15)/16);
    dim3 thr(16,16);
    fillLvl<<<blk,thr>>>(
        newInds.data_ptr<int64_t>(), G,l,
        surface.data_ptr<float>());

    //std::cout << " - inds.numel() " << inds.numel() << " inds sizes " << inds.sizes() << std::endl;
    //std::cout << " - newInds sizes " << newInds.sizes() << std::endl;
    //const auto& mask = (newInds.index({Slice(),0})>-1).view({-1,1});
    const auto& mask = (newInds.index({Slice(),0})>-1).view({-1,1}).repeat({1,4});
    //std::cout << " - newInds.selected sizes " << newInds.masked_select(mask).sizes() << std::endl;
    if (l == 0 or inds.numel() == 0)
      inds = newInds.masked_select(mask).view({-1,4});
    else inds = torch::cat({inds, newInds.masked_select(mask).view({-1,4})}, 0);
  }

  // Tensor out = in;

  return inds;
}

std::vector<torch::Tensor> makeGeo(
    const torch::Tensor& nodes_,
    int G) {
  torch::Tensor nodes = nodes_.cpu();
  int N = nodes.size(0);
  int64_t* nodePtr = nodes.data_ptr<int64_t>();

  float Gf = G;

  // Create one quad or two tris per node.
  bool needTris = true;
  bool needQuads = false;

  std::unordered_map<int64_t, int32_t> seen;
  std::vector<float> verts;
  std::vector<int32_t> quads;
  std::vector<int32_t> tris;

  // Most likely need much more, but a good start
  if (needTris) tris.reserve(N * 6);
  if (needQuads) quads.reserve(N * 4);

  for (int i=0; i<N; i++) {
    int64_t x = nodePtr[i*4+0];
    int64_t y = nodePtr[i*4+1];
    int64_t z = nodePtr[i*4+2];
    int64_t u = nodePtr[i*4+3];

    int xyzss[6*4*3] = {
      // Z+
      x     , y   , z+1 ,
      x+1   , y   , z+1 ,
      x+1   , y+1 , z+1 ,
      x     , y+1 , z+1 ,
      // Z-
      x     , y   , z+0 ,
      x+1   , y   , z+0 ,
      x+1   , y+1 , z+0 ,
      x     , y+1 , z+0 ,
      // X+
      x+1   , y   , z   ,
      x+1   , y+1 , z   ,
      x+1   , y+1 , z+1 ,
      x+1   , y   , z+1 ,
      // X-
      x     , y   , z   ,
      x     , y+1 , z   ,
      x     , y+1 , z+1 ,
      x     , y   , z+1 ,
      // Y+
      x     , y+1 , z   ,
      x+1   , y+1 , z   ,
      x+1   , y+1 , z+1 ,
      x     , y+1 , z+1 ,
      // Y+
      x     , y   , z   ,
      x+1   , y   , z   ,
      x+1   , y   , z+1 ,
      x     , y   , z+1
    };

    // For the X/Y facing faces, the correct order must be used when GL_CULL_FACE is enabled
    int flip[6] = {0,1, 0,1, 1,0};

    for (int k=0; k<6; k++) {
      if ((1<<k) & u) {
    //if (u & PLUS_Z) {
      //int* xyzs = xyzss + 0*12;
      int* xyzs = xyzss + k*12;
      int32_t inds[4];

      for (int j=0; j<4; j++) {
        int xx = xyzs[j*3+0], yy = xyzs[j*3+1], zz = xyzs[j*3+2];
        //int64_t signature = (((int64_t)zz)<<16ull) | (((int64_t)yy)<<16ull) | (((int64_t)xx)<<16ull);
        int64_t signature  = static_cast<int64_t>(zz); signature <<= 16;
                signature |= static_cast<int64_t>(yy); signature <<= 16;
                signature |= static_cast<int64_t>(xx);
        if (seen.find(signature) == seen.end()) {
          float xf = static_cast<float>(xx) / Gf,
                yf = static_cast<float>(yy) / Gf,
                zf = static_cast<float>(zz) / Gf;
          seen[signature] = verts.size()/3;
          inds[j] = verts.size()/3;
          verts.push_back(xf); verts.push_back(yf); verts.push_back(zf);
        } else
          inds[j] = seen[signature];
      }

      if (needQuads) {
        quads.push_back(inds[0]); quads.push_back(inds[1]);
        quads.push_back(inds[2]); quads.push_back(inds[3]);
      }
      if (needTris) {
        if (flip[k] == 0) {
          tris.push_back(inds[0]); tris.push_back(inds[1]); tris.push_back(inds[2]);
          tris.push_back(inds[2]); tris.push_back(inds[3]); tris.push_back(inds[0]);
        } else {
          tris.push_back(inds[1]); tris.push_back(inds[0]); tris.push_back(inds[2]);
          tris.push_back(inds[3]); tris.push_back(inds[2]); tris.push_back(inds[0]);
        }
      }
      }
      }

    }

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


  std::vector<torch::Tensor> out;
  out.push_back(outVerts);
  out.push_back(outTris);
  out.push_back(outQuads);
  return out;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Do it.");
  m.def("makeGeo", &makeGeo, "Do it.");
}

