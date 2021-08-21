#include "parse.h"
#include <fstream>
#include <stack>
#include <iostream>
#include "../io.hpp"

#include "tdt/render_context.h"
#include "tdt/extra_entities.h"

using namespace nlohmann;

void Tile::upload() {
  assert(model != nullptr);
  assert(entity == nullptr);
  entity = new TileEntity();
  entity->upload(*model->model);
  delete model; model = 0;
}
void Tile::unload() {
}
void Tile::open() {
  //if (not loaded) {
    //loader->enqueue_load(this);
    //state = State::OPENING;
  //}
  Bytes bytes;
  std::cout << " - reading file " << root->dir + contentUri << "\n";
  readFile(bytes, root->dir + contentUri);
  model = TileModel::parse_b3dm(".", (const char*) bytes.data(), bytes.size());
  upload();
}
void Tile::close() {
  if (loaded) {
    unload();
  }
  if (entity) delete entity;
  state = State::CLOSED;
}
float Tile::computeSSE(RenderState& rs) {
}
void Tile::render(RenderState& rs) {
  if (entity) entity->renderAllNodes(rs);
  for (auto& c : children) c->render(rs);

  if (bndVol.type == BoundingVolume::Type::BBOX) {
  } else if (bndVol.type == BoundingVolume::Type::SPHERE) {
    if (rs.sphereEntity != nullptr) {
      //std::cout << " - rendering sphere from bnd vol " << bndVol.data.sphere[3] << ".\n";
      rs.sphereEntity->setPositionAndRadius(bndVol.data.sphere, bndVol.data.sphere[3]);
      rs.sphereEntity->render(rs);
    }
  }
}
void TileRoot::render(RenderState& rs) {
  for (auto& c : children) c->render(rs);
}

BoundingVolume BoundingVolume::parse(const json& j) {
  BoundingVolume v;
  if (j.contains("region")) {
    v.type = BoundingVolume::Type::REGION;
    for (int i=0; i<6; i++) v.data.wsenhh[i] = j["region"][i];
  } else if (j.contains("box")) {
    v.type = BoundingVolume::Type::BBOX;
    for (int i=0; i<12; i++) v.data.box[i] = j["box"][i];
  } else if (j.contains("sphere")) {
    v.type = BoundingVolume::Type::SPHERE;
    for (int i=0; i<4; i++) v.data.sphere[i] = j["sphere"][i];
  }
  return v;
}




static const char* read_n(char* out, const char* in, int n) {
  memcpy(out, in, n);
  return in + n;
}
template <class T>
static const char* read_1(T& out, const char* in) {
  memcpy(&out, in, sizeof(T));
  return in + sizeof(T);
}

TileModel* TileModel::parse_b3dm(const std::string& dir, const char* bytes, int len) {
  TileModel* model = nullptr;


  const char* cursor = bytes;
  int32_t magic, version, fileLen, ftjl, ftbl, btjl, btbl, gltfFormat;
  cursor = read_1(magic, cursor);
  cursor = read_1(version, cursor);
  cursor = read_1(fileLen, cursor);
  cursor = read_1(ftjl, cursor);
  cursor = read_1(ftbl, cursor);
  cursor = read_1(btjl, cursor);
  cursor = read_1(btbl, cursor);
  //cursor = read_1(gltfFormat, cursor);
  printf(" - parsed header (%d): (ftjl %d) (ftbl %d) (btjl %d) (btbl %d) (v %d) (gltf %d)\n",
      cursor - bytes, ftjl,ftbl,btjl,btbl,version,gltfFormat);

  // Read Feature Table
  cursor = model->ftable.parse(cursor, ftjl, ftbl);
  //std::cout << " - parsed ftable, cursor > " << cursor - bytes << "\n";

  // Read Batch Table
  cursor = model->btable.parse(cursor, btjl, btbl);
  //std::cout << " - parsed btable, cursor > " << cursor - bytes << "\n";

  // Read Gltf
  std::string glbString { cursor , (size_t) (fileLen - (cursor - bytes)) };
  //for (int i=0; i<10; i++) std::cout << " - c " << cursor[i] << "\n";
  model = new TileModel();
  model->model = GltfModel::fromGlbString(dir, glbString);
  //auto thisAsGltfEnt = (GltfEntity*) ent;
  //thisAsGltfEnt->upload(*model);
  //delete model;
  return model;
}



const char* FeatureTable::parse(const char* bytes, int json_len, int bin_len) {
  return bytes + json_len + bin_len;
}

const char* BatchTable::parse(const char* bytes, int json_len, int bin_len) {
  return bytes + json_len + bin_len;
}

TileRoot::TileRoot(const std::string& dir_, const std::string& fname) {
//TileBase* TileBase::fromFile(const std::string& fname) {
  dir = dir_;
  if (dir[dir.length()-1] != '/') dir += '/';
  std::string json_txt;
  readFile(json_txt, fname);
  auto jobj0 = json::parse(json_txt);
  auto jobj1 = jobj0["root"];

  //TileBase* root = new TileBase();
  TileRoot* root = this;

  std::vector<TileBase> children;
  std::stack<json> st1;
  std::stack<TileBase*> st2;
  st1.push(jobj1);
  st2.push(root);
  while (not st1.empty()) {
    auto jobj = st1.top();
    auto parent = st2.top();
    st1.pop(); st2.pop();
    for (auto& j : jobj["children"]) {
      auto cur = new Tile();
      cur->root = root;
      if (j.contains("boundingVolume")) {
        cur->bndVol = BoundingVolume::parse(j["boundingVolume"]);
      }
      if (j.contains("content")) {
        cur->contentUri = j["content"]["uri"];
      }
      std::cout << " - uri " << cur->contentUri << "\n";
      cur->open();
      parent->children.push_back(cur);
      st1.push(j);
      st2.push(cur);
    }
  }
}
