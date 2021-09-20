#include "parse.h"
#include <fstream>
#include <stack>
#include <iostream>
#include "../io.hpp"


using namespace nlohmann;


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

#if 1
//TileRoot::TileRoot(const std::string& dir_, const std::string& fname) {
Tile::Tile(const std::string& dir_, const std::string& fname) {
//TileBase* TileBase::fromFile(const std::string& fname) {
  dir = dir_;
  if (dir[dir.length()-1] != '/') dir += '/';
  std::string json_txt;
  readFile(json_txt, fname);
  auto jobj0 = json::parse(json_txt);
  auto jobj1 = jobj0["root"];

  //TileBase* root = new TileBase();
  Tile* root = this;

  std::vector<TileBase> children;
  std::stack<json> st1;
  std::stack<std::pair<Tile*,TileBase*>> st2;
  st1.push(jobj1);
  st2.push({0,root});

  double baseGeoError = jobj0["geometricError"];
  root->geoError = baseGeoError;

  while (not st1.empty()) {
    auto jobj = st1.top();
    Tile* parent; TileBase* cur;
    parent = st2.top().first;
    cur = st2.top().second;
    //auto cur = st2.top();
    st1.pop(); st2.pop();

    if (jobj.contains("transform")) {
      //cur->transform = jobj.get<std::vector<double>>("transform");
      jobj.at("transform").get_to(cur->transform);
    }

    if (jobj.contains("boundingVolume")) {
      cur->bndVol = BoundingVolume::parse(jobj["boundingVolume"]);
    }
    if (jobj.contains("geometricError")) {
      cur->geoError = jobj["geometricError"];
    }

    Tile* cur_ = dynamic_cast<Tile*>(cur);
    if (cur_) {
      cur_->root = root;
      if (jobj.contains("content")) {
        cur_->contentUri = jobj["content"]["uri"];
      }
      std::cout << " - uri " << cur_->contentUri << "\n";

      if (jobj.contains("refine")) {
        cur_->refine = jobj["refine"] == "REPLACE" ? Refinement::REPLACE : Refinement::ADD;
        std::cout << " - new tile has refine " << (int)cur_->refine << ", from json " << jobj["refine"] << "\n";
      } else if (parent) {
        cur_->refine = parent->refine;
        std::cout << " - new tile has refine " << (int)cur_->refine << ", from parent.\n";
      }
    }

    for (auto& j : jobj["children"]) {
      auto nxt = new Tile();
      cur->children.push_back(nxt);
      st1.push(j);
      st2.push({cur_,nxt});

      if (not j.contains("geometricError")) nxt->geoError = cur->geoError;
    }
  }

  /*
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
      //cur->open();
      parent->children.push_back(cur);
      st1.push(j);
      st2.push(cur);
    }
  }
  */
}
#endif
