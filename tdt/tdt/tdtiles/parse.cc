#include "parse.h"
#include <fstream>
#include <stack>
#include <iostream>
#include "../io.hpp"
#include "../math.h"


using namespace nlohmann;


BoundingVolume BoundingVolume::parse(const json& j, const double xform[16]) {
  BoundingVolume v;
  if (j.contains("region")) {
    v.type = BoundingVolume::Type::REGION;
    for (int i=0; i<6; i++) v.data.wsenhh[i] = j["region"][i];
  } else if (j.contains("box")) {
    v.type = BoundingVolume::Type::BBOX;
    //for (int i=0; i<12; i++) v.data.box[i] = j["box"][i];
    for (int i=0; i<4; i++)  {
      double p[3] = {j["box"][i*3+0], j["box"][i*3+1], j["box"][i*3+2]};
      double pp[3];
      matvec43(pp, xform, p);
      for (int j=0; j<3; j++) v.data.box[i*3+j] = pp[j];
      std::cout << " - BOX B4 " << p[0] << " " << p[1] << " " << p[2] << "\n";
      std::cout << " - BOX AF " << pp[0] << " " << pp[1] << " " << pp[2] << "\n";
    }
  } else if (j.contains("sphere")) {
    v.type = BoundingVolume::Type::SPHERE;
    //for (int i=0; i<4; i++) v.data.sphere[i] = j["sphere"][i];
    double p[3] = {j["sphere"][0], j["sphere"][1], j["sphere"][2]};
    double pp[3];
    matvec43(pp, xform, p);
    for (int i=0; i<3; i++) v.data.sphere[i] = pp[i];
    v.data.sphere[3] = j["sphere"][3];
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
  std::stack<std::pair<json,int>> st1;
  std::stack<std::pair<Tile*,TileBase*>> st2;
  st1.push({jobj1,1});
  st2.push({0,root});

  double xformStack[13*16];
  for (int i=0; i<16; i++) xformStack[i] = i % 5 == 0;

  double baseGeoError = jobj0["geometricError"];
  root->geoError = baseGeoError;

  while (not st1.empty()) {
    auto jobj = st1.top().first;
    auto depth = st1.top().second;
    Tile* parent; TileBase* cur;
    parent = st2.top().first;
    cur = st2.top().second;
    //auto cur = st2.top();
    st1.pop(); st2.pop();

    if (jobj.contains("transform")) {
      jobj.at("transform").get_to(cur->transform);
      for (int i=0; i<4; i++) for (int j=0; j<i; j++) std::swap(cur->transform[i*4+j], cur->transform[j*4+i]);
      std::cout << " - Parsed Tile Transform:\n";
      for (int i=0; i<cur->transform.size(); i++) std::cout << cur->transform[i] << ((i%4==3) ? "\n" : " ");
      matmul44(xformStack+depth*16, xformStack+(depth-1)*16, cur->transform.data());
    } else {
      memcpy(xformStack+depth*16, xformStack+(depth-1)*16, sizeof(double)*16);
    }

    if (jobj.contains("boundingVolume")) {
      cur->bndVol = BoundingVolume::parse(jobj["boundingVolume"], xformStack+depth*16);
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
      st1.push({j,depth+1});
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
