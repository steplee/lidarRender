#include "parse.h"
#include <cassert>
#include <fstream>

static bool readFile(Bytes& out, const std::string& fname) {
  std::ifstream ifs(fname);
  ifs.seekg(0, std::ios::end);
  out.reserve(ifs.tellg());
  ifs.seekg(0, std::ios::beg);
  out.assign((std::istreambuf_iterator<char>(ifs)),
                  std::istreambuf_iterator<char>());

  return true;
}
static bool readFile(std::string& out, const std::string& fname) {
  std::ifstream ifs(fname);
  ifs.seekg(0, std::ios::end);
  out.reserve(ifs.tellg());
  ifs.seekg(0, std::ios::beg);
  out.assign((std::istreambuf_iterator<char>(ifs)),
                  std::istreambuf_iterator<char>());
  return true;
}

void Tile::open() {
}
void Tile::close() {
}
float Tile::computeSSE(const RenderState& rs) {
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

TileEntity* TileEntity::parse_b3dm(const std::string& dir, const char* bytes, int len) {
  TileEntity* ent = nullptr;

  const char* cursor = bytes;
  int32_t magic, version, fileLen, ftjl, ftbl, btjl, btbl, gltfFormat;
  cursor = read_1(magic, cursor);
  cursor = read_1(version, cursor);
  cursor = read_1(len, cursor);
  cursor = read_1(ftjl, cursor);
  cursor = read_1(ftbl, cursor);
  cursor = read_1(btjl, cursor);
  cursor = read_1(btbl, cursor);
  cursor = read_1(gltfFormat, cursor);

  // Read Feature Table
  cursor = ent->ftable.parse(cursor, ftjl, ftbl);

  // Read Batch Table
  cursor = ent->btable.parse(cursor, ftjl, ftbl);

  // Read Gltf
  std::string glbString { cursor , (cursor+len) - cursor };
  auto model = GltfModel::fromGlbString(dir, glbString);
  auto thisAsGltfEnt = (GltfEntity*) ent;
  thisAsGltfEnt->upload(*model);
  delete model;

  return ent;
}



const char* FeatureTable::parse(const char* bytes, int json_len, int bin_len) {
}
const char* BatchTable::parse(const char* bytes, int json_len, int bin_len) {
}

TileBase* TileBase::fromFile(const std::string& fname) {
  TileBase* out = new TileBase();
  std::string json_str;
  assert(readFile(json_str, fname));
  auto jobj = json::parse(json_str);
  //out->
}
