#pragma once
#include "../gltf_entity.h"

struct b3dm {
};
struct i3dm {
};
struct BatchTable {
  const char* parse(const char* bytes, int json_len, int bin_len);
};
struct FeatureTable {
  const char* parse(const char* bytes, int json_len, int bin_len);
};

enum class Refinement { ADD, REPLACE };
struct BoundingVolume {
  enum class Type {
    SPHERE, BBOX, REGION
  } type;
  union {
    double ctr[6];
    float box[12];
    double wsenhh[6];
  } data;
};

struct RtcOffset {
  double off[3];
};

struct RenderState;
struct Tile;
struct TileEntity;

struct TileBase {
  BoundingVolume bndVol;
  float geoError;
  std::vector<TileBase> children;

  static TileBase* fromFile(const std::string& fname);
};

struct Tileset : public TileBase {};
struct Tile : public TileBase {
  Refinement refine;

  std::string contentUri;
  char* contentPtr = 0;

  TileEntity* entity = 0;

  void open();
  void close();
  float computeSSE(const RenderState& rs);

  enum class State {
    CLOSED, OPEN, OPENING, CLOSING
  } state;
  bool loaded = false, uploaded = false;
};

struct TileEntity : public GltfEntity {
  BatchTable btable;
  FeatureTable ftable;

  static TileEntity* parse_b3dm(const std::string& dir, const char* bytes, int len);
  static TileEntity* parse_i3dm(const std::string& dir, const char* bytes, int len);
  static TileEntity* parse_pnts(const std::string& dir, const char* bytes, int len);
};
