#pragma once
#include "../gltf_entity.h"


// Three threads
// 1) Render Thread
//      - Must load and unload (finalize load)
// 2) Thread that walks tree and checks SSE
// 3) Thread-pool that loads data.

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
    double sphere[4];
    float box[12];
    double wsenhh[6];
  } data;

  static BoundingVolume parse(const nlohmann::json& j);
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
  std::vector<TileBase*> children;

  static TileBase* fromFile(const std::string& fname);

  virtual void render(RenderState& rs) =0;
};
struct TileRoot : public TileBase {
  TileRoot(const std::string& dir, const std::string& fname);
  std::string dir;
  virtual void render(RenderState& rs);
};

struct TileModel;
struct Tileset : public TileBase {};
struct Tile : public TileBase {
  Refinement refine;
  TileRoot* root = nullptr;

  std::string contentUri;
  char* contentPtr = 0;

  TileModel* model = 0;
  TileEntity* entity = 0;

  void open();
  void close();
  float computeSSE(RenderState& rs);

  void upload();
  void unload();

  virtual void render(RenderState& rs);


  enum class State {
    CLOSED, OPEN, OPENING, CLOSING
  } state = State::CLOSED;
  bool loaded = false, uploaded = false;
};

struct TileModel {
  inline ~TileModel () { if (model) delete model; }
  GltfModel* model = nullptr;
  BatchTable btable;
  FeatureTable ftable;

  static TileModel* parse_b3dm(const std::string& dir, const char* bytes, int len);
  static TileModel* parse_i3dm(const std::string& dir, const char* bytes, int len);
  static TileModel* parse_pnts(const std::string& dir, const char* bytes, int len);
};
struct TileEntity : public GltfEntity {

};
