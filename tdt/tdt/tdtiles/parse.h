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

class ErrorComputer;
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

  float distance(const ErrorComputer& rs) const;
};

struct RtcOffset {
  double off[3];
};

struct RenderState;
struct Tile;
struct TileEntity;

struct ErrorComputer {
  ErrorComputer(const RenderState& rs);
  alignas(8) double screen_mvp[16];
  double eye[3];
  int16_t w, h;
  float u, v; // v = 1. / tan(.5 * fovy)
};

//struct TileRoot;
struct TileBase {
  BoundingVolume bndVol;
  float geoError;
  std::vector<TileBase*> children;
  Tile* root = nullptr;
  //TileRoot* root = nullptr;

  bool isRoot = false;
  std::string dir; // only the root stores the dir.
  std::vector<double> transform;

  static TileBase* fromFile(const std::string& fname);

  virtual void render(RenderState& rs) =0;

  float computeSSE(const ErrorComputer& ec) const;
  float computeScreenArea(const ErrorComputer& ec) const;

  enum class State {
    CLOSED, OPEN, FRONTIER, OPENING, CLOSING
  } state = State::CLOSED;

  virtual void open() =0;
  virtual void openChildren() =0;
  virtual void close() =0;
  virtual void closeChildren() =0;
};

/*
struct TileRoot : public TileBase {
  TileRoot(const std::string& dir, const std::string& fname);
  std::string dir;
  virtual void render(RenderState& rs);

  virtual void open();
  virtual void openChildren();
  virtual void close();
  virtual void closeChildren();
};
*/

struct TileModel;
struct Tileset : public TileBase {};
struct Tile : public TileBase {

  inline Tile() {}
  Tile(const std::string& dir, const std::string& fname);

  Refinement refine;

  std::string contentUri;
  char* contentPtr = 0;

  TileModel* model = 0;
  TileEntity* entity = 0;

  virtual void open();
  virtual void openChildren();
  virtual void close();
  virtual void closeChildren();

  void upload();
  void unload();

  virtual void render(RenderState& rs);


  // anyChildOpen: used depending on refinement.
  // loaded      : that means the cpu data is available in TileModel
  // uploaded    : that means the gpu data is available in TileEntity
  //
  // active(): if refinement is:
  //  add    : then any open tile is active.
  //  replace: a tile is active only when all children are closed.
  bool anyChildOpen = false, loaded = false, uploaded = false;
  inline bool active() const {
    if (refine == Refinement::ADD) return state == TileBase::State::OPEN;

    assert(refine == Refinement::REPLACE);
    // The loop is avoided by cachhine anyChildOpen every time it should change.
    return state == TileBase::State::OPEN and not anyChildOpen;
    //if (state != TileBase::State::OPEN) return false;
    //for (auto c : children) if (c->state == TileBase::State::OPEN) return false;
    //return true;
  }
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
