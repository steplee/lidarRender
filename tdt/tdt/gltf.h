#pragma once
#include <tdt/thirdparty/json.hpp>

#include <Eigen/Core>

#include <vector>
#include <string_view>

using nlohmann::json;

using Bytes = std::vector<uint8_t>;

enum class DataType {
  SCALAR,
  VEC2, VEC3, VEC4,
  MAT2, MAT3, MAT4
};
enum class AttributeType {
  POSITION, NORMAL,
  TANGENT, TEXCOORD_0, TEXCOORD_1,
  COLOR_0, JOINTS_0, WEIGHTS_0, BATCHID
};
enum class AlphaMode {
  DEFAULT, OPAQUE, MASK, BLEND
};

struct GltfNode {
  int mesh=-1;
  //double translation[3], scale[3], quat[4];
  double xform[16];
  std::vector<int> children;
};
struct GltfBuffer {
  Bytes data;
  int byteLength;
};
struct GltfBufferView {
  int buffer;
  int byteLength;
  int byteOffset;
  int byteStride;
  int target;
};
struct GltfAccessor {
  int bufferView;
  int byteOffset;
  int componentType;
  AttributeType attributeType;
  DataType dataType;
  int count;
  double max[4] = {0};
  double min[4] = {0};
  bool normalized = false;
  // Note: sparse not supported.
};
struct GltfTexture {
  int sampler, source;
};
struct GltfImage {
  // If bufferView is -1, bytes must hold the decoded image data.
  Bytes decodedData;
  std::string uri;
  std::string mimeType;
  int bufferView=-1;
  //Bytes data;
  //int bufferView = -1;
  int channels; // 1, 3, or 4
  int w, h;
};
struct GltfSampler {
  int minFilter, magFilter, wrapS, wrapT;
};
struct GltfTextureRef {
  int index=-1;
  float scale=1;
  int texCoord=0;
  GltfTextureRef(const nlohmann::json& j);
  inline GltfTextureRef() {};
};
struct GltfMaterial {
  GltfMaterial();
  struct {
    GltfTextureRef baseColorTexture;
    GltfTextureRef metallicRoughnessTexture;
    float baseColorFactor[4], metallicFactor, roughnessFactor;
  } pbr;
  GltfTextureRef normalTexture, occlusionTexture, emissiveTexture;
  float emissiveFactor[3];
  AlphaMode alphaMode;
  float alphaCutoff;
  bool doubleSided;
  std::string name;
};
struct AttributeIndexPair {
  AttributeType attrib;
  int id; // id of accessor
};
struct GltfPrimitive {
  std::vector<AttributeIndexPair> attribs;
  int indices = -1; // accessor
  int material = -1;
  int mode;
  // Note: no targets
};
struct GltfMesh {
  std::vector<GltfPrimitive> primitives;
  std::string name;
  // Note: no weights
};
struct GltfScene {
  std::string name;
  std::vector<int> baseNodes;
};

struct GltfModel {
  public:
    GltfModel(const std::string& dir);

    static GltfModel* fromFile(const std::string& fname);
    static GltfModel* fromGltfString(const std::string& dir, const std::string_view& fname);
    static GltfModel* fromGlbString(const std::string& dir, const std::string_view& fname);

    json jobj;

    void parse(const std::string_view& jsonString);

  //private:
    std::vector<GltfBuffer> buffers;
    std::vector<GltfBufferView> bufferViews;
    std::vector<GltfAccessor> accessors;

    std::vector<GltfNode> nodes;
    std::vector<GltfScene> scenes;
    std::vector<GltfMesh> meshes;

    std::vector<GltfImage> images;
    std::vector<GltfTexture> textures;
    std::vector<GltfSampler> samplers;
    std::vector<GltfMaterial> materials;

    Bytes copyDataFromBufferView(int bv);

    std::string dir;

    std::string printInfo();

    std::string serialize_glb();
};


