#include "gltf.h"
#include <cassert>
#include <fstream>
#include <iostream>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_HDR
#define STBI_NO_LINEAR
#define STBI_NO_GIF
#define STBI_NO_PSD
#define STBI_NO_BMP
#define STBI_NO_PIC
#define STBI_NO_PNM
#define STBI_NO_TGA
#define STBI_NO_STDIO
#include "stb_image.h"

#include "io.hpp"

#include "math.h"

#define ENSURE(x) assert((x));

static void decodeImage(Bytes& out, int& outW, int& outH, int& outC, const Bytes& in, const std::string& type) {
  uint8_t* data = stbi_load_from_memory(in.data(), in.size(), &outW, &outH, &outC, 0);
  out.resize(outC*outW*outH);
  memcpy(out.data(), data, outC*outW*outH);
  stbi_image_free(data);
}

GltfModel* GltfModel::fromFile(const std::string& fname) {
  int n = fname.length();

  std::ifstream ifs(fname);
  std::string str;
  ifs.seekg(0, std::ios::end);
  str.reserve(ifs.tellg());
  ifs.seekg(0, std::ios::beg);
  str.assign((std::istreambuf_iterator<char>(ifs)),
                  std::istreambuf_iterator<char>());

  std::string dir = fname.substr(0, fname.rfind("/")) + "/";

  if ((fname[n-3] == 'G' or fname[n-3] == 'g') and
      (fname[n-2] == 'L' or fname[n-2] == 'l') and
      (fname[n-1] == 'B' or fname[n-1] == 'b'))
      return fromGlbString(dir, str);
  else
      return fromGltfString(dir, str);
}

GltfModel* GltfModel::fromGltfString(const std::string& dir, const std::string_view& str) {
  GltfModel* model = new GltfModel(dir);
  model->parse(str);
  return model;
}
GltfModel* GltfModel::fromGlbString(const std::string& dir, const std::string_view& str) {

  ENSURE(str[0] == 'g');
  ENSURE(str[1] == 'l');
  ENSURE(str[2] == 'T');
  ENSURE(str[3] == 'F');
  //ENSURE(*static_cast<const int32_t*>(str.data()+4) == 2);
  ENSURE(*(const int32_t*)(str.data()+4) == 2);

  int32_t len = *(const int32_t*)(str.data()+8);

  int32_t chunk0_len  = *(const int32_t*)(str.data()+12);
  int32_t chunk0_type = *(const int32_t*)(str.data()+16);
  const char* chunk0_data = str.data() + 20;
  std::cout << " - chunk type : " << chunk0_type << " " << 0x4E4F534A << "\n";
  ENSURE(chunk0_type == 0x4E4F534A);
  std::string jsonString(chunk0_data, chunk0_len);

  // Chunk1 only exists if there is space.
  GltfBuffer glbBuffer;
  if (chunk0_len + 20 < len) {
    int32_t chunk1_len  = *(const int32_t*)(str.data()+20+chunk0_len);
    int32_t chunk1_type = *(const int32_t*)(str.data()+24+chunk0_len);
    const uint8_t* chunk1_data = (const uint8_t*) str.data() + 28 + chunk0_len;
    std::cout << " - off " << chunk0_len << " chunk type : " << chunk1_type << " " << 0x004E4942 << std::endl;
    ENSURE(chunk1_type == 0x004E4942);

    glbBuffer.data = Bytes(chunk1_data, chunk1_data + chunk1_len);
    glbBuffer.byteLength = chunk1_len;
  }

  // Could be more chunks for extensions. Nevermind them

  GltfModel* model = new GltfModel(dir);
  model->buffers.insert(model->buffers.begin(), glbBuffer);
  model->parse(jsonString);

  return model;
}

static bool readUri(Bytes& out, const std::string& uri, const std::string& dir="./") {
  if (uri.length() > 4 and 
      uri[0] == 'd' and uri[1] == 'a' and uri[2] == 't' and uri[3] == 'a' and uri[4] == ':') {
    // parse the base64 here...
  } else {
    std::string f = dir + uri;
    return readFile(out, f);
  }
  return true;
}

static std::string attributeTypeString(const AttributeType& s) {
  if (s == AttributeType::POSITION) return "POSITION";
  if (s == AttributeType::NORMAL) return "NORMAL";
  if (s == AttributeType::TANGENT) return "TANGENT";
  if (s == AttributeType::TEXCOORD_0) return "TEXCOORD_0";
  if (s == AttributeType::TEXCOORD_1) return "TEXCOORD_1";
  if (s == AttributeType::COLOR_0) return "COLOR_0";
  if (s == AttributeType::JOINTS_0) return "JOINTS_0";
  if (s == AttributeType::WEIGHTS_0) return "WEIGHTS_0";
  if (s == AttributeType::BATCHID) return "_BATCHID";
}
static AttributeType parseAttributeType(const std::string& s) {
  if (s == "POSITION") return AttributeType::POSITION;
  if (s == "NORMAL") return AttributeType::NORMAL;
  if (s == "TANGENT") return AttributeType::TANGENT;
  if (s == "TEXCOORD_0") return AttributeType::TEXCOORD_0;
  if (s == "TEXCOORD_1") return AttributeType::TEXCOORD_1;
  if (s == "COLOR_0") return AttributeType::COLOR_0;
  if (s == "JOINTS_0") return AttributeType::JOINTS_0;
  if (s == "WEIGHTS_0") return AttributeType::WEIGHTS_0;
  if (s == "_BATCHID") return AttributeType::BATCHID;
  ENSURE(false);
}
static int dataTypeCount(const DataType& s) {
  if (s == DataType::SCALAR) return 1;
  if (s == DataType::VEC2) return 2;
  if (s == DataType::VEC3) return 3;
  if (s == DataType::VEC4) return 4;
  if (s == DataType::MAT2) return 4;
  if (s == DataType::MAT3) return 9;
  if (s == DataType::MAT4) return 16;
}
static std::string dataTypeString(const DataType& s) {
  if (s == DataType::SCALAR) return "SCALAR";
  if (s == DataType::VEC2) return "VEC2";
  if (s == DataType::VEC3) return "VEC3";
  if (s == DataType::VEC4) return "VEC4";
  if (s == DataType::MAT2) return "MAT2";
  if (s == DataType::MAT3) return "MAT3";
  if (s == DataType::MAT4) return "MAT4";
}
static DataType parseDataType(const std::string& s) {
  if (s == "SCALAR") return DataType::SCALAR;
  if (s == "VEC2") return DataType::VEC2;
  if (s == "VEC3") return DataType::VEC3;
  if (s == "VEC4") return DataType::VEC4;
  if (s == "MAT2") return DataType::MAT2;
  if (s == "MAT3") return DataType::MAT3;
  if (s == "MAT4") return DataType::MAT4;
  throw std::runtime_error("could not parseDataType() " + s);
}

static void printMatrix(double a[16]) {
  for (int i=0; i<4; i++) {
  for (int j=0; j<4; j++) {
    std::cout << " " << a[i*4+j];
  }
  std::cout << "\n";
  }
}

GltfModel::GltfModel(const std::string& dir_) : dir(dir_) {
}

GltfMaterial::GltfMaterial() {
  pbr.baseColorFactor[0] = -1;
  pbr.metallicFactor = -1;
  pbr.roughnessFactor = -1;
  emissiveFactor[0] = -1;
  alphaMode = AlphaMode::DEFAULT;
  alphaCutoff = -1;
  doubleSided = true;
}

void GltfModel::parse(const std::string_view& jsonString) {
  jobj = json::parse(jsonString);

  // Parse it.
  for (const json &j : jobj["nodes"]) {
    GltfNode node;
    node.mesh = j.value("mesh", -1);
    // TODO parse it. aslo rot/trans/scale
    if (j.contains("matrix"))
      //for (int i=0; i<16; i++) node.xform[i] = j["matrix"][i];
      for (int i=0; i<4; i++) for (int k=0; k<4; k++) node.xform[i*4+k] = j["matrix"][k*4+i];
    else
      for (int i=0; i<16; i++) node.xform[i] = i%4 == i/4;
    if (j.contains("scale")) {
      double scale[3];
      scale[0] = j["scale"][0];
      scale[1] = j["scale"][1];
      scale[2] = j["scale"][2];
      for (int i=0; i<3; i++)
        for (int k=0; k<4; k++)
          //node.xform[k*4+i] *= scale[i];
          node.xform[i*4+k] *= scale[i];
    }
    if (j.contains("rotation")) {
      double x = j["rotation"][0];
      double y = j["rotation"][1];
      double z = j["rotation"][2];
      double w = j["rotation"][3];
      double R[16];
      R[0] = 1. - 2*(y*y+z*z);
      R[1] = 2*(x*y-w*z);
      R[2] = 2*(w*y+x*z);
      R[3] = 0;
      R[4] = 2*(x*y+w*z);
      R[5] = 1. - 2*(x*x+z*z);
      R[6] = 2*(y*z-w*x);
      R[7] = 0;
      R[8] = 2*(x*z-w*y);
      R[9] = 2*(w*x+y*z);
      R[10] = 1. - 2*(x*x+y*y);
      R[11] = 0;
      R[12] = R[13] = R[14] = 0;
      R[15] = 1;
      //std::cout << " - RotationMatrix:\n"; printMatrix(R);
      double tmp[16]; memcpy(tmp, node.xform, sizeof(tmp));
      matmul44(node.xform, tmp, R);
    }
    if (j.contains("translation")) {
      node.xform[3] += (double)j["translation"][0];
      node.xform[7] += (double)j["translation"][1];
      node.xform[11] += (double)j["translation"][2];
    }
    //std::cout << " - Node Transform:\n"; printMatrix(node.xform);
    if (j.contains("children")) for (const int& i: j["children"]) node.children.push_back(i);
    nodes.push_back(node);
    printf(" - Parsed Node Xform:\n"); printMatrix(node.xform);
  }

  for (const json &j : jobj["meshes"]) {
    GltfMesh mesh;
    mesh.name = j.value("name", "untitledMesh" + std::to_string(meshes.size()));
    for (const json& jj : j["primitives"]) {
      GltfPrimitive prim;
      for (auto it = jj["attributes"].begin(); it != jj["attributes"].end(); ++it) {
        AttributeIndexPair aip { parseAttributeType(it.key()) , it.value() };
        prim.attribs.push_back(aip);
      }
      prim.indices = jj.value("indices", -1);
      prim.mode = jj.value("mode", 0x004); // tris
      prim.material = jj.value("material", -1);
      mesh.primitives.push_back(prim);
    }
    meshes.push_back(mesh);
  }

  for (const json &j : jobj["buffers"]) {
    GltfBuffer buf;
    bool good = false;
    if (j.contains("uri")) {
      ENSURE(readUri(buf.data, j["uri"], dir));
      buf.byteLength = j["byteLength"];
      good = true;
    } else
      {
        // The GLB buffer is handled outside of the constructor.
      }
    if (good) buffers.push_back(buf);
  }

  for (const json &j : jobj["bufferViews"]) {
    GltfBufferView bv;
    bv.buffer = j["buffer"];
    bv.byteLength = j["byteLength"];
    bv.byteStride = j.value("byteStride", 0);
    bv.byteOffset = j.value("byteOffset", 0);
    bv.target = j.value("target", -1);
    bufferViews.push_back(bv);
  }

  for (const json &j : jobj["accessors"]) {
    GltfAccessor acc;
    acc.bufferView = j["bufferView"];
    acc.byteOffset = j.value("byteOffset",0);
    acc.componentType = j["componentType"];
    acc.normalized = j.value("normalized",false);
    acc.dataType = parseDataType(j["type"]);
    acc.count = j["count"];
    memset(acc.max, 0, sizeof(acc.max));
    memset(acc.min, 0, sizeof(acc.max));
    if (j.contains("max")) {
      int ii = 0;
      for (const double& v : j["max"]) acc.max[ii++] = v;
    }
    if (j.contains("min")) {
      int ii = 0;
      for (const double& v : j["min"]) acc.min[ii++] = v;
    }
    accessors.push_back(acc);
  }


  for (const json &j : jobj["images"]) {
    GltfImage img;
    // Either file uri, data uri, or link to bufferView
    Bytes data;
    if (j.contains("uri")) {
      img.uri = j["uri"];
      ENSURE(readUri(data, j["uri"], dir));
    } else {
      int bv = j["bufferView"];
      img.bufferView = bv;
      data = copyDataFromBufferView(bv);
    }
    if (j.contains("mimeType")) img.mimeType = j["mimeType"];

    // TODO Decode data...
    //img.channels = 3;
    //img.w = 512;
    //img.h = 512;
    //img.decodedData.resize(img.channels*img.w*img.h);
    //std::fill(img.decodedData.begin(), img.decodedData.end(), 155);
    decodeImage(img.decodedData, img.w, img.h, img.channels, data, "");
    //std::cout << " - Decoded Image : " << img.w << " "<< img.h << " " << img.channels << "\n";
    //for (int i=0; i<512*512*3; i++) std::cout << (int)img.decodedData[i] << " ";
    //std::cout << "\n";

    images.push_back(img);
  }

  for (const json &j : jobj["textures"]) {
    GltfTexture t;
    t.sampler = j.value("sampler", -1);
    t.source = j.value("source", -1);
    textures.push_back(t);
  }

  for (const json &j : jobj["samplers"]) {
    GltfSampler s;
    s.minFilter = j.value("minFilter",0x2601); // GL_LINEAR
    s.magFilter = j.value("magFilter",0x2601);
    s.wrapS = j.value("wrapS",10497);
    s.wrapT = j.value("wrapT",10497);
    samplers.push_back(s);
  }

  for (const json &j : jobj["materials"]) {
    GltfMaterial m;
    if (j.contains("name")) m.name = j["name"];
    if (j.contains("pbrMetallicRoughness")) {
      auto jj = j["pbrMetallicRoughness"];
      if (jj.contains("baseColorTexture")) m.pbr.baseColorTexture = GltfTextureRef { jj["baseColorTexture"] };
      if (jj.contains("metallicRoughnessTexture")) m.pbr.metallicRoughnessTexture = GltfTextureRef { jj["metallicRoughnessTexture"] };
      if (jj.contains("baseColorFactor"))
        for (int i=0; i<4; i++) m.pbr.baseColorFactor[i] = jj["baseColorFactor"][i];
      else
        for (int i=0; i<4; i++) m.pbr.baseColorFactor[i] = 1;
      m.pbr.metallicFactor = jj.value("metallicFactor", 1);
      m.pbr.roughnessFactor = jj.value("roughnessFactor", 1);
    }
    if (j.contains("normalTexture")) m.normalTexture = GltfTextureRef { j["normalTexture"] };
    if (j.contains("emissiveTexture")) m.emissiveTexture = GltfTextureRef { j["emissiveTexture"] };
    if (j.contains("emissiveFactor"))
      for (int i=0; i<3; i++) m.emissiveFactor[i] = j["emissiveFactor"][i];
    else
      for (int i=0; i<3; i++) m.emissiveFactor[i] = 0;
    if (j.contains("occlusionTexture")) m.occlusionTexture = GltfTextureRef { j["occlusionTexture"] };
    materials.push_back(m);
    //std::cout << " - pushing material.\n" << "\n";
  }

  for (const json& j : jobj["scenes"]) {
    GltfScene s;
    s.name = j.value("name", "<noname>");
    for (const int i : j["nodes"]) s.baseNodes.push_back(i);
    scenes.push_back(s);
  }

}

GltfTextureRef::GltfTextureRef(const nlohmann::json& j) {
  scale = j.value("scale", 1);
  index = j.value("index", -1);
  texCoord = j.value("texCoord", 1);
}

Bytes GltfModel::copyDataFromBufferView(int bv_) {
  Bytes out;
  const auto& bv = bufferViews[bv_];
  out.resize(bv.byteLength);
  memcpy(out.data(), buffers[bv.buffer].data.data() + bv.byteOffset, bv.byteLength);
  return out;
}

#include <sstream>
std::string GltfModel::printInfo() {
  char buf[4096]; int n = 0;
  n += sprintf(buf+n, " - GltfModel at dir '%s'\n", dir.c_str());
  n += sprintf(buf+n, "\t- (%d scenes / %d nodes / %d meshes)\n", scenes.size(), nodes.size(), meshes.size());
  n += sprintf(buf+n, "\t- (%d buffers) (sizes ", buffers.size());
  for (int i=0; i<buffers.size(); i++) n += sprintf(buf+n, "%d", buffers[i].byteLength); n += sprintf(buf+n, ")\n");
  n += sprintf(buf+n, "\t- (%d bufferViews)\n", bufferViews.size());
  n += sprintf(buf+n, "\t- (%d accessors)\n", accessors.size());
  n += sprintf(buf+n, "\t- (%d images / %d textures / %d materials / %d samplers)\n", images.size(), textures.size(), materials.size(), samplers.size());

  return std::string{buf, n};
}

template<class T>
static void copy_to_string(std::string& out, const T& t) {
  std::string s { (char*)(&t) , sizeof(T) };
  out = out + s;
}
std::string GltfModel::serialize_glb() {
  std::string out;
  std::string chunk0;
  std::string chunk1;

  ENSURE(buffers.size() == 1);

  // Strategy:
  // Create json while creating the glb-buffer.
  // Cannot fill out the json[buffers][0][length] until we create all buffers.
  // (Because the length may change while creating e.g. images)
  // Cannot write header until we fill the other two.

  chunk1 = std::string((const char*)buffers[0].data.data(), buffers[0].byteLength);

  nlohmann::json jobj;
  jobj["scenes"] = json::array();
  jobj["buffers"] = json::array();
  jobj["bufferViews"] = json::array();
  jobj["images"] = json::array();
  jobj["textures"] = json::array();
  jobj["samplers"] = json::array();
  jobj["nodes"] = json::array();
  jobj["meshes"] = json::array();
  jobj["accessors"] = json::array();
  jobj["materials"] = json::array();
  for (const auto& b : bufferViews) {
    nlohmann::json j; j["buffer"] = b.buffer;
    j["byteLength"] = b.byteLength;
    j["byteOffset"] = b.byteOffset;
    if (b.byteStride != 0) j["byteStride"] = b.byteStride;
    if (b.target != 0) j["target"] = b.target;
    jobj["bufferViews"].push_back(j);
  }
  for (const auto& s : scenes) {
    nlohmann::json j; j["name"] = s.name;
    j["nodes"] = s.baseNodes;
    jobj["scenes"].push_back(j);
  }
  for (const auto& i : images) {
    nlohmann::json j; j["channels"] = i.channels;
    j["width"] = i.w;
    j["height"] = i.h;
    if (i.mimeType.length() > 0) j["mimeType"] = i.mimeType;
    if (i.uri.length()) {
      // TODO. Must copy image data over to buffer!
      //j["bufferView"] = ,,,;
    } else
      j["bufferView"] = i.bufferView;
    jobj["images"].push_back(j);
  }
  for (const auto& t : textures) {
    nlohmann::json j;
    if (t.sampler != -1) j["sampler"] = t.sampler;
    j["source"] = t.source;
    jobj["textures"].push_back(j);
  }
  for (const auto& n : nodes) {
    nlohmann::json j;
    j["mesh"] = n.mesh;
    j["matrix"] = json::array();
    for (int i=0; i<16; i++) j["matrix"].push_back(n.xform[i]);
    if (n.children.size()) j["children"] = n.children;
    jobj["nodes"].push_back(j);
  }
  for (const auto& m : meshes) {
    nlohmann::json j;
    j["name"] = m.name;
    j["primitives"] = json::array();
    for (auto p : m.primitives) {
      nlohmann::json jj;
      jj["indices"] = p.indices;
      if (p.material != -1) jj["material"] = p.material;
      jj["mode"] = p.mode;
      jj["attributes"] = json::object();
      for (auto ap : p.attribs) {
        auto k = attributeTypeString(ap.attrib);
        jj["attributes"][k] = ap.id;
      }
      j["primitives"].push_back(jj);
    }
    jobj["meshes"].push_back(j);
  }
  for (const auto& a : accessors) {
    nlohmann::json j;
    j["bufferView"] = a.bufferView;
    j["byteOffset"] = a.byteOffset;
    j["componentType"] = a.componentType;
    j["type"] = dataTypeString(a.dataType);
    j["count"] = a.count;
    j["normalized"] = a.normalized;
    if (a.max[0] != a.min[0]) {
      j["max"] = json::array();
      j["min"] = json::array();
      for (int i=0; i<dataTypeCount(a.dataType); i++) j["max"].push_back(a.max[i]);
      for (int i=0; i<dataTypeCount(a.dataType); i++) j["min"].push_back(a.min[i]);
    }
    jobj["accessors"].push_back(j);
  }
  for (const auto& m : materials) {
    // TODO not complete!
    nlohmann::json j;
    j["pbrMetallicRoughness"] = json::object();
    if (m.pbr.baseColorTexture.index != -1) {
      j["pbrMetallicRoughness"]["baseColorTexture"] = json::object();
      j["pbrMetallicRoughness"]["baseColorTexture"]["index"] = m.pbr.baseColorTexture.index;
    }
    if (m.pbr.baseColorFactor[0] >= 0) {
      j["pbrMetallicRoughness"]["baseColorFactor"] = json::array();
      for (int i=0; i<4; i++) j["pbrMetallicRoughness"].push_back(m.pbr.baseColorFactor[i]);
    }
    if (j["pbrMetallicRoughness"].size() == 0) j.erase("pbrMetallicRoughness");
    if (m.emissiveFactor[0] >= 0) {
      j["emissiveFactor"] = json::array();
      for (int i=0; i<3; i++) j["emissiveFactor"].push_back(m.emissiveFactor[i]);
    }
    //if (m.alphaMode != AlphaMode::DEFAULT) j["alphaMode"] = m.
    if (m.name.length() > 0) j["name"] = m.name;
    jobj["materials"].push_back(j);
  }

  // NOTE: Do buffer[0] last, so that length is correct.
  for (const auto& b : buffers) {
    // NOTE: this is the GLB buffer, has NO URI.
    nlohmann::json j; j["byteLength"] = chunk1.length();
    jobj["buffers"].push_back(j);
  }

  jobj["asset"] = json::object();
  jobj["asset"]["generator"] = "https://github.com/steplee";
  jobj["asset"]["version"] = "2.0";

  chunk0 = jobj.dump();

  std::cout << " - Serialized JSON:\n" << jobj << "\n";

  while (chunk0.length() % 4 != 0) chunk0 += ' ';
  while (chunk1.length() % 4 != 0) chunk1 += '\0';

  auto chunk0_type = "JSON";
  std::string chunk0_sz; copy_to_string(chunk0_sz, (uint32_t)chunk0.length());
  chunk0 = chunk0_sz + chunk0_type + chunk0;

  std::string chunk1_type;
  //chunk1_type.push_back(0); chunk1_type = chunk1_type + "BIN";
  chunk1_type = chunk1_type + "BIN"; chunk1_type.push_back(0);
  std::string chunk1_sz; copy_to_string(chunk1_sz, (uint32_t)chunk1.length());
  chunk1 = chunk1_sz + chunk1_type + chunk1;

  int32_t len = 12 + chunk0.length() + chunk1.length();

  out = "glTF";
  int32_t two = 2; copy_to_string(out, two);
  copy_to_string(out, len);

  out = out + chunk0 + chunk1;

  return out;

}
