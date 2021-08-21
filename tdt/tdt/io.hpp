#include <fstream>
#include <vector>

using Bytes = std::vector<uint8_t>;

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
