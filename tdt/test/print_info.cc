#include "tdt/gltf.h"
#include "tdt/gltf_entity.h"

#include <unistd.h>
#include <iostream>


int main(int argc, char** argv) {

  std::string fname { argv[1] };
  GltfModel* model = GltfModel::fromFile(fname);

  std::string s = model->printInfo();
  std::cout << s << "\n";

  return 0;
}
