#include "tdt/window.h"

#include "tdt/gltf.h"
#include "tdt/gltf_entity.h"

#include "tdt/tdtiles/parse.h"

#include <unistd.h>
#include <iostream>


int main(int argc, char** argv) {

  std::string fname { argv[1] };

  auto tb = TileBase::fromFile(fname);

}

