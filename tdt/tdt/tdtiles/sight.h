#pragma once

#include <vector>
#include <string>

#include "tdt/camera.h"

class TileRoot;
class TileBase;
class Tile;
class RenderState;

bool tileWantsToOpen(const Camera& cam, const TileBase& tile);

class Sight {
  public:
    Sight();

    void addTileset(const std::string& dir, const std::string& filepath);

    void render(RenderState&);
    void update(RenderState&);
    void updateTile(TileBase* tile, RenderState& rs);

  private:
    std::vector<TileRoot*> roots;

    //Camera cam;

};
