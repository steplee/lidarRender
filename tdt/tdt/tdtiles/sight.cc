#include "sight.h"
#include "parse.h"
#include "tdt/render_context.h"

Sight::Sight() :
  cam(Camera(CamSpec(32,32,1)))
{
}

void Sight::addTileset(const std::string& filepath) {
}

void Sight::render(RenderState&) {
}

void Sight::update(RenderState&) {
  for (auto root : roots) {
  }
}
void Sight::updateTile(TileBase* tile, RenderState& rs) {

  for (auto c : tile->children) {
    updateTile(c, rs);
  }
}

bool tileWantsToOpen(const Camera& cam, const TileBase& tile) {
  auto ge = tile.geoError;
  return true;
}
