#include "sight.h"
#include "parse.h"
#include "tdt/render_context.h"
#include <Eigen/Core>
#include <iostream>

Sight::Sight()
  //: cam(Camera(CamSpec(32,32,1)))
{
}

void Sight::addTileset(const std::string& dir, const std::string& filepath) {
  auto root = new TileRoot(dir, filepath);
  root->root = root;
  roots.push_back(root);
}

void Sight::render(RenderState& rs) {
  for (auto r : roots) {
    if (r->state == TileBase::State::OPEN ||
        r->state == TileBase::State::FRONTIER ||
        r->state == TileBase::State::CLOSING) r->render(rs);
  }
}

void Sight::update(RenderState& rs) {
  std::cout << " update.\n";
  for (auto root : roots) {
    std::cout << " update1.\n";
    updateTile(root, rs);
  }
}

void Sight::updateTile(TileBase* tile, RenderState& rs) {
  // 1) If we are closed:
  //    1) project and check whether to open
  // 2) If we are open:
  //    I  ) Check SSE and visibility to see if to open children
  //    II ) If refinement is 'add'
  //        a) any children that want to close can be closed.
  //    III) If refinement is 'replace':
  //        a) if all children want to close, make self active and close them.
  //           otherwise if some child wants to stay open, they all must.
  // 3) If we are opening or closing:
  //    do nothing.
  //
  //    TODO XXX:
  //    Bad. When my error is too large, open the children. Func needs to be written like that

  ErrorComputer ec(rs);

  float tol = 512;

  if (tile->state == TileBase::State::CLOSED) {
    // only a root may open itself.
    if (tile->root == tile) {
      auto sse = tile->computeSSE(ec);
      if (sse >= tol) {
        tile->open();
        tile->openChildren();
        std::cout << " - root opening itself.\n";
      }
    }
  } else if (tile->state == TileBase::State::OPEN) {
    // See if we should close all children.
    auto sse = tile->computeSSE(ec);
    if (sse < tol) tile->closeChildren();
    else for (auto& c : tile->children) updateTile(c,rs);
  } else if (tile->state == TileBase::State::FRONTIER) {
    // See if we should open all children.
    auto sse = tile->computeSSE(ec);
    if (sse >= tol) tile->openChildren();
  }


  /*
  if (tile->state == TileBase::State::CLOSED) {
    if (tile->geoError == 0) {
      std::cout << " - closed tile had 0 geoError, so opening unconditionally\n";
      tile->open();
    } else {
      auto sse = tile->computeSSE(ec);
      std::cout << " - closed tile had sse " << sse << "\n";
      if (sse > 32) {
        tile->open();
      }
    }
  }
  else if (tile->state == TileBase::State::OPEN) {
    for (auto c : tile->children) updateTile(c, rs);
    bool close_me = true;
    if (tile->geoError == 0) close_me = false;
    for (auto c : tile->children) if (c->state != TileBase::State::CLOSED) close_me = false;
    if (close_me) {
      auto sse = tile->computeSSE(ec);
      std::cout << " - open tile had sse " << sse << ".\n";
      if (sse >= 32) close_me = false;
      if (close_me) {
        std::cout << " - closing open tile.\n";
        tile->close();
      }
    }
  } else {
    // Case (3), do nothing
  }
  */
}

bool tileWantsToOpen(const Camera& cam, const TileBase& tile) {
  auto ge = tile.geoError;
  return true;
}
