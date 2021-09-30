#include "parse.h"
#include "../io.hpp"
#include <iostream>

#include "tdt/render_context.h"
#include "tdt/extra_entities.h"

#include "tdt/camera.h"

#include <Eigen/Core>
#include "tdt/math.h"

ErrorComputer::ErrorComputer(const RenderState& rs) {
  w = rs.w;
  h = rs.h;
  Eigen::Map<Eigen::Matrix<double,4,4,Eigen::RowMajor>> SMVP(screen_mvp);
  Eigen::Map<const Eigen::Matrix<double,4,4,Eigen::RowMajor>> MV(rs.modelView);
  Eigen::Map<const Eigen::Matrix<double,4,4,Eigen::RowMajor>> P(rs.proj);
  Eigen::Matrix<double,4,4,Eigen::RowMajor> S; S <<
    rs.w / 2., 0, 0, rs.w / 2.,
    0, rs.h / 2., 0, rs.h / 2.,
    0, 0, 1, 0,
    0, 0, 0, 1;
  //SMVP = S * MVP;
  SMVP = S * P * MV;

  rs.cam->getPos(eye);
  u = rs.cam->getU();
  v = rs.cam->getV();
}

float TileBase::computeSSE(const ErrorComputer& ec) const {
  auto dist = bndVol.distance(ec);
  float sse = ec.h * geoError * ec.u / dist;
  std::cout << " - dist " << dist << " ec.h " << ec.h << " ge " << geoError << " ec.u " << ec.u << " -> " << sse << "\n";
  return sse;
  //return 9999;
}
float TileBase::computeScreenArea(const ErrorComputer& ec) const {
  return 9999;
}

/*
void TileRoot::open() {
  state = TileBase::State::FRONTIER;
}
void TileRoot::openChildren() {
  state = State::OPEN;
  for (auto c : children) {
    assert(c->state == State::CLOSED);
    c->open();
  }
}
void TileRoot::close() {
  state = State::CLOSED;
}
void TileRoot::closeChildren() {
  assert(state == TileBase::State::OPEN);
  state = TileBase::State::FRONTIER;
  for (auto c : children) {
    assert(c->state == State::OPEN || c->state == State::FRONTIER);
    if (c->state == State::OPEN)
      c->closeChildren();
    c->close();
  }
  state = TileBase::State::FRONTIER;
}
*/

void Tile::upload() {
  assert(model != nullptr);
  assert(entity == nullptr);
  entity = new TileEntity();
  entity->upload(*model->model);
  loaded = true;
  uploaded = true;
  delete model; model = 0;
}
void Tile::unload() {
  assert(entity);
  delete entity; entity = 0;
}
void Tile::openChildren() {
  state = State::OPEN;
  anyChildOpen = true;
  for (auto c : children) {
    assert(c->state == State::CLOSED);
    c->open();
  }
}
void Tile::open() {
  //if (not loaded) {
    //loader->enqueue_load(this);
    //state = State::OPENING;
  //}

  if (isRoot) {
    state = TileBase::State::FRONTIER;
    return;
  }

  if (contentUri.length()) {
    Bytes bytes;
    std::cout << " - reading file " << root->dir + contentUri << "\n";
    readFile(bytes, root->dir + contentUri);
    model = TileModel::parse_b3dm(".", (const char*) bytes.data(), bytes.size());
    upload();
  }
  state = State::FRONTIER;
}
void Tile::closeChildren() {
  assert(state == TileBase::State::OPEN);
  for (auto c : children) {
    assert(c->state == State::OPEN || c->state == State::FRONTIER);
    if (c->state == State::OPEN)
      c->closeChildren();
    c->close();
  }
  state = TileBase::State::FRONTIER;
  anyChildOpen = false;
}
void Tile::close() {
  if (loaded) {
    unload();
  }
  if (entity) delete entity;
  state = State::CLOSED;
}

void Tile::render(RenderState& rs0) {
  // 1) If we are closed:
  //    do nothing
  // 2) If we are open, or if we are closing:
  //    recurse to children, possibly render self if replacement is add
  // 3) If we are opening:
  //    do nothing.

  if (state == TileBase::State::CLOSED) return;

  RenderState rs { rs0 };
  if (transform.size())
    matmul44(rs.modelView, rs0.modelView, transform.data());

  if (
      state == TileBase::State::OPEN ||
      state == TileBase::State::FRONTIER ||
      state == TileBase::State::CLOSING
      ) {
    if (refine == Refinement::REPLACE and anyChildOpen) {
      // do not render self.
    } else {
      if (entity) entity->renderAllNodes(rs);
    }
    if (anyChildOpen) for (auto c : children) c->render(rs);
  }

#if 0
  if (entity) entity->renderAllNodes(rs);
  for (auto& c : children) c->render(rs);

  if (bndVol.type == BoundingVolume::Type::BBOX) {
  } else if (bndVol.type == BoundingVolume::Type::SPHERE) {
    if (rs.sphereEntity != nullptr) {
      //std::cout << " - rendering sphere from bnd vol " << bndVol.data.sphere[3] << ".\n";
      rs.sphereEntity->setPositionAndRadius(bndVol.data.sphere, bndVol.data.sphere[3]);
      rs.sphereEntity->render(rs);
    }
  }
#endif
}
/*
void TileRoot::render(RenderState& rs) {
  for (auto& c : children) c->render(rs);
}
*/

float BoundingVolume::distance(const ErrorComputer& ec) const {
  Eigen::Vector3d t { ec.eye[0], ec.eye[1], ec.eye[2] };

  if (type == Type::SPHERE) {
    Eigen::Vector3d c { data.sphere[0], data.sphere[1], data.sphere[2] };
    return (t - c).norm();
  } else if (type == Type::BBOX) {
    Eigen::Vector3d c0 { data.box[0*3+0], data.box[0*3+1], data.box[0*3+2] };
    Eigen::Vector3d c1 { data.box[1*3+0], data.box[1*3+1], data.box[1*3+2] };
    Eigen::Vector3d c2 { data.box[2*3+0], data.box[2*3+1], data.box[2*3+2] };
    Eigen::Vector3d c3 { data.box[3*3+0], data.box[3*3+1], data.box[3*3+2] };
    auto c = (c0+c1+c2+c3)/4.;
    I guess you must also multiply geometricError by Xform scale...
    //Eigen::Vector3d c { data.box[0], data.box[1], data.box[2] };
    return (t - c).norm();
  } else {
    // TODO must do geodetic -> ecef
    return 1;
  }
}
