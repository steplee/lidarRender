#pragma once
#include <cstdint>

class RenderState;

class Entity {
  public:
    virtual void init() =0;
    virtual void destroy() =0;
    virtual void render(RenderState& rs) =0;
};

class SphereEntity : public Entity {
  public:
    float color[4] = { 0, 1, 0, .2 };
    double pos[3] = {0,0,0}, radius = 1;

    virtual void init() override;
    virtual void destroy() override;
    virtual void render(RenderState& rs) override;

    void setPositionAndRadius(double pos[3], double r);
  private:
    uint32_t vbo = 0;
    uint32_t ibo = 0;
    int ninds = 0;
};


class BoxEntity : public Entity {
  public:
    float color[4] = { 0, 1, 0, .2 };

    virtual void init() override;
    virtual void destroy() override;
    virtual void render(RenderState& rs) override;
  private:
    uint32_t vbo = 0;
    uint32_t ibo = 0;
    int ninds = 0;
};
