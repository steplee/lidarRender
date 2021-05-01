#include "common.h"
#include <unordered_set>
#include <unordered_map>
#include <stack>
#include <queue>

struct Edge {
  int32_t u;
  int32_t v;
  int32_t c;
  //int32_t f; // Can't store in unordered_set and modify.

  bool operator==(const Edge& o) const {
    return u == o.u and v == o.v;
  }
};


namespace std {
    template<> struct hash<std::pair<int32_t,int32_t>>
    {
        std::size_t operator()(std::pair<int32_t,int32_t> const& s) const noexcept
        {
          int64_t a = (static_cast<int64_t>(s.first)<<32) | static_cast<int64_t>(s.second);
          return hash<int64_t>{}(a);
        }
    };

    // Note: Capacity is not in the hash
    template<> struct hash<Edge>
    {
        std::size_t operator()(Edge const& s) const noexcept
        {
          int64_t a = (static_cast<int64_t>(s.u)<<32) | static_cast<int64_t>(s.v);
          // TODO: Does this stack allocate???
          return hash<int64_t>{}(a);
        }
    };
};

struct MaxFlowMinCut {
  MaxFlowMinCut(int N);

  // Number of verts.
  int N;

  // Source and sink
  int32_t s, t;

  // Input capacities (pairs must be ordered!)
  //std::unordered_map<std::pair<int32_t,int32_t>, int32_t> c;
  std::vector<std::unordered_set<Edge>> c; // Edges, each inner item is (u,v, cap)

  // Resulting assigned flow (length = length of e).
  //std::vector<int32_t> flow;
  std::unordered_map<Edge, int32_t> flow;
  int maxFlow = 0;

  // Resulting ST cut.
  std::unordered_set<int32_t> minCutS;
  std::unordered_set<int32_t> minCutT;
  std::unordered_set<Edge> minCutEdges;

  void setSourceSink(int32_t s, int32_t t);
  void addEdge(int32_t u, int32_t v, int32_t c);
  void run();

  // Erase the graph and the flow, to save some memory (note: this will mess up printViz())
  void deallocateSome();

  std::string printViz();
};
