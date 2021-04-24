#include "mfmc.h"
#include <iostream>

MaxFlowMinCut::MaxFlowMinCut(int N) : N(N) {
  c.resize(N);
}

void MaxFlowMinCut::setSourceSink(int32_t s, int32_t t) {
  this->s = s;
  this->t = t;
}
void MaxFlowMinCut::addEdge(int32_t u, int32_t v, int32_t c) {
  this->c[u].insert(Edge{u,v,c});
  //this->c[v].insert(Edge{v,u,c}); // residual edge.
}

void MaxFlowMinCut::run() {
  //flow.resize(N);
  //for (int i=0; i<N; i++) flow[i] = 0;
  //for (int i=0; i<N; i++) for (auto& e : c[i]) *const_cast<int32_t*>(&e.f) = 0;

  maxFlow = 0;
  std::cout << " - computing flow from " << std::to_string(s) << " to " << std::to_string(t) << "\n";

  while (true) {
    std::queue<int32_t> q;
    q.push(s);
    std::vector<const Edge*> pred;
    pred.resize(N, nullptr);
    while (not q.empty()) {
      int32_t cur = q.front(); q.pop();
      for (auto& e : c[cur]) {
        int32_t u = e.u;
        if (u!=cur) exit(1);
        int32_t v = e.v;
        int32_t cap = e.c;
        if (pred[v] == nullptr and v != s and cap > flow[e]) {
          pred[v] = &e;
          q.push(v);
        }
      }
    }

    if (pred[t] != nullptr) {
      int32_t df = 1073741824;

      for (auto e = pred[t]; e != nullptr; e = pred[e->u])
        df = std::min(df, e->c - flow[*e]);

      std::string path = " - path: ";
      for (auto e = pred[t]; e != nullptr; e = pred[e->u]) {
        flow[*e] = flow[*e] + df;
        //flow[Edge{e->v,e->u,e->c}] -= df;
        path += std::to_string(e->v) + " ";
      }
      path += "(df " + std::to_string(df) + ")";
      std::cout << path << "\n";
      maxFlow += df;
    } else break;
  }

  std::cout << " - finding min cut.\n";
  {
    std::queue<int32_t> q;
    q.push(s);
    minCutS.insert(s);
    while (not q.empty()) {
      int32_t cur = q.front(); q.pop();
      for (auto& e : c[cur]) {
        int32_t u = e.u;
        int32_t v = e.v;
        int32_t cap = e.c;
        if (flow[e] < cap) {
          if (minCutS.find(v) == minCutS.end()) q.push(v);
          minCutS.insert(v);
        }
      }
    }
    for (int32_t i=0; i<N; i++) {
      if (minCutS.find(i) == minCutS.end()) minCutT.insert(i);
      else {
        for (auto& e : c[i])
          if (minCutS.find(e.v) == minCutS.end()) minCutEdges.insert(e);
      }
    }
  }
  std::cout << " - |S| = " << minCutS.size() << ", |T| = " << minCutT.size() << ", |S^T| " << minCutEdges.size() << "\n";

}


std::string MaxFlowMinCut::printViz() {
  std::string s;
  s += "digraph G {\n";
  s += "  nslimit=.1;\n";
  s += "  nslimit1=.1;\n";
  s += "  mclimit=.1;\n";
  s += "  " + std::to_string(this->s) + " [style=filled, fillcolor=green]\n";
  s += "  " + std::to_string(this->t) + " [style=filled, fillcolor=blue]\n";
  for (auto i : minCutS) if (i != this->s)
    s += "  " + std::to_string(i) + " [style=filled, fillcolor=palegreen]\n";
  for (auto i : minCutT) if (i != this->t)
    s += "  " + std::to_string(i) + " [style=filled, fillcolor=lightblue]\n";
  for (auto& ee : c) {
    for (auto& e : ee) {
      int32_t u = e.u;
      int32_t v = e.v;
      int32_t c = e.c;
      int32_t f = flow[e];
      std::string color = f>0 ? "orange" : "black";
      s += "  " + std::to_string(u) + " -> " + std::to_string(v) + "[label=\"" +
            std::to_string(f) + "/" + std::to_string(c) + "\", fontcolor="+color+"];\n";
    }
  }
  s = s + "}";
  return s;
}

void MaxFlowMinCut::deallocateSome() {
  c.clear();
  flow.clear();
}
