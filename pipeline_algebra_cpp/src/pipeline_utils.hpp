#pragma once
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/adjacency_matrix.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/iteration_macros.hpp>
#include <boost/graph/properties.hpp>
#include <boost/property_map/property_map.hpp>

#define dtype float

typedef std::vector<std::vector<dtype>> Matrix;

struct PairHash {
  std::size_t operator()(const std::pair<int, int> &p) const noexcept {
    return (static_cast<std::size_t>(p.first) << 32) ^
           static_cast<std::size_t>(p.second);
  }
};

class GraphData;

struct FirstHopState {
  int v;                     // node index (1-based)
  int i;                     // ancilla index
  int l;                     // neighbor index (1-based)
  std::vector<dtype> state; // 16-dimensional vector
};

// Define custom vertex properties
struct VertexProperties {
  int id;
  dtype feature;
  int classLabel;
  int degree;
  int secondOrderDegree;
};

// Define custom edge properties
struct EdgeProperties {
  int weight;
};

// Define the graph type using Boost Graph Library
typedef boost::adjacency_list<boost::vecS, // OutEdgeList: vector for efficiency
                              boost::vecS, // VertexList: vector for random
                                           // access
                              boost::undirectedS, // Undirected graph
                              VertexProperties,   // Vertex properties
                              EdgeProperties      // Edge properties
                              >
    Graph;

// Vertex and edge descriptors
typedef boost::graph_traits<Graph>::vertex_descriptor Vertex;
typedef boost::graph_traits<Graph>::edge_descriptor Edge;
typedef boost::graph_traits<Graph>::vertex_iterator VertexIterator;
typedef boost::graph_traits<Graph>::edge_iterator EdgeIterator;
typedef boost::graph_traits<Graph>::adjacency_iterator AdjacencyIterator;

class GraphData {
public:
  int n, nrNodes, m, maxD, s, mC1, sC1, p, P;
  Graph graph;
  std::vector<std::vector<int>> adjacencyMatrix;
  std::vector<std::vector<int>> adjacencyList;
  std::vector<std::vector<int>> paddedAdjList; // ✅ padded adjacency list

  // Map type: key=(v,i), value=vector of states
  using StateMap =
      std::unordered_map<std::pair<int, int>, std::vector<std::vector<dtype>>,
                         PairHash>;

  StateMap firstHopStateMap; //

  GraphData() {}

  // Get vertex by ID
  Vertex getVertex(int id) const {
    VertexIterator vi, vi_end;
    for (boost::tie(vi, vi_end) = boost::vertices(graph); vi != vi_end; ++vi) {
      if (graph[*vi].id == id) {
        return *vi;
      }
    }
    throw std::runtime_error("Vertex not found");
  }

  // Calculate degrees and second-order degrees
  void calculateDegrees() {
    VertexIterator vi, vi_end;

    // Calculate first-order degrees
    for (boost::tie(vi, vi_end) = boost::vertices(graph); vi != vi_end; ++vi) {
      graph[*vi].degree = boost::degree(*vi, graph);
    }

    // Calculate second-order degrees
    for (boost::tie(vi, vi_end) = boost::vertices(graph); vi != vi_end; ++vi) {
      int secondOrderDegree = 0;
      AdjacencyIterator ai, ai_end;
      for (boost::tie(ai, ai_end) = boost::adjacent_vertices(*vi, graph);
           ai != ai_end; ++ai) {
        secondOrderDegree += graph[*ai].degree;
      }
      graph[*vi].secondOrderDegree = secondOrderDegree;
    }
  }

  // Build adjacency matrix from BGL graph
  void buildAdjacencyMatrix() {
    adjacencyMatrix.resize(nrNodes, std::vector<int>(nrNodes, 0));

    EdgeIterator ei, ei_end;
    for (boost::tie(ei, ei_end) = boost::edges(graph); ei != ei_end; ++ei) {
      Vertex source = boost::source(*ei, graph);
      Vertex target = boost::target(*ei, graph);
      int src_id = graph[source].id;
      int tgt_id = graph[target].id;

      adjacencyMatrix[src_id][tgt_id] = 1;
      adjacencyMatrix[tgt_id][src_id] = 1;
    }
  }

  void buildAdjacencyList() {
    adjacencyList.assign(nrNodes, {});
    VertexIterator vi, vi_end;
    for (boost::tie(vi, vi_end) = boost::vertices(graph); vi != vi_end; ++vi) {
      int v_id = graph[*vi].id;
      AdjacencyIterator ai, ai_end;
      for (boost::tie(ai, ai_end) = boost::adjacent_vertices(*vi, graph);
           ai != ai_end; ++ai) {
        adjacencyList[v_id].push_back(graph[*ai].id);
      }
    }
  }

  void buildPaddedAdjList(int padSize) {
    paddedAdjList.assign(
        nrNodes, std::vector<int>(padSize, nrNodes)); // fill with dummy nrNodes

    VertexIterator vi, vi_end;
    for (boost::tie(vi, vi_end) = boost::vertices(graph); vi != vi_end; ++vi) {
      int v_id = graph[*vi].id;
      AdjacencyIterator ai, ai_end;
      int idx = 0;
      for (boost::tie(ai, ai_end) = boost::adjacent_vertices(*vi, graph);
           ai != ai_end && idx < padSize; ++ai, ++idx) {
        paddedAdjList[v_id][idx] = graph[*ai].id;
      }
      // rest stay = nrNodes (already padded)
    }
  }

  // ✅ Build map from firstHopStates
  void buildFirstHopStateMap(const std::vector<FirstHopState> &firstHopStates) {
    firstHopStateMap.clear();
    firstHopStateMap.reserve(firstHopStates.size());

    for (const auto &s : firstHopStates) {
      firstHopStateMap[{s.v, s.i}].push_back(s.state);
    }
  }
};
