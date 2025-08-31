#pragma once

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/adjacency_matrix.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/iteration_macros.hpp>
#include <boost/graph/properties.hpp>
#include <boost/property_map/property_map.hpp>
#include <iostream>

#define dtype float

typedef std::vector<std::vector<dtype>> Matrix;

struct PairHash {
  std::size_t operator()(const std::pair<int, int> &p) const noexcept {
    return (static_cast<std::size_t>(p.first) << 32) ^
           static_cast<std::size_t>(p.second);
  }
};

struct FirstHopState {
  int v;                    // node index (1-based)
  int i;                    // ancilla index
  int l;                    // neighbor index (1-based)
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
  Vertex getVertex(int id) const;

  // Calculate degrees and second-order degrees
  void calculateDegrees();

  // Build adjacency matrix from BGL graph
  void buildAdjacencyMatrix();

  void buildAdjacencyList();

  void buildPaddedAdjList(int padSize);

  void buildFirstHopStateMap(const std::vector<FirstHopState> &firstHopStates);
};

// Create a 2x2 matrix
Matrix makeMatrix(dtype a, dtype b, dtype c, dtype d);

// Kronecker product of two vectors
std::vector<dtype> kron(const std::vector<dtype> &a,
                        const std::vector<dtype> &b);

Matrix hadamard(int k);

Matrix Wval(dtype val);

Matrix Wtheta(dtype theta);

inline int get_r(const GraphData &data, int v_idx, int l_idx);

const std::vector<int> &get_r_all(const GraphData &data, int v_idx);

Matrix get_feature_rotation(const GraphData &data,
                            const std::vector<dtype> &featuresNorm, int v,
                            int l);


std::vector<dtype>
generate_node_state(int v, int c, int i, int s,
                    const std::vector<FirstHopState> &firstHopStates,
                    const GraphData &data, const std::vector<int> &nodeDegrees,
                    const std::vector<int> &nodeDegreesC1); 
