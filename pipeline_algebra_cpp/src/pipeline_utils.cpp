#include "pipeline_utils.hpp"

// Get vertex by ID
Vertex GraphData::getVertex(int id) const {
  VertexIterator vi, vi_end;
  for (boost::tie(vi, vi_end) = boost::vertices(graph); vi != vi_end; ++vi) {
    if (graph[*vi].id == id) {
      return *vi;
    }
  }
  throw std::runtime_error("Vertex not found");
}

// Calculate degrees and second-order degrees
void GraphData::calculateDegrees() {
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
void GraphData::buildAdjacencyMatrix() {
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

void GraphData::buildAdjacencyList() {
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

void GraphData::buildPaddedAdjList(int padSize) {
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
void GraphData::buildFirstHopStateMap(
    const std::vector<FirstHopState> &firstHopStates) {
  firstHopStateMap.clear();
  firstHopStateMap.reserve(firstHopStates.size());

  for (const auto &s : firstHopStates) {
    firstHopStateMap[{s.v, s.i}].push_back(s.state);
  }
}

// Create a 2x2 matrix
Matrix makeMatrix(dtype a, dtype b, dtype c, dtype d) {
  return {{a, b}, {c, d}};
}

// Kronecker product of two vectors
std::vector<dtype> kron(const std::vector<dtype> &a,
                        const std::vector<dtype> &b) {
  std::vector<dtype> result;
  result.reserve(a.size() * b.size());
  for (dtype x : a) {
    for (dtype y : b) {
      result.push_back(x * y);
    }
  }
  return result;
}

Matrix hadamard(int k) {
  if (k == 1) {
    dtype norm = 1.0 / std::sqrt(2.0);
    return makeMatrix(norm, norm, norm, -norm);
  }
  Matrix H = hadamard(k - 1);
  int size = H.size();
  int newSize = size * 2;
  Matrix result(newSize, std::vector<dtype>(newSize, 0.0));

  dtype norm = 1.0 / std::sqrt(2.0);
  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      result[i][j] = norm * H[i][j];
      result[i][j + size] = norm * H[i][j];
      result[i + size][j] = norm * H[i][j];
      result[i + size][j + size] = -norm * H[i][j];
    }
  }
  return result;
}

Matrix Wval(dtype val) {
  dtype sqrt_term = std::sqrt(1.0 - val * val);
  return {{val, -sqrt_term}, {sqrt_term, val}};
}

Matrix Wtheta(dtype theta) {
  return makeMatrix(std::cos(theta / 2), -std::sin(theta / 2),
                    std::sin(theta / 2), std::cos(theta / 2));
}

inline int get_r(const GraphData &data, int v_idx, int l_idx) {
  // if (v_idx < 0 || v_idx >= static_cast<int>(data.adjacencyList.size())) {
  //   return data.nrNodes; // dummy
  // }
  if (l_idx < static_cast<int>(data.adjacencyList[v_idx].size())) {
    return data.adjacencyList[v_idx][l_idx];
  }
  return data.nrNodes; // dummy
}

const std::vector<int> &get_r_all(const GraphData &data, int v_idx) {
  return data.paddedAdjList[v_idx]; // already padded with nrNodes
}

Matrix get_feature_rotation(const GraphData &data,
                            const std::vector<dtype> &featuresNorm, int v,
                            int l) {
  int v_idx = v;
  int l_idx = l;

  if (0 <= v_idx && v_idx < data.nrNodes && 0 <= l_idx && l_idx < data.maxD) {
    int u = get_r(data, v_idx, l_idx);
    if (u < data.nrNodes) {
      dtype featVal = featuresNorm[u]; // |featVal| <= 1
      return Wval(featVal);
    }
  }
  return Wval(0.0);
}


std::vector<dtype>
generate_node_state(int v, int c, int i, int s,
                    const std::vector<FirstHopState> &firstHopStates,
                    const GraphData &data, const std::vector<int> &nodeDegrees,
                    const std::vector<int> &nodeDegreesC1) {
  // std::cout << "[DEBUG] Entering generate_node_state(v=" << v << ", c=" << c
  // << ", i=" << i << ", s=" << s << ")\n";

  if (c == 1) {
    // std::cout << "[DEBUG] Case: First-hop state (c == 1)\n";
    std::vector<std::vector<dtype>> states_v_i;

    // Find all states for node v with index i
    // std::cout << "[DEBUG] Scanning firstHopStates for (v=" << v << ", i=" <<
    // i
    // << ")\n";
    // for (const auto &state : firstHopStates) {
    //   if (state.v == v && state.i == i) {
    //     states_v_i.push_back(state.state);
    //   }
    // }

    auto it = data.firstHopStateMap.find({v, i});
    if (it != data.firstHopStateMap.end()) {
      for (const auto &st : it->second) {

        states_v_i.push_back(st);
      }
    }
    // std::cout << "[DEBUG] Found " << states_v_i.size()
    //           << " matching first-hop states\n";

    if (states_v_i.empty()) {
      std::cout << "[DEBUG] No states found, returning empty vector\n";
      return std::vector<dtype>();
    }

    int stateDim = 16;
    // std::cout << "[DEBUG] stateDim = " << stateDim << "\n";

    dtype angle = 2.0 * std::acos(1.0 / nodeDegrees[v]);
    // std::cout << "[DEBUG] degree angle = " << angle << "\n";
    Matrix degRotMat = Wtheta(angle);
    std::vector<dtype> degRotVec(2);
    degRotVec[0] = degRotMat[0][0] * 1.0 + degRotMat[0][1] * 0.0;
    degRotVec[1] = degRotMat[1][0] * 1.0 + degRotMat[1][1] * 0.0;

    std::vector<dtype> finalState(s * stateDim * 2, 0.0);
    // std::cout << "[DEBUG] Allocated finalState of size " << finalState.size()
    // << "\n";

    for (int idx_l = 0; idx_l < static_cast<int>(states_v_i.size()); idx_l++) {
      if (idx_l % 100 == 0) { // print progress every 100
        // std::cout << "[DEBUG] Processing state index " << idx_l << "/"
        // << states_v_i.size() << "\n";
      }
      for (int idx_bit = 0;
           idx_bit < static_cast<int>(states_v_i[idx_l].size()); idx_bit++) {
        dtype val_h = states_v_i[idx_l][idx_bit] / std::sqrt(s);
        int final_idx = (idx_l * stateDim + idx_bit) * 2;

        if (final_idx + 1 < static_cast<int>(finalState.size())) {
          finalState[final_idx] = val_h * degRotVec[0];
          finalState[final_idx + 1] = val_h * degRotVec[1];
        }
      }
    }

    // Normalize
    // std::cout << "[DEBUG] Starting normalization\n";
    // dtype norm = 0.0;
    // for (dtype val : finalState) {
    //   norm += val * val;
    // }
    // norm = std::sqrt(norm);
    // // std::cout << "[DEBUG] Computed norm = " << norm << "\n";
    //
    // if (norm > 1e-10) {
    //   for (dtype &val : finalState) {
    //     val /= norm;
    //   }
    //   // std::cout << "[DEBUG] Normalized finalState\n";
    // }
    //
    // // std::cout << "[DEBUG] Returning finalState of size " <<
    // finalState.size()
    // //           << "\n";
    return finalState;

  } else {
    // std::cout << "[DEBUG] Case: Second-hop state (c == 2)\n";
    std::vector<std::vector<dtype>> state_list;

    std::vector<int> neighbors = get_r_all(data, v); // already length s, padded

    for (int l0 = 0; l0 < s; l0++) {
      int u0 = neighbors[l0]; // either neighbor id or nrNodes dummy

      if (l0 % 50 == 0) {
        // std::cout << "[DEBUG] Processing neighbor l0=" << l0 << " (u0=" << u0
        //           << ")\n";
      }

      // for (const auto &state : firstHopStates) {
      //   if (state.v == u0 && state.i == i) {
      //     state_list.push_back(state.state);
      //   }
      // }
      auto it = data.firstHopStateMap.find({u0, i});
      if (it != data.firstHopStateMap.end()) {
        for (const auto &st : it->second) {

          state_list.push_back(st);
        }
      }

      if (l0 % 50 == 0) {
        // std::cout << "[DEBUG] Finished Processing neighbor l0=" << l0
        //           << " (u0=" << u0 << ")\n";
      }
    }
    // std::cout << "[DEBUG] Collected " << state_list.size()
    //           << " neighbor states\n";

    if (state_list.empty()) {
      // std::cout << "[DEBUG] No neighbor states found, returning empty
      // vector\n";
      return std::vector<dtype>();
    }

    int stateDim = static_cast<int>(state_list[0].size());
    // std::cout << "[DEBUG] stateDim = " << stateDim << "\n";

    dtype angle = 2.0 * std::acos(1.0 / nodeDegreesC1[v]);
    // std::cout << "[DEBUG] degree angle (2nd hop) = " << angle << "\n";
    Matrix degRotMat = Wtheta(angle);
    std::vector<dtype> degRotVec(2);
    degRotVec[0] = degRotMat[0][0] * 1.0 + degRotMat[0][1] * 0.0;
    degRotVec[1] = degRotMat[1][0] * 1.0 + degRotMat[1][1] * 0.0;

    int final_len = s * s * stateDim * 2;
    std::vector<dtype> finalState(final_len, 0.0);
    // std::cout << "[DEBUG] Allocated finalState of size " << final_len <<
    // "\n";

    for (int idx_l = 0; idx_l < static_cast<int>(state_list.size()); idx_l++) {
      if (idx_l % 100 == 0) {
        // std::cout << "[DEBUG] Processing neighbor state index " << idx_l <<
        // "/"
        // << state_list.size() << "\n";
      }
      for (int idx_bit = 0;
           idx_bit < static_cast<int>(state_list[idx_l].size()); idx_bit++) {
        dtype val_h = state_list[idx_l][idx_bit] / s;
        int final_idx = (idx_l * stateDim + idx_bit) * 2;

        if (final_idx + 1 < final_len) {
          finalState[final_idx] = val_h * degRotVec[0];
          finalState[final_idx + 1] = val_h * degRotVec[1];
        }
      }
    }

    // Normalize
    // std::cout << "[DEBUG] Starting normalization\n";
    // dtype norm = 0.0;
    // for (dtype val : finalState) {
    //   norm += val * val;
    // }
    // norm = std::sqrt(norm);
    // // std::cout << "[DEBUG] Computed norm = " << norm << "\n";
    //
    // if (norm > 1e-10) {
    //   for (dtype &val : finalState) {
    //     val /= norm;
    //   }
    //   // std::cout << "[DEBUG] Normalized finalState\n";
    // }

    // std::cout << "[DEBUG] Returning finalState of size " << finalState.size()
    //           << "\n";
    return finalState;
  }
}
