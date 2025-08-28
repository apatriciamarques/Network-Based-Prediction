#include <algorithm>
#include <cblas.h> // Link with -lblas or Intel MKL
#include <cmath>
#include <cnpy.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <map>
#include <numeric>
#include <omp.h>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "input.hpp"
#include "pipeline_utils.hpp"

// Boost Graph Library includes
//
//
//
//

// Fast integer power function (avoid std::pow in hot loop)
inline dtype fast_pow(dtype base, int exp) {
  if (exp == 0)
    return 1.0;
  if (exp == 1)
    return base;
  if (exp == 2)
    return base * base;
  if (exp == 4) {
    dtype sq = base * base;
    return sq * sq;
  }
  if (exp == 6) {
    dtype sq = base * base;
    return sq * sq * sq;
  }
  if (exp == 8) {
    dtype sq = base * base;
    dtype qu = sq * sq;
    return qu * qu;
  }

  // Binary exponentiation for larger powers
  dtype result = 1.0;
  dtype current_power = base;
  while (exp > 0) {
    if (exp & 1)
      result *= current_power;
    current_power *= current_power;
    exp >>= 1;
  }
  return result;
}

void save_matrix_npy(const Matrix &mat, const std::string &filename) {
  if (mat.empty() || mat[0].empty()) {
    throw std::runtime_error("Matrix is empty!");
  }

  size_t rows = mat.size();
  size_t cols = mat[0].size();

  // Flatten row-major
  std::vector<dtype> flat;
  flat.reserve(rows * cols);
  for (const auto &row : mat) {
    if (row.size() != cols)
      throw std::runtime_error("Inconsistent row size");
    flat.insert(flat.end(), row.begin(), row.end());
  }

  // Save as (rows, cols) shaped array
  cnpy::npy_save(filename, flat.data(), {rows, cols}, "w");
}

void printMatrix(const Matrix &M, const std::string &name = "") {
  if (!name.empty()) {
    std::cout << name << " =\n";
  }
  for (const auto &row : M) {
    for (dtype val : row) {
      std::cout << std::setw(10) << std::fixed << std::setprecision(6) << val
                << " ";
    }
    std::cout << "\n";
  }
  std::cout << std::endl;
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

inline const std::vector<int> &get_r_all(const GraphData &data, int v_idx) {
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

// Structure for classification results
struct ClassificationResult {
  int nodeId;
  int trueLabel;
  int predictedLabel;
  bool correct;
  dtype expectationValue;
};

// Structure for metrics
struct ClassificationMetrics {
  dtype accuracy;
  dtype precision;
  dtype recall;
  dtype f1Score;
};

// Check node function - equivalent to Python check_node
std::vector<int> check_node(const GraphData &data, int nodeCheck) {
  std::cout << "\nNode: " << nodeCheck << std::endl;

  try {
    Vertex v = data.getVertex(nodeCheck);
    std::vector<int> neighbors;
    std::vector<int> neighborsDegrees;

    // Get neighbors
    AdjacencyIterator ai, ai_end;
    for (boost::tie(ai, ai_end) = boost::adjacent_vertices(v, data.graph);
         ai != ai_end; ++ai) {
      int neighborId = data.graph[*ai].id;
      neighbors.push_back(neighborId);
      neighborsDegrees.push_back(data.graph[*ai].degree);
    }

    std::cout << "  Degree: " << data.graph[v].degree << std::endl;
    std::cout << "  Neighbors: [";
    for (size_t i = 0; i < neighbors.size(); i++) {
      std::cout << neighbors[i];
      if (i < neighbors.size() - 1)
        std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    std::cout << "  Neighbors' Degrees: [";
    for (size_t i = 0; i < neighborsDegrees.size(); i++) {
      std::cout << neighborsDegrees[i];
      if (i < neighborsDegrees.size() - 1)
        std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    std::cout << "  C1 Degree (sum of neighbors' degrees): "
              << data.graph[v].secondOrderDegree << std::endl;

    return neighbors;
  } catch (const std::exception &e) {
    std::cerr << "Error in check_node: " << e.what() << std::endl;
    return std::vector<int>();
  }
}

// Compute kernel matrix (simplified version - placeholder for quantum kernel)
std::vector<std::vector<dtype>>
compute_kernel_matrix(const std::vector<std::vector<dtype>> &trainVecs,
                      int nPower = 1) {
  int n = trainVecs.size();
  std::vector<std::vector<dtype>> kernelMatrix(n, std::vector<dtype>(n, 0.0));

  std::cout << "Computing kernel matrix (" << n << "x" << n << ")..."
            << std::endl;

  // Simple dot product kernel for now (placeholder)
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      dtype dotProduct = 0.0;
      if (trainVecs[i].size() == trainVecs[j].size()) {
        for (size_t k = 0; k < trainVecs[i].size(); k++) {
          dotProduct += trainVecs[i][k] * trainVecs[j][k];
        }
      }
      kernelMatrix[i][j] = std::pow(dotProduct, nPower);
    }
    if ((i + 1) % 100 == 0) {
      std::cout << "  Processed " << (i + 1) << "/" << n << " rows"
                << std::endl;
    }
  }

  std::cout << "Kernel matrix computation finished." << std::endl;
  return kernelMatrix;
}

// Plot kernel matrix (text-based visualization)
void plot_kernel_matrix(const std::vector<std::vector<dtype>> &kernelMatrix,
                        bool show = true, int maxDisplay = 200) {
  if (!show || kernelMatrix.empty())
    return;

  std::cout << "\n=== Kernel Matrix Visualization ===" << std::endl;
  int n = std::min(maxDisplay, (int)kernelMatrix.size());

  // Print header
  std::cout << "     ";
  for (int j = 0; j < n; j++) {
    std::cout << std::setw(8) << j;
  }
  std::cout << std::endl;

  // Print matrix
  for (int i = 0; i < n; i++) {
    std::cout << std::setw(4) << i << ":";
    for (int j = 0; j < n; j++) {
      std::cout << std::setw(8) << std::fixed << std::setprecision(3)
                << kernelMatrix[i][j];
    }
    std::cout << std::endl;
  }

  // Print statistics
  dtype minVal = kernelMatrix[0][0], maxVal = kernelMatrix[0][0], sumVal = 0.0;
  int totalElements = 0;

  for (const auto &row : kernelMatrix) {
    for (dtype val : row) {
      minVal = std::min(minVal, val);
      maxVal = std::max(maxVal, val);
      sumVal += val;
      totalElements++;
    }
  }

  std::cout << "\nKernel Matrix Stats:" << std::endl;
  std::cout << "  Min: " << std::fixed << std::setprecision(6) << minVal
            << std::endl;
  std::cout << "  Max: " << std::fixed << std::setprecision(6) << maxVal
            << std::endl;
  std::cout << "  Mean: " << std::fixed << std::setprecision(6)
            << (sumVal / totalElements) << std::endl;
}

// Compute expectation values (simplified version)
std::vector<dtype>
compute_expectation_values(const std::vector<std::vector<dtype>> &kernelMatrix,
                           const std::vector<int> &classLabels,
                           const std::vector<dtype> &features) {
  int n = kernelMatrix.size();
  std::vector<dtype> expectationValues(n, 0.0);

  std::cout << "Computing expectation values..." << std::endl;

  for (int i = 0; i < n; i++) {
    dtype sum = 0.0;
    for (int j = 0; j < n; j++) {
      // Simple weighted sum using kernel values, class labels, and features
      dtype weight = kernelMatrix[i][j];
      if (j < (int)classLabels.size())
        weight *= std::pow(-1, classLabels[j]);
      // if (j < (int)features.size())
      //   weight *= features[j];
      sum += weight;
    }
    expectationValues[i] = sum / n; // Normalize
  }

  std::cout << "Expectation values computed." << std::endl;
  return expectationValues;
}

// Predict labels based on expectation values
std::vector<int> predict_labels(const std::vector<dtype> &expectationValues,
                                dtype threshold = 0.0) {
  std::vector<int> predictedLabels;

  for (dtype val : expectationValues) {
    predictedLabels.push_back(val > threshold ? 0 : 1);
  }

  return predictedLabels;
}

// Evaluate predictions and compute metrics
ClassificationMetrics evaluate_predictions(
    const std::vector<int> &trueLabels, const std::vector<int> &predictedLabels,
    const std::vector<dtype> &expectationValues,
    std::vector<ClassificationResult> &results, bool show = true) {

  std::cout << "Evaluating predictions..." << std::endl;

  // Create results
  results.clear();
  int n = std::min({(int)trueLabels.size(), (int)predictedLabels.size(),
                    (int)expectationValues.size()});

  for (int i = 0; i < n; i++) {
    ClassificationResult result;
    result.nodeId = i;
    result.trueLabel = trueLabels[i];
    result.predictedLabel = predictedLabels[i];
    result.correct = (predictedLabels[i] == trueLabels[i]);
    result.expectationValue = expectationValues[i];
    results.push_back(result);
  }

  // Compute metrics
  int tp = 0, fp = 0, fn = 0, tn = 0;
  int correct = 0;

  for (const auto &r : results) {
    if (r.correct)
      correct++;

    if (r.predictedLabel == 1 && r.trueLabel == 1)
      tp++;
    else if (r.predictedLabel == 1 && r.trueLabel == 0)
      fp++;
    else if (r.predictedLabel == 0 && r.trueLabel == 1)
      fn++;
    else if (r.predictedLabel == 0 && r.trueLabel == 0)
      tn++;
  }

  ClassificationMetrics metrics;
  metrics.accuracy = (dtype)correct / results.size();
  metrics.precision = (tp + fp > 0) ? (dtype)tp / (tp + fp) : 0.0;
  metrics.recall = (tp + fn > 0) ? (dtype)tp / (tp + fn) : 0.0;
  metrics.f1Score = (metrics.precision + metrics.recall > 0)
                        ? 2 * metrics.precision * metrics.recall /
                              (metrics.precision + metrics.recall)
                        : 0.0;

  if (show) {
    std::cout << "\n=== Classification Results (first 20) ===" << std::endl;
    std::cout << "Node | True | Pred | Correct | Expectation" << std::endl;
    std::cout << "-----|------|------|---------|------------" << std::endl;

    int displayCount = std::min(128, (int)results.size());
    for (int i = 0; i < displayCount; i++) {
      const auto &r = results[i];
      std::cout << std::setw(4) << r.nodeId << " | " << std::setw(4)
                << r.trueLabel << " | " << std::setw(4) << r.predictedLabel
                << " | " << std::setw(7) << (r.correct ? "Yes" : "No") << " | "
                << std::setw(11) << std::fixed << std::setprecision(6)
                << r.expectationValue << std::endl;
    }

    std::cout << "\nConfusion Matrix:" << std::endl;
    std::cout << "           Predicted" << std::endl;
    std::cout << "         0    1" << std::endl;
    std::cout << "Actual 0 " << std::setw(4) << tn << " " << std::setw(4) << fp
              << std::endl;
    std::cout << "       1 " << std::setw(4) << fn << " " << std::setw(4) << tp
              << std::endl;
  }

  return metrics;
}

// Save/Load kernel matrix functions
void save_kernel_matrix(const std::vector<std::vector<dtype>> &kernelMatrix,
                        const std::string &filename) {
  std::ofstream file(filename, std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "Error: Could not save kernel matrix to " << filename
              << std::endl;
    return;
  }

  int n = kernelMatrix.size();
  file.write(reinterpret_cast<const char *>(&n), sizeof(n));

  for (const auto &row : kernelMatrix) {
    for (dtype val : row) {
      file.write(reinterpret_cast<const char *>(&val), sizeof(val));
    }
  }

  file.close();
  std::cout << "Kernel matrix saved to " << filename << std::endl;
}

std::vector<std::vector<dtype>>
load_kernel_matrix(const std::string &filename) {
  std::ifstream file(filename, std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "Error: Could not load kernel matrix from " << filename
              << std::endl;
    return std::vector<std::vector<dtype>>();
  }

  int n;
  file.read(reinterpret_cast<char *>(&n), sizeof(n));

  std::vector<std::vector<dtype>> kernelMatrix(n, std::vector<dtype>(n));

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      file.read(reinterpret_cast<char *>(&kernelMatrix[i][j]), sizeof(dtype));
    }
  }

  file.close();
  std::cout << "Kernel matrix loaded from " << filename << std::endl;
  return kernelMatrix;
}

// Check if file exists
bool file_exists(const std::string &filename) {
  std::ifstream file(filename);
  return file.good();
}

std::vector<dtype> oracleX_kronecker(const std::vector<dtype> &feat_vec, int i,
                                     int q = 4) {
  int final_len = 1 << q; // 2^q
  std::vector<dtype> result(final_len, 0.0);

  for (int idx = 0; idx < final_len; idx++) {
    dtype val = 1.0;
    for (int slot = 0; slot < q; slot++) {
      int bit = (idx >> (q - 1 - slot)) & 1;
      if (slot <= i) { // i here is -1 than in python
        val *= feat_vec[bit];
      } else {
        val *= (bit == 0 ? 1.0 : 0.0);
      }
    }
    result[idx] = val;
  }
  return result;
}

void check_feat_rotation(const GraphData &data,
                         const std::vector<dtype> &featuresNorm, int nodeCheck,
                         const std::vector<int> &neighborsCheck) {
  std::cout << "Node: " << nodeCheck << "\n";
  std::cout << "Neighbor: " << neighborsCheck[0] << "\n";
  std::cout << "Neighbor's FeatureNorm: " << featuresNorm[neighborsCheck[0]]
            << "\n";

  Matrix rot = get_feature_rotation(data, featuresNorm, nodeCheck + 1, 1);
  std::cout << "Rotation matrix:\n";
  for (auto &row : rot) {
    for (dtype val : row)
      std::cout << val << " ";
    std::cout << "\n";
  }
}

std::vector<dtype> normalizeFeatures(const std::vector<dtype> &raw) {
  if (raw.empty())
    return {};
  dtype minVal = *std::min_element(raw.begin(), raw.end());
  dtype maxVal = *std::max_element(raw.begin(), raw.end());

  std::vector<dtype> normed;
  normed.reserve(raw.size());

  if (maxVal == minVal) {
    // Avoid divide by zero: all features same → map to 0
    normed.assign(raw.size(), 0.0);
    return normed;
  }

  for (dtype x : raw) {
    // dtype scaled = 2.0 * (x - minVal) / (maxVal - minVal) - 1.0;
    dtype scaled = (x - minVal) / (maxVal - minVal);
    normed.push_back(scaled);
  }
  return normed;
}

std::vector<dtype> extractFeatures(const GraphData &data) {
  std::vector<dtype> features;
  VertexIterator vi, vi_end;
  for (boost::tie(vi, vi_end) = boost::vertices(data.graph); vi != vi_end;
       ++vi) {
    features.push_back(data.graph[*vi].feature);
  }
  return features;
}

std::vector<int> extractDegrees(const GraphData &data) {
  std::vector<int> degrees;
  VertexIterator vi, vi_end;
  for (boost::tie(vi, vi_end) = boost::vertices(data.graph); vi != vi_end;
       ++vi) {
    degrees.push_back(data.graph[*vi].degree);
  }
  return degrees;
}

std::vector<int> extractC1Degrees(const GraphData &data) {
  std::vector<int> degrees;
  VertexIterator vi, vi_end;
  for (boost::tie(vi, vi_end) = boost::vertices(data.graph); vi != vi_end;
       ++vi) {
    degrees.push_back(data.graph[*vi].secondOrderDegree);
  }
  return degrees;
}

std::vector<dtype> getNormalizedFeatures(const GraphData &data) {
  auto raw = extractFeatures(data);
  for (auto &feat : raw) {
    std::cout << "feature: " << std::fixed << std::setprecision(7) << feat
              << "\n";
  }

  return raw;
}

std::vector<FirstHopState>
get_firstHopStates(const GraphData &data,
                   const std::vector<dtype> &featuresNorm, int s) {
  std::vector<FirstHopState> firstHopStates;

  for (int v = 0; v <= data.nrNodes; v++) {
    std::cout << "firstHopStates: v: " << v << "/" << (data.nrNodes) << "\n";

    for (int l_idx = 0; l_idx < s; l_idx++) {
      // Base feature vector: Wval rotation applied to |0> = [1,0]^T
      Matrix rot = get_feature_rotation(data, featuresNorm, v, l_idx);
      // printMatrix(rot, "Rotation Matrix");

      // multiply 2x2 matrix by [1,0]^T
      std::vector<dtype> feat_vec(2, 0.0);
      feat_vec[0] = rot[0][0] * 1.0 + rot[0][1] * 0.0;
      feat_vec[1] = rot[1][0] * 1.0 + rot[1][1] * 0.0;

      // int u_val = get_r(data, v, l_idx);
      // std::cout << "u_val " << u_val << std::endl;

      for (int i = 0; i <= 3; i++) {
        std::vector<dtype> combined_vec = oracleX_kronecker(feat_vec, i);
        // std::cout << "combined_vec (v=" << v  << ", l=" << l_idx
        //           << ", i=" << i << "): [";
        //
        // for (size_t k = 0; k < combined_vec.size(); k++) {
        //   std::cout << std::fixed << std::setprecision(7) << combined_vec[k];
        //   if (k + 1 < combined_vec.size())
        //     std::cout << ", ";
        // }
        // std::cout << "]\n";
        firstHopStates.push_back({v, // 1-based node index
                                  i, l_idx, combined_vec});
      }
    }
  }
  return firstHopStates;
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

Matrix compute_kernel_block_batched(
    int c, int i, int s, const std::vector<FirstHopState> &firstHopStates,
    const std::vector<int> &node_ids, const GraphData &data,
    const std::vector<int> &nodeDegrees, const std::vector<int> &nodeDegreesC1,
    int nPower = 1, int batch_size = -1) {

  std::cout << "Compute_kernel_block_batched: c=" << c << ", i=" << i
            << std::endl;

  // Adaptive batch size
  if (batch_size <= 0) {
    batch_size = (c == 1 ? 7000 : 250);
  }

  int n_nodes = static_cast<int>(node_ids.size());
  Matrix K_block(n_nodes, std::vector<dtype>(n_nodes, 0.0));

  std::unordered_set<int> missing_nodes; // Track missing nodes

  // Process in batches over i
  for (int start_i = 0; start_i < n_nodes; start_i += batch_size) {
    int end_i = std::min(start_i + batch_size, n_nodes);
    std::vector<int> batch_i_ids(node_ids.begin() + start_i,
                                 node_ids.begin() + end_i);

    // Generate vectors for batch i
    std::unordered_map<int, std::vector<dtype>> batch_i_vectors;
    for (int v : batch_i_ids) {
      // std::cout << "    Node i: " << v << "/" << (n_nodes - 1) << std::endl;
      auto vec = generate_node_state(v, c, i, s, firstHopStates, data,
                                     nodeDegrees, nodeDegreesC1);
      if (vec.empty()) {
        if (missing_nodes.find(v) == missing_nodes.end()) {
          std::cout << "[DEBUG] Node " << v << " has no states for c=" << c
                    << ", i=" << i << std::endl;
          missing_nodes.insert(v);
        }
      } else {
        batch_i_vectors[v] = std::move(vec);
      }
    }

    // Process in batches over j
    for (int start_j = start_i; start_j < n_nodes; start_j += batch_size) {
      int end_j = std::min(start_j + batch_size, n_nodes);
      std::vector<int> batch_j_ids(node_ids.begin() + start_j,
                                   node_ids.begin() + end_j);

      std::cout << "Batch: row= " << start_i
                << ", column= " << start_j 
                << std::endl;
      // Generate vectors for batch j
      std::unordered_map<int, std::vector<dtype>> batch_j_vectors;
      for (int v : batch_j_ids) {
        // std::cout << "    Node j: " << v << "/" << (n_nodes - 1) <<
        // std::endl;
        auto vec = generate_node_state(v, c, i, s, firstHopStates, data,
                                       nodeDegrees, nodeDegreesC1);
        if (vec.empty()) {
          if (missing_nodes.find(v) == missing_nodes.end()) {
            // std::cout << "[DEBUG] Node " << v << " has no states for c=" << c
            //           << ", i=" << i << std::endl;
            missing_nodes.insert(v);
          }
        } else {
          batch_j_vectors[v] = std::move(vec);
        }
      }

      // std::cout << "Computing Dot Products: c=" << c << ", i=" << i
      //           << std::endl;

      // Pre-compute expensive operations outside the parallel region
      std::vector<dtype> sqrt_cache;
      sqrt_cache.reserve(batch_i_ids.size());
      for (int idx_i = 0; idx_i < (int)batch_i_ids.size(); idx_i++) {
        int v_i = batch_i_ids[idx_i];
        auto it_i = batch_i_vectors.find(v_i);
        if (it_i != batch_i_vectors.end()) {
          sqrt_cache.push_back(1.0 / std::sqrt((dtype)it_i->second.size()));
        } else {
          sqrt_cache.push_back(0.0); // Invalid marker
        }
      }

      // Compute kernel values between batches
      // Main parallel loop
#pragma omp parallel for schedule(dynamic, 1) collapse(1)
      for (int idx_i = 0; idx_i < (int)batch_i_ids.size(); idx_i++) {
        // Skip if pre-computation marked as invalid
        if (sqrt_cache[idx_i] == 0.0)
          continue;

        int v_i = batch_i_ids[idx_i];
        auto it_i = batch_i_vectors.find(v_i);
        if (it_i == batch_i_vectors.end())
          continue;

        const auto &vec_i = it_i->second;
        dtype inv_sqrt_i = sqrt_cache[idx_i];

        // Inner loop - not parallelized to avoid race conditions
        for (int idx_j = 0; idx_j < (int)batch_j_ids.size(); idx_j++) {
          int v_j = batch_j_ids[idx_j];

          // upper-triangle only
          if (start_i == start_j && idx_j < idx_i)
            continue;

          auto it_j = batch_j_vectors.find(v_j);
          if (it_j == batch_j_vectors.end())
            continue;

          const auto &vec_j = it_j->second;

          // Use minimum size for safety
          size_t min_size = std::min(vec_i.size(), vec_j.size());

          // BLAS dot product (thread-safe)
          dtype dot_val =
              cblas_sdot(min_size, vec_i.data(), 1, vec_j.data(), 1);

          // Apply normalization
          dot_val *= inv_sqrt_i;

          // Fast power calculation
          dtype kernel_val = fast_pow(dot_val, 2 * nPower);

          int global_i = start_i + idx_i;
          int global_j = start_j + idx_j;

          // Write to matrix (no race condition since each thread writes unique
          // indices)
          K_block[global_i][global_j] = kernel_val;
          K_block[global_j][global_i] = kernel_val; // symmetric
        }
      }
    }
  }

  return K_block;
}

Matrix compute_kernel_block(int c, int i, int s,
                            const std::vector<FirstHopState> &firstHopStates,
                            const std::vector<int> &node_ids,
                            const GraphData &data,
                            const std::vector<int> &nodeDegrees,
                            const std::vector<int> &nodeDegreesC1,
                            int nPower = 1) {

  std::cout << "compute_kernel_block: c: " << c << ", i: " << i << std::endl;

  int batch_size = (c == 1 ? 500 : 10);

  int n_nodes = static_cast<int>(node_ids.size());
  Matrix K_block(n_nodes, std::vector<dtype>(n_nodes, 0.0));

  // Precompute all node vectors for this block
  std::vector<std::vector<dtype>> node_vectors(n_nodes);
  std::vector<bool> has_vector(n_nodes, false);

  for (int idx = 0; idx < n_nodes; idx++) {
    int v = node_ids[idx];
    std::cout << "Precompute node: " << v << "\n";
    std::vector<dtype> vec = generate_node_state(
        v, c, i, s, firstHopStates, data, nodeDegrees, nodeDegreesC1);
    if (!vec.empty()) {
      node_vectors[idx] = vec;
      has_vector[idx] = true;
    }
  }

  // Compute partial kernel
  for (int idx_i = 0; idx_i < n_nodes; idx_i++) {

    if (!has_vector[idx_i])
      continue;

    for (int idx_j = idx_i; idx_j < n_nodes; idx_j++) {
      if (!has_vector[idx_j])
        continue;

      const auto &vec_i = node_vectors[idx_i];
      const auto &vec_j = node_vectors[idx_j];

      std::cout << "Compute partial kernel idx_i: " << idx_i
                << " idx_j: " << idx_j << "\n";

      // Compute dot product
      dtype dot_val = 0.0;
      for (size_t k = 0; k < vec_i.size() && k < vec_j.size(); k++) {
        dot_val += vec_i[k] * vec_j[k];
      }
      dot_val /= std::sqrt(vec_i.size());

      // Apply power
      dtype kernel_val = std::pow(dot_val, 2 * nPower);

      K_block[idx_i][idx_j] = kernel_val;
      K_block[idx_j][idx_i] = kernel_val; // Symmetric
    }
  }

  return K_block;
}

std::pair<Matrix, std::vector<int>> compute_kernel_matrix_blockwise(
    int s, const std::vector<FirstHopState> &firstHopStates, int nrNodes,
    const GraphData &data, const std::vector<int> &nodeDegrees,
    const std::vector<int> &nodeDegreesC1, int nPower = 1) {

  std::vector<int> node_ids;
  for (int i = 0; i < nrNodes; i++) {
    node_ids.push_back(i); // 0-based indexing
  }

  Matrix K_total(nrNodes, std::vector<dtype>(nrNodes, 0.0));

  // Process all (c,i) blocks
  std::vector<std::pair<int, int>> blocks = {{1, 0}, {1, 1}, {1, 2}, {1, 3},
                                             {2, 0}, {2, 1}, {2, 2}, {2, 3}};

  for (const auto &block : blocks) {
    int c = block.first;
    int i = block.second;

    Matrix K_block =
        compute_kernel_block_batched(c, i, s, firstHopStates, node_ids, data,
                                     nodeDegrees, nodeDegreesC1, nPower);

    // Add to total kernel matrix
    for (int row = 0; row < nrNodes; row++) {
      for (int col = 0; col < nrNodes; col++) {
        K_total[row][col] += K_block[row][col];
      }
    }
  }

  return std::make_pair(K_total, node_ids);
}

int main() {
  // Configuration
  std::string graph_type =
      "PPI";         // Change to "PPI" for protein-protein interaction data
  int nodeCheck = 0; // 0-based
  bool show = true;
  std::string KERNEL_FILE = "output/" + graph_type + "_kernel_matrix.bin";

  std::cout
      << "Computing kernel matrix from scratch using Boost Graph Library..."
      << std::endl;

  // Load graph data
  GraphData data = input_data(graph_type, 7, 2, show);

  if (boost::num_vertices(data.graph) == 0) {
    std::cerr << "Error: Failed to load graph data." << std::endl;
    return 1;
  }

  data.buildAdjacencyList();       // for O(1) neighbor access
  data.buildPaddedAdjList(data.s); // for O(1) neighbor access

  // Check node
  std::vector<int> neighborsCheck = check_node(data, nodeCheck);

  // Print basic results
  printAdjacencyMatrix(data, 15);
  printGraphProperties(data, 15);
  printGraphStatistics(data);

  // Example normalized features
  // std::vector<dtype> featuresNorm = {0.8, 0.5, 0.2, -0.6, 0.9};
  auto featuresNorm = getNormalizedFeatures(data);

  // for (int i = 0; i < 4; i++) {
  //
  //   auto neighbor = get_r(data, 0, i);
  //   std::cout << "Neighbor: " << neighbor << std::endl;
  //   std::cout << "Feature norm Node 0 Neighbor " << neighbor << ": "
  //             << featuresNorm[neighbor] << std::endl;
  // }

  // s = number of neighbor slots to consider
  int s = data.maxD;

  std::vector<int> nodeDegrees = extractDegrees(data);
  std::vector<int> nodeDegreesC1 = extractC1Degrees(data);

  auto firstHopStates = get_firstHopStates(data, featuresNorm, s);

  data.buildFirstHopStateMap(firstHopStates);

  auto kernelMatrix = compute_kernel_matrix_blockwise(
      s, firstHopStates, data.nrNodes, data, nodeDegrees, nodeDegreesC1);

  save_matrix_npy(kernelMatrix.first, "kernel_matrix.npy");

  // Visualization and classification
  plot_kernel_matrix(kernelMatrix.first, show);

  // Extract class labels and features from graph
  std::vector<int> classLabels;

  VertexIterator vi, vi_end;
  for (boost::tie(vi, vi_end) = boost::vertices(data.graph); vi != vi_end;
       ++vi) {
    classLabels.push_back(data.graph[*vi].classLabel);
    featuresNorm.push_back(data.graph[*vi].feature);
  }

  std::vector<dtype> expectationValsAll =
      compute_expectation_values(kernelMatrix.first, classLabels, featuresNorm);
  std::vector<int> predictedLabels = predict_labels(expectationValsAll);

  std::vector<ClassificationResult> results;
  ClassificationMetrics metrics = evaluate_predictions(
      classLabels, predictedLabels, expectationValsAll, results, show);

  std::cout << "\nMetrics:" << std::endl;
  std::cout << "  Accuracy: " << std::fixed << std::setprecision(4)
            << metrics.accuracy << std::endl;
  std::cout << "  Precision: " << std::fixed << std::setprecision(4)
            << metrics.precision << std::endl;
  std::cout << "  Recall: " << std::fixed << std::setprecision(4)
            << metrics.recall << std::endl;
  std::cout << "  F1 Score: " << std::fixed << std::setprecision(4)
            << metrics.f1Score << std::endl;

  return 0;
}
