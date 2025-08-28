// Helper functions to read CSV files
#include "input.hpp"

std::vector<std::vector<std::string>> readCSV(const std::string &filename) {
  std::vector<std::vector<std::string>> data;
  std::ifstream file(filename);
  std::string line;

  if (!file.is_open()) {
    std::cerr << "Error: Could not open file " << filename << std::endl;
    return data;
  }

  while (std::getline(file, line)) {
    std::vector<std::string> row;
    std::stringstream ss(line);
    std::string cell;

    while (std::getline(ss, cell, ',')) {
      // Remove quotes if present
      if (!cell.empty() && cell.front() == '"' && cell.back() == '"') {
        cell = cell.substr(1, cell.length() - 2);
      }
      row.push_back(cell);
    }
    data.push_back(row);
  }

  file.close();
  return data;
}

// Read gene list for class labels
std::set<std::string> readGeneList(const std::string &filename) {
  std::set<std::string> genes;
  std::ifstream file(filename);
  std::string line;

  if (!file.is_open()) {
    std::cerr << "Error: Could not open file " << filename << std::endl;
    return genes;
  }

  // Skip the header
  std::getline(file, line);

  while (std::getline(file, line)) {
    std::stringstream ss(line);
    std::string gene_name;
    // Read the first cell, which is the gene name.
    std::getline(ss, gene_name, ',');

    // Remove quotes if present.
    if (!gene_name.empty() && gene_name.front() == '"' &&
        gene_name.back() == '"') {
      gene_name = gene_name.substr(1, gene_name.length() - 2);
    }

    if (!gene_name.empty()) {
      genes.insert(gene_name);
    }
  }

  file.close();
  return genes;
}

GraphData input_data(const std::string &graph_type, int n, int m, bool show) {
  GraphData data;

  if (graph_type == "synthetic") {
    std::cout << "Creating synthetic graph with Boost Graph Library..."
              << std::endl;

    // Derived quantities
    data.nrNodes = 1 << n; // 2^n
    data.s = 1 << m;       // 2^m
    data.maxD = data.s;
    data.n = n;
    data.m = m;
    data.p = n;
    data.P = 1 << data.p;

    std::cout << "Number of Nodes: " << data.nrNodes << std::endl;
    dtype start = 1.0 - 1.0 / data.P;
    dtype stop = 0.0;
    int num = data.nrNodes;

    // Add vertices to the graph
    std::vector<Vertex> vertices(data.nrNodes);
    for (int i = 0; i < data.nrNodes; i++) {
      dtype val = start + i * (stop - start) / (num - 1);
      vertices[i] = boost::add_vertex(data.graph);
      data.graph[vertices[i]].id = i;
      // Initialize feature as descending values
      data.graph[vertices[i]].feature = val;
      // Class labels: 0 if feature < 0.5, else 1
      data.graph[vertices[i]].classLabel =
          (data.graph[vertices[i]].feature < 0.5) ? 0 : 1;
    }

    // Circulant graph shifts
    std::vector<int> shifts;
    if (data.s <= 2) {
      shifts = {1, -1};
    } else {
      int half_s = data.s / 2;
      for (int i = 1; i <= half_s; i++) {
        shifts.push_back(i);
        shifts.push_back(-i);
      }
    }

    // Add edges to create circulant graph
    std::set<std::pair<int, int>> addedEdges; // To avoid duplicate edges

    for (int i = 0; i < data.nrNodes; i++) {
      for (int shift : shifts) {
        int j =
            (i + shift + data.nrNodes) % data.nrNodes; // Handle negative modulo

        // Avoid self-loops and duplicate edges
        if (i != j && addedEdges.find({std::min(i, j), std::max(i, j)}) ==
                          addedEdges.end()) {
          boost::add_edge(vertices[i], vertices[j], EdgeProperties{1},
                          data.graph);
          addedEdges.insert({std::min(i, j), std::max(i, j)});
        }
      }
    }

    std::cout << "Created synthetic circulant graph with "
              << boost::num_vertices(data.graph) << " vertices and "
              << boost::num_edges(data.graph) << " edges." << std::endl;

  } else if (graph_type == "PPI") {
    std::cout << "\nImporting PPI dataset with Boost Graph Library..."
              << std::endl;

    // Read CSV files
    auto edgesRaw =
        readCSV("./dataset/OmniPath_gene_edges_filtered_with_index.csv");
    auto degreesRaw =
        readCSV("./dataset/OmniPath_gene_degrees_filtered_with_index.csv");
    auto featuresRaw =
        readCSV("./dataset/MutSig_gene_pvalues_filtered_with_index.csv");
    auto genes_all_set = readGeneList("./dataset/Census_all.csv");

    if (edgesRaw.empty() || degreesRaw.empty() || featuresRaw.empty()) {
      std::cerr << "Error: Could not read input files properly." << std::endl;
      return data;
    }

    std::cout << "Finished reading files." << std::endl;

    // Determine number of nodes from degrees file
    data.nrNodes = degreesRaw.size() - 1; // Subtract header
    data.n = (int)std::ceil(std::log2(data.nrNodes));
    data.p = 7;
    data.P = 1 << data.p;

    // Add vertices to the graph
    std::vector<Vertex> vertices(data.nrNodes);
    std::vector<std::string> geneNames;

    // Read features and gene names (limit to 6850)
    size_t maxFeatures = std::min((size_t)6851, featuresRaw.size());
    std::vector<dtype> rawFeatures;

    for (size_t i = 1; i < maxFeatures; i++) { // Skip header
      if (featuresRaw[i].size() > 3) {
        rawFeatures.push_back(std::stod(featuresRaw[i][3]));
        geneNames.push_back(featuresRaw[i][1]);
      }
    }

    // Normalize features
    dtype minFeat = *std::min_element(rawFeatures.begin(), rawFeatures.end());
    dtype maxFeat = *std::max_element(rawFeatures.begin(), rawFeatures.end());

    // Create vertices with properties
    for (int i = 0; i < std::min(data.nrNodes, (int)rawFeatures.size()); i++) {
      vertices[i] = boost::add_vertex(data.graph);
      data.graph[vertices[i]].id = i;
      data.graph[vertices[i]].feature =
          (rawFeatures[i] - minFeat) / (maxFeat - minFeat);

      // Set class label based on gene list
      if (i < (int)geneNames.size()) {
        data.graph[vertices[i]].classLabel =
            (genes_all_set.count(geneNames[i]) > 0) ? 1 : 0;
      } else {
        data.graph[vertices[i]].classLabel = 0;
      }
    }

    // Add remaining vertices if needed
    for (int i = rawFeatures.size(); i < data.nrNodes; i++) {
      vertices[i] = boost::add_vertex(data.graph);
      data.graph[vertices[i]].id = i;
      data.graph[vertices[i]].feature = 0.0;
      data.graph[vertices[i]].classLabel = 0;
    }

    // Add edges from the edges file
    std::cout << "Adding edges..." << std::endl;
    size_t maxEdges = std::min((size_t)27076, edgesRaw.size());
    int edgesAdded = 0;

    for (size_t i = 1; i < maxEdges; i++) { // Skip header
      if (edgesRaw[i].size() > 2) {
        int v = std::stoi(edgesRaw[i][0]) - 1; // Convert to 0-based
        int u = std::stoi(edgesRaw[i][2]) - 1; // Convert to 0-based

        if (v >= 0 && v < data.nrNodes && u >= 0 && u < data.nrNodes &&
            v != u) {
          boost::add_edge(vertices[v], vertices[u], EdgeProperties{1},
                          data.graph);
          edgesAdded++;
        }
      }
    }

    std::cout << "Added " << edgesAdded << " edges to the graph." << std::endl;

    // Calculate max degree
    data.calculateDegrees();

    int maxDegree = 0;
    VertexIterator vi, vi_end;
    for (boost::tie(vi, vi_end) = boost::vertices(data.graph); vi != vi_end;
         ++vi) {
      maxDegree = std::max(maxDegree, data.graph[*vi].degree);
    }
    data.maxD = maxDegree;
    data.m = (int)std::ceil(std::log2(data.maxD));
    data.s = 1 << data.m;

    // Count class labels
    int positiveLabels = 0;
    for (boost::tie(vi, vi_end) = boost::vertices(data.graph); vi != vi_end;
         ++vi) {
      if (data.graph[*vi].classLabel == 1)
        positiveLabels++;
    }

    std::cout << "Number of positive labels: " << positiveLabels << " / "
              << boost::num_vertices(data.graph) << std::endl;
    std::cout << "Number of negative labels: "
              << (boost::num_vertices(data.graph) - positiveLabels) << " / "
              << boost::num_vertices(data.graph) << std::endl;
  }

  // Calculate degrees for all vertices
  data.calculateDegrees();

  // Calculate mC1 and sC1 based on second-order degrees
  int maxC1 = 0;
  VertexIterator vi, vi_end;
  for (boost::tie(vi, vi_end) = boost::vertices(data.graph); vi != vi_end;
       ++vi) {
    maxC1 = std::max(maxC1, data.graph[*vi].secondOrderDegree);
  }
  data.mC1 = (int)std::ceil(std::log2(maxC1));
  data.sC1 = 1 << data.mC1;

  // Build adjacency matrix
  data.buildAdjacencyMatrix();

  // Print summary
  std::cout << "\n=== Graph Summary ===" << std::endl;
  std::cout << "Number of nodes: " << boost::num_vertices(data.graph)
            << ", n = " << data.n << ", N = " << (1 << data.n) << std::endl;
  std::cout << "Number of edges: " << boost::num_edges(data.graph) << std::endl;
  std::cout << "Immediate neighbors: max = " << data.maxD << ", m = " << data.m
            << ", s = " << data.s << std::endl;
  std::cout << "Second-hop neighbors: max = " << maxC1
            << ", m_c1 = " << data.mC1 << ", s_c1 = " << data.sC1 << std::endl;
  std::cout << "Features scaling: p = " << data.p << ", P = " << data.P
            << std::endl;

  return data;
}

void printAdjacencyMatrix(const GraphData &data, int maxNodes) {
  std::cout << "\n=== Adjacency Matrix (first "
            << std::min(maxNodes, (int)boost::num_vertices(data.graph)) << "x"
            << std::min(maxNodes, (int)boost::num_vertices(data.graph))
            << " entries) ===" << std::endl;

  int printSize = std::min(maxNodes, (int)boost::num_vertices(data.graph));

  // Print column headers
  std::cout << "    ";
  for (int j = 0; j < printSize; j++) {
    std::cout << std::setw(3) << j;
  }
  std::cout << std::endl;

  // Print rows
  for (int i = 0; i < printSize; i++) {
    std::cout << std::setw(3) << i << ":";
    for (int j = 0; j < printSize; j++) {
      std::cout << std::setw(3) << data.adjacencyMatrix[i][j];
    }
    std::cout << std::endl;
  }
}

void printGraphProperties(const GraphData &data, int maxNodes) {
  std::cout << "\n=== Vertex Properties (first "
            << std::min(maxNodes, (int)boost::num_vertices(data.graph))
            << " nodes) ===" << std::endl;

  VertexIterator vi, vi_end;
  int count = 0;

  for (boost::tie(vi, vi_end) = boost::vertices(data.graph);
       vi != vi_end && count < maxNodes; ++vi, ++count) {
    std::cout << "Node " << data.graph[*vi].id << ": ";
    std::cout << "degree=" << data.graph[*vi].degree;
    std::cout << ", 2nd_degree=" << data.graph[*vi].secondOrderDegree;
    std::cout << ", feature=" << std::fixed << std::setprecision(4)
              << data.graph[*vi].feature;
    std::cout << ", class=" << data.graph[*vi].classLabel;

    // Print neighbors
    std::cout << ", neighbors=[";
    AdjacencyIterator ai, ai_end;
    bool first = true;
    for (boost::tie(ai, ai_end) = boost::adjacent_vertices(*vi, data.graph);
         ai != ai_end; ++ai) {
      if (!first)
        std::cout << ", ";
      std::cout << data.graph[*ai].id;
      first = false;
    }
    std::cout << "]" << std::endl;
  }
}

void printGraphStatistics(const GraphData &data) {
  std::cout << "\n=== Graph Statistics ===" << std::endl;

  // Basic statistics
  std::cout << "Vertices: " << boost::num_vertices(data.graph) << std::endl;
  std::cout << "Edges: " << boost::num_edges(data.graph) << std::endl;

  // Degree statistics
  std::vector<int> degrees;
  std::vector<dtype> features;
  int positiveLabels = 0;

  VertexIterator vi, vi_end;
  for (boost::tie(vi, vi_end) = boost::vertices(data.graph); vi != vi_end;
       ++vi) {
    degrees.push_back(data.graph[*vi].degree);
    features.push_back(data.graph[*vi].feature);
    if (data.graph[*vi].classLabel == 1)
      positiveLabels++;
  }

  if (!degrees.empty()) {
    dtype avgDegree =
        std::accumulate(degrees.begin(), degrees.end(), 0.0) / degrees.size();
    int minDegree = *std::min_element(degrees.begin(), degrees.end());
    int maxDegree = *std::max_element(degrees.begin(), degrees.end());

    std::cout << "Degree - Min: " << minDegree << ", Max: " << maxDegree
              << ", Avg: " << std::fixed << std::setprecision(2) << avgDegree
              << std::endl;
  }

  if (!features.empty()) {
    dtype avgFeature = std::accumulate(features.begin(), features.end(), 0.0) /
                        features.size();
    dtype minFeature = *std::min_element(features.begin(), features.end());
    dtype maxFeature = *std::max_element(features.begin(), features.end());

    std::cout << "Features - Min: " << std::fixed << std::setprecision(4)
              << minFeature << ", Max: " << maxFeature
              << ", Avg: " << avgFeature << std::endl;
  }

  std::cout << "Class Labels - Positive: " << positiveLabels << " ("
            << std::fixed << std::setprecision(1)
            << (100.0 * positiveLabels / boost::num_vertices(data.graph))
            << "%)"
            << ", Negative: "
            << (boost::num_vertices(data.graph) - positiveLabels) << std::endl;
}
