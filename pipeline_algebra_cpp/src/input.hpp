// Helper functions to read CSV files
#pragma once

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "pipeline_utils.hpp"

std::vector<std::vector<std::string>> readCSV(const std::string &filename);

// Read gene list for class labels
std::set<std::string> readGeneList(const std::string &filename);

GraphData input_data(const std::string &graph_type = "synthetic", int n = 7,
                     int m = 2, bool show = false);

void printAdjacencyMatrix(const GraphData &data, int maxNodes = 10);

void printGraphProperties(const GraphData &data, int maxNodes = 10);

void printGraphStatistics(const GraphData &data);
