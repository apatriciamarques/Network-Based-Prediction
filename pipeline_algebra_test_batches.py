# Note: in the C++ pipeline, gene symbols/features indexed in the order they appear (QMME)
# in MutSig_gene_pvalues_filtered_with_index.csv (rows 1…6850 → indices 0…6849)
# MoPro embeddings live on a very different scale than QMME.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import ttest_rel

OUTPUT_DIR = "output/PPI_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
DTYPE = np.float32

# -------------------------
# Data loading
# -------------------------
def input_data(show=False):
    featuresRaw = pd.read_csv("dataset/MutSig_gene_pvalues_filtered_with_index.csv")
    genes_all = pd.read_excel("dataset/Census_all.xlsx")

    featuresGenes = featuresRaw.iloc[1:6851, 1].astype(str).tolist()
    classLabels = np.array([1 if g in set(genes_all['Gene Symbol'].values) else 0 for g in featuresGenes], dtype=np.int32)

    if show:
        print(f"Number of nodes: {len(classLabels)}")
        print(f"Positive labels: {classLabels.sum()}, Negative labels: {len(classLabels)-classLabels.sum()}")

    return classLabels, featuresGenes

def inspect_kernel(kernel, name="Kernel"):
    print(f"\n=== Inspecting {name} ===")
    print(f"Shape: {kernel.shape}")
    print(f"Type: {kernel.dtype}")
    print(f"Min: {np.nanmin(kernel):.6e}")
    print(f"Max: {np.nanmax(kernel):.6e}")
    print(f"Mean: {np.nanmean(kernel):.6e}")
    print(f"Median: {np.nanmedian(kernel):.6e}")
    print(f"Diag mean: {np.nanmean(np.diag(kernel)):.6e}")
    print(f"Diag min: {np.nanmin(np.diag(kernel)):.6e}")
    print(f"Diag max: {np.nanmax(np.diag(kernel)):.6e}")
    print(f"Any NaNs? {np.isnan(kernel).any()}")
    print(f"Any Infs? {np.isinf(kernel).any()}")

def check_extremes(kernel, name="Kernel"):
    max_idx = np.unravel_index(np.argmax(kernel), kernel.shape)
    min_idx = np.unravel_index(np.argmin(kernel), kernel.shape)
    print(f"\n{name} extremes:")
    print(f"  Max value {kernel[max_idx]:.6e} at index {max_idx}")
    print(f"  Min value {kernel[min_idx]:.6e} at index {min_idx}")
    if max_idx[0] == max_idx[1]:
        print("  → Max is on diagonal ✅")
    else:
        print("  → Max is off-diagonal ❌")
    if min_idx[0] == min_idx[1]:
        print("  → Min is on diagonal ⚠️")
    else:
        print("  → Min is off-diagonal ✅")

# -------------------------
# Compute expectation values
# -------------------------
def compute_expectation_values(kernelMatrix, trainLabels, train_idx, test_idx, dtype=np.float32):
    wV = np.ones(len(train_idx), dtype=dtype) / len(train_idx)
    expectationVals = []
    for v in test_idx:
        val = np.sum(((-1) ** trainLabels[train_idx]) * wV * kernelMatrix[v, train_idx])
        expectationVals.append(val)
    return np.array(expectationVals, dtype=dtype)

def predict_labels(expectationValsAll, dtype=np.float32):
    predictedLabels = 0.5 * (1 - np.sign(expectationValsAll))
    return predictedLabels.astype(int)

# -------------------------
# Evaluate batch predictions
# -------------------------
def evaluate_predictions(trueLabels, predictedLabels, expectationValsAll, test_idx=None):
    posClass, negClass = 1, 0
    tp = np.sum((predictedLabels==posClass)&(trueLabels==posClass))
    fp = np.sum((predictedLabels==posClass)&(trueLabels==negClass))
    fn = np.sum((predictedLabels==negClass)&(trueLabels==posClass))
    accuracy = np.mean(predictedLabels==trueLabels)
    precision = tp/(tp+fp) if tp+fp else 0
    recall = tp/(tp+fn) if tp+fn else 0
    f1 = 2*precision*recall/(precision+recall) if precision+recall else 0
    return {"Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1 Score": f1}

# -------------------------
# Generate batches
# -------------------------
def generate_batches(classLabels, n_runs=50, test_frac=0.85, seed_base=42):
    rng = np.random.default_rng(seed_base)

    pos_idx = np.where(classLabels == 1)[0]
    neg_idx = np.where(classLabels == 0)[0]

    # Fixed train set
    n_train_pos = int(len(pos_idx) * 0.8)
    train_pos_idx = rng.choice(pos_idx, size=n_train_pos, replace=False)
    train_neg_idx = rng.choice(neg_idx, size=n_train_pos, replace=False)
    train_idx = np.concatenate([train_pos_idx, train_neg_idx])

    # Remaining indices for test batches
    remaining_pos_idx = np.setdiff1d(pos_idx, train_pos_idx)
    remaining_neg_idx = np.setdiff1d(neg_idx, train_neg_idx)

    batches = []
    for seed in range(n_runs):
        rng = np.random.default_rng(seed_base + seed)
        n_test_pos = int(len(remaining_pos_idx) * test_frac)
        test_pos_idx = rng.choice(remaining_pos_idx, size=n_test_pos, replace=False)
        n_test_neg = n_test_pos
        test_neg_idx = rng.choice(remaining_neg_idx, size=n_test_neg, replace=False)
        test_idx = np.concatenate([test_pos_idx, test_neg_idx])
        batches.append(test_idx)

    return train_idx, batches

# -------------------------
# Run for a kernel
# -------------------------
def run_pipeline(kernelMatrix, classLabels, train_idx, test_batches, tag="QMME"):
    all_metrics = []
    all_sizes = []

    for i, test_idx in enumerate(test_batches):
        expectationValsTest = compute_expectation_values(kernelMatrix, classLabels, train_idx, test_idx)
        predictedLabels = predict_labels(expectationValsTest)
        metrics = evaluate_predictions(classLabels[test_idx], predictedLabels, expectationValsTest, test_idx=test_idx)

        all_metrics.append(metrics)
        all_sizes.append({
            "TrainSize": len(train_idx),
            "TrainPos": np.sum(classLabels[train_idx] == 1),
            "TrainNeg": np.sum(classLabels[train_idx] == 0),
            "TestSize": len(test_idx),
            "TestPos": np.sum(classLabels[test_idx] == 1),
            "TestNeg": np.sum(classLabels[test_idx] == 0)
        })

    # Save results
    avg_metrics = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0]}
    final_file = os.path.join(OUTPUT_DIR, f"final_results_{tag}.txt")
    with open(final_file, "w") as f:
        f.write(f"=== Training Set ({tag}) ===\n")
        f.write(f"Train size: {len(train_idx)} (Pos: {np.sum(classLabels[train_idx] == 1)}, "
                f"Neg: {np.sum(classLabels[train_idx] == 0)})\n\n")
        f.write("=== Test Batches Info ===\n")
        for j, (size, metrics) in enumerate(zip(all_sizes, all_metrics)):
            f.write(f"Batch {j+1}: Test size: {size['TestSize']} (Pos: {size['TestPos']}, Neg: {size['TestNeg']}) "
                    f"Metrics: Accuracy={metrics['Accuracy']:.4f}, Precision={metrics['Precision']:.4f}, "
                    f"Recall={metrics['Recall']:.4f}, F1={metrics['F1 Score']:.4f}\n")
        f.write("\n=== Average Metrics Over All Test Batches ===\n")
        for k, v in avg_metrics.items():
            f.write(f"{k}: {v:.4f}\n")

    print(f"[{tag}] Saved results to {final_file}")
    return all_metrics

# -------------------------
# Main
# -------------------------
classLabels, featuresGenes = input_data(show=True)

# Load kernels
kernel_qmme = np.load("output/kernel_matrix_final.npy")
# kernel_qmme /= np.sqrt(np.mean(np.diag(kernel_qmme)))
kernel_mopro = np.load("mopro/mopro_kernel_matrix.npy")
# kernel_mopro /= np.sqrt(np.mean(np.diag(kernel_mopro)))

print("QMME kernel shape:", kernel_qmme.shape)
print("MoPro kernel shape:", kernel_mopro.shape)

# Sanity check: diagonal values should be > 0
print("QMME diag mean:", np.mean(np.diag(kernel_qmme)))
print("MoPro diag mean:", np.mean(np.diag(kernel_mopro)))

# Ensure MoPro aligns
if kernel_mopro.shape != kernel_qmme.shape:
    raise ValueError(f"Kernel size mismatch: QMME={kernel_qmme.shape}, MoPro={kernel_mopro.shape}")

seeds = [0, 42, 123, 999]
for seed in seeds:
    train_idx, test_batches = generate_batches(classLabels, n_runs=50, test_frac=0.85, seed_base=seed)
    
    results_qmme = run_pipeline(kernel_qmme, classLabels, train_idx, test_batches, tag=f"QMME_seed{seed}")
    results_mopro = run_pipeline(kernel_mopro, classLabels, train_idx, test_batches, tag=f"MoPro_seed{seed}")

    # Now compute paired t-test (batches are aligned)
    print("\n=== Paired t-tests (QMME vs MoPro) ===")
    metrics_names = results_qmme[0].keys()
    for metric in metrics_names:
        qm = np.array([m[metric] for m in results_qmme])
        mp = np.array([m[metric] for m in results_mopro])
        diff = qm - mp
        mean_diff = np.mean(diff)
        std_diff = np.std(diff)
        print(f"\nMetric: {metric}")
        print(f"  QMME mean: {qm.mean():.4f}, MoPro mean: {mp.mean():.4f}")
        print(f"  Mean difference: {mean_diff:.4f} ± {std_diff:.4f}")
        if std_diff > 0:
            t_stat, p_val = ttest_rel(qm, mp)
            print(f"  Paired t-test: t={t_stat:.4f}, p={p_val:.4f}")
            if p_val < 0.05:
                print("    → Significant difference (p<0.05)")
            else:
                print("    → Not significantly different")
        else:
            print("  Paired t-test skipped (no variation in differences)")

# compute paired t-test here

# -------------------------
# Plot Full Kernels
# -------------------------
# plt.figure(figsize=(18,8))

# plt.subplot(1,2,1)
# sns.heatmap(kernel_qmme, cmap="viridis", cbar_kws={'label': 'Kernel value'})
# plt.title("QMME Kernel")
# plt.xlabel("Node Index")
# plt.ylabel("Node Index")

# plt.subplot(1,2,2)
# sns.heatmap(kernel_mopro, cmap="viridis", cbar_kws={'label': 'Kernel value'})
# plt.title("MoPro Kernel")
# plt.xlabel("Node Index")
# plt.ylabel("Node Index")

# plt.tight_layout()
# plt.show()