# Note: in the C++ pipeline, gene symbols/features indexed in the order they appear (QMME)
# in MutSig_gene_pvalues_filtered_with_index.csv (rows 1…6850 → indices 0…6849)
# MoPro embeddings live on a very different scale than QMME.
# Refactored pipeline: repeated random splits with dynamic training sets

# Training set changes for every split → 80% of positives + equal negatives.
# Test set is fully disjoint, includes remaining positives + remaining negatives.
# Repeats 50 splits → metrics averaged across splits.
# No single seed dominates performance.
# Simplified run_pipeline to loop through (train_idx, test_idx) pairs.

import numpy as np
import pandas as pd
import os
from scipy.stats import ttest_rel
from sklearn.metrics import average_precision_score

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

# -------------------------
# Compute expectation values
# -------------------------
def compute_expectation_values(kernelMatrix, trainLabels, train_idx, test_idx, dtype=np.float32):
    wV = np.ones(len(train_idx), dtype=dtype) / len(train_idx)
    expectationVals = np.sum(((-1) ** trainLabels[train_idx]) * wV * kernelMatrix[np.ix_(test_idx, train_idx)], axis=1)
    return expectationVals.astype(dtype)

def predict_labels(expectationValsAll):
    predictedLabels = 0.5 * (1 - np.sign(expectationValsAll))
    return predictedLabels.astype(int)

# -------------------------
# Evaluate predictions
# -------------------------
def evaluate_predictions(trueLabels, predictedLabels):
    posClass, negClass = 1, 0
    tp = np.sum((predictedLabels==posClass) & (trueLabels==posClass))
    fp = np.sum((predictedLabels==posClass) & (trueLabels==negClass))
    fn = np.sum((predictedLabels==negClass) & (trueLabels==posClass))
    accuracy = np.mean(predictedLabels==trueLabels)
    precision = tp/(tp+fp) if tp+fp else 0
    recall = tp/(tp+fn) if tp+fn else 0
    f1 = 2*precision*recall/(precision+recall) if precision+recall else 0
    return {"Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1 Score": f1}

# -------------------------
# Generate repeated random splits
# -------------------------
def generate_splits(classLabels, n_splits=50, train_pos_frac=0.8, seed_base=42, balanced_test=False, verbose=True):
    rng = np.random.default_rng(seed_base)
    pos_idx = np.where(classLabels == 1)[0]
    neg_idx = np.where(classLabels == 0)[0]

    splits = []
    for s in range(n_splits):
        rng = np.random.default_rng(seed_base + s)
        # Training
        n_train_pos = int(len(pos_idx) * train_pos_frac)
        train_pos_idx = rng.choice(pos_idx, size=n_train_pos, replace=False)
        train_neg_idx = rng.choice(neg_idx, size=n_train_pos, replace=False)
        train_idx = np.concatenate([train_pos_idx, train_neg_idx])

        # Test
        remaining_pos_idx = np.setdiff1d(pos_idx, train_pos_idx)
        remaining_neg_idx = np.setdiff1d(neg_idx, train_neg_idx)

        if balanced_test:
            n_test_pos = len(remaining_pos_idx)
            test_pos_idx = rng.choice(remaining_pos_idx, size=n_test_pos, replace=False)
            test_neg_idx = rng.choice(remaining_neg_idx, size=n_test_pos, replace=False)
        else:
            test_pos_idx = remaining_pos_idx
            test_neg_idx = remaining_neg_idx
        test_idx = np.concatenate([test_pos_idx, test_neg_idx])

        splits.append((train_idx, test_idx))

        if verbose:
            print(f"Split {s+1}:")
            print(f"  Train size: {len(train_idx)} (Pos: {np.sum(classLabels[train_idx]==1)}, Neg: {np.sum(classLabels[train_idx]==0)})")
            print(f"  Test size : {len(test_idx)} (Pos: {np.sum(classLabels[test_idx]==1)}, Neg: {np.sum(classLabels[test_idx]==0)})\n")

    return splits

# -------------------------
# Run pipeline for a kernel
# -------------------------
def run_pipeline(kernelMatrix, classLabels, splits, tag="QMME"):
    all_metrics = []

    for i, (train_idx, test_idx) in enumerate(splits):
        expectationVals = compute_expectation_values(kernelMatrix, classLabels, train_idx, test_idx)
        predictedLabels = predict_labels(expectationVals)
        metrics = evaluate_predictions(classLabels[test_idx], predictedLabels)
        all_metrics.append(metrics)

    # Save results
    avg_metrics = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0]}
    final_file = os.path.join(OUTPUT_DIR, f"final_results_{tag}.txt")
    with open(final_file, "w") as f:
        f.write(f"=== {tag} Pipeline ===\n")
        for j, metrics in enumerate(all_metrics):
            f.write(f"Split {j+1}: Accuracy={metrics['Accuracy']:.4f}, "
                    f"Precision={metrics['Precision']:.4f}, Recall={metrics['Recall']:.4f}, "
                    f"F1={metrics['F1 Score']:.4f}\n")
        f.write("\n=== Average Metrics ===\n")
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
kernel_mopro = np.load("mopro/mopro_kernel_matrix.npy")

# Ensure alignment
if kernel_mopro.shape != kernel_qmme.shape:
    raise ValueError(f"Kernel size mismatch: QMME={kernel_qmme.shape}, MoPro={kernel_mopro.shape}")

# Generate repeated random splits
splits = generate_splits(classLabels, n_splits=50, seed_base=42)

# Run pipelines
results_qmme = run_pipeline(kernel_qmme, classLabels, splits, tag="QMME")
results_mopro = run_pipeline(kernel_mopro, classLabels, splits, tag="MoPro")

# Paired t-tests with significance
print("\n=== Paired t-tests (QMME vs MoPro) ===")
metrics_names = results_qmme[0].keys()
for metric in metrics_names:
    qm = np.array([m[metric] for m in results_qmme])
    mp = np.array([m[metric] for m in results_mopro])
    t_stat, p_val = ttest_rel(qm, mp)
    sig = "Significant" if p_val < 0.05 else "Not significant"
    print(f"{metric}: QMME mean={qm.mean():.4f}, MoPro mean={mp.mean():.4f}, "
          f"t={t_stat:.4f}, p={p_val:.4e} → {sig}")
    
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


def compute_auprc(trueLabels, predictedScores):
    return average_precision_score(trueLabels, predictedScores)

# Example usage for your repeated splits:
auprc_qmme = [compute_auprc(classLabels[test_idx], compute_expectation_values(kernel_qmme, classLabels, train_idx, test_idx)) for train_idx, test_idx in splits]
auprc_mopro = [compute_auprc(classLabels[test_idx], compute_expectation_values(kernel_mopro, classLabels, train_idx, test_idx)) for train_idx, test_idx in splits]

print("QMME AUPRC: mean={:.4f}, std={:.4f}".format(np.mean(auprc_qmme), np.std(auprc_qmme)))
print("MoPro AUPRC: mean={:.4f}, std={:.4f}".format(np.mean(auprc_mopro), np.std(auprc_mopro)))