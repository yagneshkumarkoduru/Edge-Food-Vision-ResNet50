"""
Tier 2 Benchmark: Pruning-Aware Quantization (PAQ) & Pareto Efficiency

Status of the numbers in this module
------------------------------------
The four series below (model_sizes_mb, latencies_ms, accuracies, and the
derived compression / speedup ratios) are MODEL literals: they are analytic
targets with no executed measurement behind them in this repository.

Measured replacements now exist for the INT8 row of this table, produced by
src/measure_food101_conformal.py on a real Food-101 test split with a real
ResNet-50 backbone (results/food101_measured.json, evidence class MEASURED):

  classifier-head INT8 dynamic quantization (torch.ao.quantization, qnnpack):
    head size        0.79 MB -> 0.20 MB  (-74.71%)
    top-1 accuracy   71.32%  -> 71.24%   (-0.08 pp)
    CPU latency      37.50 ms FP32 -> 56.40 ms INT8 dynamic (0.66x, i.e. a
                     slowdown on this x86 host: dynamic quantization of a
                     ConvNet only quantizes the Linear layers, and the
                     quantized-head overhead dominates at this scale)
    conformal coverage after INT8 recalibration: 90.37% at alpha=0.10
    (FP32 static clean coverage 90.29%; coverage transfer with the FP32
    q_hat onto INT8 probabilities: 90.23%). The coverage guarantee survives
    INT8 quantization within 0.1 percentage points.

The FP32 / FP16 / pruning rows below remain unmeasured MODEL targets.
Saves publication figure: docs/fig_pruning_quantization_benchmark.png
"""

import os
import numpy as np
import matplotlib.pyplot as plt


def run_paq_benchmark():
    models = ["FP32 Baseline", "FP16 Dynamic", "INT8 Symmetric", "PAQ INT8 (Ours)"]
    # MODEL literals (unmeasured targets). INT8 head-only measured values are
    # documented in the module docstring and results/food101_measured.json.
    model_sizes_mb = [97.8, 48.9, 24.5, 12.2]  # MODEL
    latencies_ms   = [46.2, 28.4, 14.5, 9.8]   # MODEL
    accuracies     = [89.4, 89.2, 88.6, 88.1]  # MODEL

    os.makedirs("docs", exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Latency vs Size
    colors = ['#e74c3c', '#e67e22', '#3498db', '#2ecc71']
    for i, model in enumerate(models):
        ax1.scatter(model_sizes_mb[i], latencies_ms[i], s=200, color=colors[i], label=model, edgecolors='black')
        ax1.annotate(f"{model}\n({accuracies[i]}%)", (model_sizes_mb[i] + 1.5, latencies_ms[i] - 1.0), fontsize=9)

    ax1.set_xlabel("Model Memory Footprint (MB)", fontsize=11, fontweight='bold')
    ax1.set_ylabel("Inference Latency (ms)", fontsize=11, fontweight='bold')
    ax1.set_title("Edge Hardware Pareto Frontier: Memory vs Latency (MODEL targets)", fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left")

    # Plot 2: Compression & Speedup Factors
    x = np.arange(len(models))
    width = 0.35
    compression_ratio = [1.0, 2.0, 4.0, 8.0]   # MODEL
    speedup_ratio     = [1.0, 1.63, 3.19, 4.71]  # MODEL

    ax2.bar(x - width/2, compression_ratio, width, label='Compression Factor (x)', color='#3498db', alpha=0.85)
    ax2.bar(x + width/2, speedup_ratio, width, label='Latency Speedup (x)', color='#2ecc71', alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=15, ha='right', fontsize=10, fontweight='bold')
    ax2.set_ylabel("Improvement Factor (x)", fontsize=11, fontweight='bold')
    ax2.set_title("Edge Efficiency Gains over FP32 Baseline (MODEL targets)", fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.legend(loc="upper left")

    out_path = os.path.join("docs", "fig_pruning_quantization_benchmark.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    print("=" * 70)
    print("TIER 2: PRUNING-AWARE QUANTIZATION (PAQ) BENCHMARK - MODEL TARGETS")
    print("FP32 Baseline : 97.8 MB | 46.2 ms | 89.4% Top-1 Accuracy (MODEL)")
    print("PAQ INT8      : 12.2 MB (87.5% cut) | 9.8 ms (4.7x speedup) | 88.1% Top-1 (MODEL)")
    print("Measured INT8 head: 0.79 MB -> 0.20 MB (-74.71%), top-1 -0.08 pp,")
    print("conformal coverage 90.29% FP32 -> 90.37% INT8 (MEASURED, Food-101).")
    print(f"Publication benchmark plot saved to: {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    run_paq_benchmark()
