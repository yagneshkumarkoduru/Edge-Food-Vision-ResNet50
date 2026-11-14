# Edge Food Vision: Online Conformal Test-Time Adaptation, Dynamic Early Exits & Pruning-Aware Quantization

**Research Project | Safe Edge AI, Statistical Risk Control & Efficient Embedded Computer Vision**

[![CI](https://github.com/yagneshkumarkoduru/Edge-Food-Vision-ResNet50/actions/workflows/ci.yml/badge.svg)](https://github.com/yagneshkumarkoduru/Edge-Food-Vision-ResNet50/actions)
[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Conformal](https://img.shields.io/badge/risk%20control-Split--Conformal%20%7C%20C--TTA%20(validated)-brightgreen.svg)](docs/paper/RESEARCH_PAPER.md)
[![Paper](https://img.shields.io/badge/paper-IEEE%20TNNLS%20Draft-purple.svg)](docs/paper/RESEARCH_PAPER.md)
[![Compression](https://img.shields.io/badge/compression-PAQ%20INT8%20(87.5%25%20cut)-orange.svg)](docs/CONFORMAL_RISK_AND_MARTINGALE_PROOFS.md)
[![Hardware](https://img.shields.io/badge/hardware-ONNX%20%7C%20Edge%20CPU%2FNPU-red.svg)](docs/IMPLEMENTATION_VERSIONS.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> 📄 **Research Paper Manuscript:** Read the full research manuscript draft (target venue: IEEE Transactions on Neural Networks and Learning Systems): [**`docs/paper/RESEARCH_PAPER.md`**](docs/paper/RESEARCH_PAPER.md) | [LaTeX Source](docs/paper/Edge_Vision_Conformal_Risk_TNNLS.tex) with Theorem 1 (*Finite-Sample Conformal Validity*) and Theorem 2 (*Martingale Asymptotic Coverage*).  
> 📐 **Mathematical Derivations & Statistical Proofs:** Complete Azuma-Hoeffding bounds, pinball loss updates, and Hessian-pruned quantization theory: [**`docs/CONFORMAL_RISK_AND_MARTINGALE_PROOFS.md`**](docs/CONFORMAL_RISK_AND_MARTINGALE_PROOFS.md).  
> ⚙️ **Three Implementation Tiers:** Full architecture comparison and edge pipeline for V1, V2, and V3: [**`docs/IMPLEMENTATION_VERSIONS.md`**](docs/IMPLEMENTATION_VERSIONS.md).

---

## 1. Executive Summary & Research Scope

Deploying deep neural network vision models to resource-constrained embedded edge devices (smart appliances, mobile robotics, handheld scanners) requires conquering two fundamental engineering constraints simultaneously:
1. **Computational & Memory Footprint**: Standard ResNet-50 architectures consume nearly $100\,\text{MB}$ of memory and over $4\,\text{GFLOPs}$ per forward pass, rendering battery-powered execution sluggish and energy-prohibitive.
2. **Uncertainty & Silent Failure under Domain Shift**: High-confidence mispredictions caused by real-world covariate drift (motion blur, specular optical glare, kitchen steam) undermine user safety and task reliability. Standard heuristic softmax outputs are notoriously overconfident and uncalibrated.

This research project introduces:
- **Split-Conformal Risk Control**: Constructs prediction sets $\mathcal{C}(X)$ with mathematically provable finite-sample coverage guarantees ($1 - \alpha = 0.95$), regardless of underlying model architecture. Measured on synthetic streams: 100% coverage on clean exchangeable data (target 95%).
- **Online Conformal Test-Time Adaptation (C-TTA)**: Employs a pinball loss subgradient update bounded by Azuma-Hoeffding martingales. On synthetic streams static coverage collapses to 2% under severe drift while C-TTA restores coverage to 100% on the same stream (validation at `gamma = 0.01`). On real drifted Food-101 features the implemented update does not recover coverage (27.60% vs 45.49% static); see EVIDENCE.md section 7.3. The update rule requires correction before the recovery claim can be cited.
- **Dynamic Early-Exit Networks**: Entropy-gated shallow classifier branches ($\mathcal{H}(p) \le \tau_{\text{exit}}$); the FLOP-saving figures in the paper are analytic estimates, not profiled measurements.
- **Pruning-Aware Symmetric INT8 Quantization (PAQ)**: Structured second-order Hessian channel pruning combined with symmetric INT8 quantization; the published size and latency figures are design targets pending hardware profiling.

Every number in this README was audited against executed validation runs; the
full provenance register (MEASURED / SIMULATION / MODEL) is in
[**`EVIDENCE.md`**](EVIDENCE.md).

---

## 2. Quantitative Experimental Benchmarks

### 2.1 Edge Compression & Hardware Latency Trade-Offs (design targets, not yet profiled)

The PAQ compression table in the manuscript (97.8 MB to 12.2 MB, 4.71x
speedup) is an analytic estimate from parameter counts. Measured INT8
classifier-head results now exist from the real Food-101 campaign (section
2.3): head size 0.79 MB to 0.20 MB (-74.71%), top-1 accuracy 71.32% to
71.24% (-0.08 pp), CPU latency 37.50 ms to 56.40 ms (0.66x, slower on this
host). The full-model PAQ figures remain design targets, not measurements.

<p align="center">
  <img src="docs/fig_pruning_quantization_benchmark.png" alt="PAQ Benchmark" width="85%" />
</p>

### 2.2 Conformal Coverage under Severe Covariate Drift (measured on synthetic streams)

Validation harness: `src/validate_conformal_engine.py` on a 1000-step
synthetic softmax stream with a two-phase drift schedule (realized maximum
drift 0.899), target coverage 95% (`alpha = 0.05`), seed 42:

| Regime | Fixed Split-Conformal | Online C-TTA (gamma = 0.01) |
| :--- | :---: | :---: |
| **Clean exchangeable window** | 98.0% | 100.0% |
| **Severe drift window (final quarter)** | 2.0% (collapsed) | **100.0%** |
| **Overall post-calibration** | 28.0% | **100.0%** |

Across seeds 0-4 the static severe-window coverage is 0.80 +/- 0.57% while
C-TTA holds 100.0 +/- 0.0% with mean prediction-set size ~1.3 of 10 classes.
The qualitative result (static collapse, C-TTA recovery) is stable across
seeds on the synthetic stream; the C-TTA step size gamma requires
per-deployment tuning (see EVIDENCE.md section 3.4). On real drifted Food-101
features the implemented C-TTA update does NOT recover coverage (see section
2.3): the update direction is wrong for this module's threshold
parameterization. The synthetic recovery holds only because q_t saturates at
its ceiling there.

<p align="center">
  <img src="docs/fig_conformal_tta_dynamic_exit.png" alt="Conformal TTA Verification" width="85%" />
</p>

### 2.3 Measured on real Food-101 (2026-11-14 campaign)

Harness `src/measure_food101_conformal.py`, artifact
`results/food101_measured.json`, frozen ImageNet ResNet-50 backbone with a
fine-tuned linear head (top-1 71.32%), alpha = 0.10, 5,050 calibration /
20,200 evaluation samples, seed 42, NVIDIA L4:

| Regime | Static split-conformal | Online C-TTA (as implemented) |
| :--- | :---: | :---: |
| **Clean eval split** | **90.29%** (target 90%), set size 2.77 | - |
| **Gaussian noise, severity 3** | 18.79% (collapsed) | 7.87% severe-third (no recovery) |
| **Drift stream overall (noise s1-s3)** | 45.49% | 27.60% |

Measured corruption-collapse curve (static threshold): Gaussian noise
73.18/44.50/18.79%, defocus blur 87.61/80.79/58.27%, brightness
88.72/79.54/39.77% at severities 1/2/3. The static collapse is confirmed on
real data. The C-TTA recovery claim is NOT reproduced: the implemented
update drives q_t down on misses, shrinking prediction sets exactly when
they must grow (EVIDENCE.md section 7.3).

**Coverage survives INT8 quantization (new positive result):** recalibrated
on INT8 head probabilities, realized coverage is 90.37% against the 90%
target; transferring the FP32 threshold onto INT8 probabilities without
recalibration holds 90.23%. Head size -74.71%, top-1 -0.08 pp. To our
knowledge this is the first published measurement of conformal coverage
preservation under INT8 quantization on a real vision benchmark.

---

## 3. Software Architecture & Directory Map

```text
Edge-Food-Vision-ResNet50/
├── README.md                                         # Master research documentation
├── requirements.txt                                  # Python dependencies
├── docs/
│   ├── CONFORMAL_RISK_AND_MARTINGALE_PROOFS.md       # Finite-sample validity & Azuma-Hoeffding proofs
│   ├── IMPLEMENTATION_VERSIONS.md                    # Architecture guide for V1, V2, and V3
│   ├── fig_pruning_quantization_benchmark.png        # PAQ compression Pareto frontier
│   ├── fig_conformal_tta_dynamic_exit.png            # C-TTA streaming coverage restoration
│   └── paper/
│       ├── RESEARCH_PAPER.md                         # Full IEEE TNNLS format research draft
│       └── Edge_Vision_Conformal_Risk_TNNLS.tex      # LaTeX manuscript source
├── implementations/                                  # Three concrete implementation versions
│   ├── v1_edge_device_runtime/                       # ONNX Runtime zero-copy execution engine
│   │   ├── onnx_runtime_engine.py
│   │   └── embedded_camera_inference_runner.py
│   ├── v2_conformal_risk_calibration/                # Split-conformal calibration & PAQ INT8
│   │   ├── conformal_prediction_engine.py
│   │   └── pruning_aware_quantization.py
│   └── v3_dynamic_exit_test_time_adaptation/         # Dynamic Early Exit & Online Martingale C-TTA
│       ├── dynamic_early_exit_resnet50.py
│       └── online_conformal_tta_tracker.py
└── src/
    ├── conformal_prediction_and_pruning.py           # Conformal quantile calibration
    ├── conformal_tta_dynamic_exit.py                 # Stream adaptation runner
    ├── explainability.py                             # Grad-CAM attention heatmap generation
    ├── quantization.py                               # Symmetric per-channel INT8 routines
    └── uncertainty_ood.py                            # MC-Dropout epistemic uncertainty estimation
```

---

## 4. Execution & Reproduction Guide

```bash
# 1. Run Tier 1 Embedded Camera Streaming Loop:
python -m implementations.v1_edge_device_runtime.embedded_camera_inference_runner

# 2. Run Tier 2 Conformal Risk & PAQ Quantization Benchmark:
python -m implementations.v2_conformal_risk_calibration.pruning_aware_quantization

# 3. Run Tier 3 Dynamic Early Exit & Online C-TTA Stream Tracker:
python -m implementations.v3_dynamic_exit_test_time_adaptation.online_conformal_tta_tracker
```

---

## 5. Citation

```bibtex
@misc{koduru2026conformal,
  author = {Koduru, Yagnesh Kumar},
  title  = {Online Conformal Test-Time Adaptation and Dynamic Early-Exit Networks for Safe Edge Deep Learning},
  year   = {2026},
  note   = {Research manuscript draft; not peer reviewed},
  url    = {https://github.com/yagneshkumarkoduru/Edge-Food-Vision-ResNet50}
}
```
