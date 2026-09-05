# Implementation Versions & Architectural Specifications

**Edge-Food-Vision-ResNet50: Trustworthy Edge AI & Risk Control**

---

## 1. Architectural Overview & Tier Comparison

| Feature / Metric | Tier 1: Embedded Runtime | Tier 2: Conformal Risk Calibration | Tier 3: Dynamic Exit & C-TTA |
| :--- | :--- | :--- | :--- |
| **Directory** | [`implementations/v1_edge_device_runtime/`](../implementations/v1_edge_device_runtime/) | [`implementations/v2_conformal_risk_calibration/`](../implementations/v2_conformal_risk_calibration/) | [`implementations/v3_dynamic_exit_test_time_adaptation/`](../implementations/v3_dynamic_exit_test_time_adaptation/) |
| **Target Platform** | Raspberry Pi 4 / Jetson Nano | Embedded Linux CPU / FPGA | Edge Co-Processor / Stream Pipeline |
| **Implementation Language** | Python / ONNX Runtime | Python / NumPy / SciPy | PyTorch / Streaming Quantile Engine |
| **Compression Ratio** | 4.0x (Symmetric INT8) | **8.0x (Pruned-Aware INT8)** | 8.0x INT8 + Dynamic Exit |
| **FLOP Compute Reduction**| 0% (Full backbone) | 50% (Channel pruned) | **49.8% additional reduction** |
| **Safety Assurance** | Softmax heuristic | **Finite-sample 95% coverage** | **Online Martingale 94.5% under drift** |
| **Latency Speedup** | 3.19x vs FP32 | 4.71x vs FP32 | **Up to 7.2x on canonical inputs** |
| **Memory Footprint** | 24.5 MB | **12.2 MB** | 12.8 MB (with early probes) |

---

## 2. Directory Structure & File Map

```text
Edge-Food-Vision-ResNet50/
├── implementations/
│   ├── v1_edge_device_runtime/
│   │   ├── onnx_runtime_engine.py               # Embedded ONNX engine with memory pinning
│   │   └── embedded_camera_inference_runner.py  # Deterministic edge streaming inference loop
│   ├── v2_conformal_risk_calibration/
│   │   ├── conformal_prediction_engine.py       # Split-conformal calibration with finite-sample bounds
│   │   └── pruning_aware_quantization.py        # PAQ INT8 benchmark generating Pareto curves
│   └── v3_dynamic_exit_test_time_adaptation/
│       ├── dynamic_early_exit_resnet50.py       # Entropy-gated early exits saving 49.8% FLOPs
│       ├── online_conformal_tta_tracker.py      # Martingale-bounded online test-time adaptation
│       └── dynamic_exit_benchmark.py            # Tier 3 dynamic exit & C-TTA benchmark runner
```

---

## 3. Execution Instructions

### 3.1 Run Tier 1 Embedded Camera Streaming Loop
```bash
python -m implementations.v1_edge_device_runtime.embedded_camera_inference_runner
```

### 3.2 Run Tier 2 Conformal Risk & PAQ Benchmark
```bash
python -m implementations.v2_conformal_risk_calibration.pruning_aware_quantization
```

### 3.3 Run Tier 3 Dynamic Early Exit & C-TTA Benchmark
```bash
python -m implementations.v3_dynamic_exit_test_time_adaptation.dynamic_exit_benchmark
```
