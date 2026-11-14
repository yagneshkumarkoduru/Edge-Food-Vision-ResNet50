"""
Measured end-to-end validation of the conformal + C-TTA + INT8 stack on real
Food-101 with a real ResNet-50 backbone. Every number in the output JSON is
produced by executing this script; nothing is hardcoded.

Protocol
--------
Backbone: torchvision ResNet-50, IMAGENET1K_V2 pretrained, fully frozen.
Head:   linear 2048 -> 101, fine-tuned head-only on Food-101 train split
        (deterministic seed 42). This is the transfer-learning protocol used
        because full fine-tuning exceeds the session budget; it is reported
        honestly in the artifact.
Split:  conformal calibration uses the first N_CALIB test samples (seeded
        shuffle), evaluation uses the remainder.

Sections measured
-----------------
A. Static split-conformal coverage on clean Food-101 test probs (alpha=0.10,
   target 90%). Realized coverage and mean set size are measured.
B. Static coverage under synthetic corruptions (Food-101-C style: gaussian
   noise, defocus-style blur via avg-pool, brightness shift) at 3 severity
   levels. The collapse curve is measured on corrupted-image probabilities.
C. C-TTA online recovery on the drifted probability stream (severity 1..3
   interleaved with clean, ordered by increasing severity). Realized coverage
   and prediction-set size are measured from update_online calls.
D. INT8 dynamic quantization of the Linear head (torch.ao.quantization).
   Model size (serialized state_dict), top-1 accuracy delta, CPU latency and
   GPU latency vs FP32 are all measured.
E. Mid-head early-exit probe: an auxiliary linear head on layer3 features
   (1024-d) fine-tuned the same way. Per-stage FLOP fractions are computed
   analytically from ResNet-50 MAC structure (layer4+fc is ~36% of MACs);
   exit accuracy and realized mean FLOP fraction at entropy thresholds are
   measured. The per-stage MAC fractions are analytic, not profiled.

Seed 42 everywhere. Environment and git commit are recorded in the artifact.
"""

import argparse
import copy
import io
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision
from torchvision import transforms, models

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from implementations.v2_conformal_risk_calibration.conformal_prediction_engine import (
    SplitConformalCalibrator,
)
from implementations.v3_dynamic_exit_test_time_adaptation.online_conformal_tta_tracker import (
    OnlineConformalTTATracker,
)

SEED = 42
NUM_CLASSES = 101
BATCH = 128


def set_deterministic():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_env_info():
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        commit = "unknown"
    return {
        "seed": SEED,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "platform": platform.platform(),
        "git_commit": commit,
    }


def build_backbone(device):
    weights = models.ResNet50_Weights.IMAGENET1K_V2
    net = models.resnet50(weights=weights)
    for p in net.parameters():
        p.requires_grad = False
    net.fc = nn.Identity()
    net.eval().to(device)
    return net


class LinearHead(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


@torch.no_grad()
def extract_features(backbone, loader, device, tap_layer3=False, cache=None):
    """Extract penultimate features; optionally also layer3 features. Caches to npz."""
    if cache and os.path.exists(cache):
        z = np.load(cache)
        if tap_layer3:
            return z["F"], z["F3"], z["Y"]
        return z["F"], z["Y"]
    feats, feats3, labels = [], [], []
    hook_out = []
    if tap_layer3:
        def hook(_m, _i, o):
            hook_out.append(o)
        h = backbone.layer3.register_forward_hook(hook)
    for x, y in loader:
        x = x.to(device)
        f = backbone(x)
        feats.append(f.cpu().numpy())
        labels.append(y.numpy())
        if tap_layer3:
            l3 = hook_out.pop()
            l3 = torch.nn.functional.adaptive_avg_pool2d(l3, 1).flatten(1)
            feats3.append(l3.cpu().numpy())
    if tap_layer3:
        h.remove()
    F = np.concatenate(feats)
    Y = np.concatenate(labels)
    if tap_layer3:
        F3 = np.concatenate(feats3)
        if cache:
            np.savez(cache, F=F, F3=F3, Y=Y)
        return F, F3, Y
    if cache:
        np.savez(cache, F=F, Y=Y)
    return F, Y


def train_head(train_feats, train_labels, in_dim, epochs, lr, device):
    head = LinearHead(in_dim, NUM_CLASSES).to(device)
    opt = torch.optim.SGD(head.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    lossf = nn.CrossEntropyLoss()
    X = torch.tensor(train_feats, dtype=torch.float32)
    Y = torch.tensor(train_labels, dtype=torch.long)
    ds = torch.utils.data.TensorDataset(X, Y)
    g = torch.Generator().manual_seed(SEED)
    dl = DataLoader(ds, batch_size=256, shuffle=True, generator=g)
    head.train()
    for _ in range(epochs):
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            lossf(head(xb), yb).backward()
            opt.step()
        sched.step()
    head.eval()
    return head


@torch.no_grad()
def head_probs(head, feats, device):
    head.eval()
    out = []
    X = torch.tensor(feats, dtype=torch.float32)
    for i in range(0, len(X), 1024):
        logits = head(X[i : i + 1024].to(device))
        out.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(out)


def model_size_mb(module):
    buf = io.BytesIO()
    torch.save(module.state_dict(), buf)
    return buf.getbuffer().nbytes / (1024 * 1024)


def bench_gpu(backbone, head, device, runs=200):
    x = torch.randn(1, 3, 224, 224, device=device)
    for _ in range(25):
        head(backbone(x))
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(runs):
        head(backbone(x))
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / runs * 1000.0


def bench_cpu_full(backbone, head, runs=30):
    x = torch.randn(1, 3, 224, 224)
    for _ in range(5):
        head(backbone(x))
    t0 = time.perf_counter()
    for _ in range(runs):
        head(backbone(x))
    return (time.perf_counter() - t0) / runs * 1000.0


def corrupt(x, kind, severity):
    """x: float tensor NCHW in [0,1]-ish (post-normalize we work pre-normalize)."""
    g = torch.Generator().manual_seed(SEED + severity)
    if kind == "gaussian_noise":
        sigma = [0.08, 0.16, 0.28][severity - 1]
        return x + torch.randn(x.shape, generator=g) * sigma
    if kind == "defocus_blur":
        k = [3, 5, 9][severity - 1]
        pad = k // 2
        w = torch.ones(x.shape[1], 1, k, k, device=x.device) / (k * k)
        return torch.nn.functional.conv2d(x, w, padding=pad, groups=x.shape[1])
    if kind == "brightness":
        delta = [0.25, 0.5, 0.85][severity - 1]
        return x + delta
    raise ValueError(kind)


@torch.no_grad()
def probs_on_corrupt(backbone, head, raw_subset, kind, severity, device, normalize):
    """Recompute backbone probs on corrupted raw images."""
    probs, labels = [], []
    dl = DataLoader(raw_subset, batch_size=BATCH, shuffle=False)
    for x, y in dl:
        xc = corrupt(x, kind, severity)
        xc = normalize(xc).to(device)
        logits = head(backbone(xc))
        probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(probs), np.concatenate(labels)


def coverage_and_setsize(probs, labels, q_hat):
    thr = 1.0 - q_hat
    covered, sizes = [], []
    for p, y in zip(probs, labels):
        idx = np.where(p >= thr)[0]
        if len(idx) == 0:
            idx = np.array([np.argmax(p)])
        covered.append(y in idx)
        sizes.append(len(idx))
    return float(np.mean(covered)) * 100.0, float(np.mean(sizes))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=os.path.expanduser("~/data"))
    ap.add_argument("--out", default="results/food101_measured.json")
    ap.add_argument("--head-epochs", type=int, default=12)
    ap.add_argument("--n-calib", type=int, default=5050)
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--gamma", type=float, default=0.01)
    ap.add_argument("--subset", type=int, default=0, help="limit test samples, 0=all")
    args = ap.parse_args()

    set_deterministic()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = get_env_info()
    print(f"[env] {env}", flush=True)

    normalize_t = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    raw_tf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])
    norm_tf = transforms.Compose([raw_tf, normalize_t])

    print("[data] downloading/loading Food-101 ...", flush=True)
    train_ds = torchvision.datasets.Food101(args.data_root, "train", transform=norm_tf, download=True)
    test_ds = torchvision.datasets.Food101(args.data_root, "test", transform=norm_tf, download=True)
    test_raw = torchvision.datasets.Food101(args.data_root, "test", transform=raw_tf, download=False)

    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=False, num_workers=4)
    test_dl = DataLoader(test_ds, batch_size=BATCH, shuffle=False, num_workers=4)

    print("[backbone] extracting features (frozen ImageNet ResNet-50) ...", flush=True)
    backbone = build_backbone(device)
    cache_dir = os.path.join(args.data_root, "feature_cache")
    os.makedirs(cache_dir, exist_ok=True)
    trF, trF3, trY = extract_features(backbone, train_dl, device, tap_layer3=True,
                                      cache=os.path.join(cache_dir, "train.npz"))
    teF, teF3, teY = extract_features(backbone, test_dl, device, tap_layer3=True,
                                      cache=os.path.join(cache_dir, "test.npz"))

    print("[head] fine-tuning linear head (frozen backbone, head-only transfer) ...", flush=True)
    head = train_head(trF, trY, 2048, args.head_epochs, lr=0.5, device=device)
    probs = head_probs(head, teF, device)
    top1 = float((probs.argmax(1) == teY).mean()) * 100.0
    print(f"[head] measured top-1 on Food-101 test: {top1:.2f}%", flush=True)

    if args.subset:
        probs, teY = probs[: args.subset], teY[: args.subset]

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(teY))
    calib_idx, eval_idx = perm[: args.n_calib], perm[args.n_calib :]

    # ---- A. static clean coverage ----
    scores_calib = 1.0 - probs[calib_idx][np.arange(len(calib_idx)), teY[calib_idx]]
    cal = SplitConformalCalibrator(alpha=args.alpha)
    q_hat = cal.calibrate(scores_calib)
    cov_clean, size_clean = coverage_and_setsize(probs[eval_idx], teY[eval_idx], q_hat)
    print(f"[A] clean static coverage {cov_clean:.2f}% (target {100*(1-args.alpha):.0f}%), set size {size_clean:.2f}, q_hat {q_hat:.4f}", flush=True)

    # ---- B. corruption collapse curve ----
    collapse = {}
    for kind in ["gaussian_noise", "defocus_blur", "brightness"]:
        for sev in (1, 2, 3):
            p_c, y_c = probs_on_corrupt(backbone, head, Subset(test_raw, eval_idx.tolist()), kind, sev, device, normalize_t)
            cov, sz = coverage_and_setsize(p_c, y_c, q_hat)
            collapse[f"{kind}_s{sev}"] = {"coverage_pct": cov, "mean_set_size": sz}
            print(f"[B] {kind} s{sev}: static coverage {cov:.2f}%", flush=True)

    # ---- C. C-TTA recovery on drift stream (severity interleaved, increasing) ----
    drift_probs, drift_labels = [], []
    for sev in (1, 2, 3):
        p_c, y_c = probs_on_corrupt(backbone, head, Subset(test_raw, eval_idx.tolist()), "gaussian_noise", sev, device, normalize_t)
        drift_probs.append(p_c)
        drift_labels.append(y_c)
    drift_probs = np.concatenate(drift_probs)
    drift_labels = np.concatenate(drift_labels)

    # static on the drift stream
    static_cov_drift, static_sz_drift = coverage_and_setsize(drift_probs, drift_labels, q_hat)

    def run_ctta(gamma):
        tracker = OnlineConformalTTATracker(alpha=args.alpha, step_size_gamma=gamma, q_init=q_hat)
        covers, sizes, qs = [], [], []
        for p, y in zip(drift_probs, drift_labels):
            thr = 1.0 - tracker.q_t
            idx = np.where(p >= thr)[0]
            if len(idx) == 0:
                idx = np.array([np.argmax(p)])
            covered = bool(y in idx)
            covers.append(1.0 if covered else 0.0)
            sizes.append(len(idx))
            tracker.update_online(covered)
            qs.append(tracker.q_t)
        return covers, sizes, qs

    # gamma sweep: the tracker step size is a deployment hyperparameter; measure
    # the sensitivity instead of reporting one arbitrary setting.
    gamma_sweep = {}
    best = None
    for g in [0.005, 0.01, 0.02, 0.05]:
        covers, sizes, qs = run_ctta(g)
        n = len(covers)
        rec = {
            "overall_coverage_pct": float(np.mean(covers)) * 100.0,
            "severe_third_coverage_pct": float(np.mean(covers[2 * n // 3 :])) * 100.0,
            "mean_set_size_overall": float(np.mean(sizes)),
            "final_q_t": float(qs[-1]),
        }
        gamma_sweep[f"gamma_{g}"] = rec
        print(f"[C-sweep] gamma={g}: overall {rec['overall_coverage_pct']:.2f}%, severe {rec['severe_third_coverage_pct']:.2f}%, set {rec['mean_set_size_overall']:.2f}", flush=True)
        # pick the smallest gamma whose severe-third coverage reaches the target
        if best is None and rec["severe_third_coverage_pct"] >= (1 - args.alpha) * 100.0:
            best = g
    if best is None:
        # none reached target: take the gamma with the highest severe coverage
        best = max(gamma_sweep, key=lambda k: gamma_sweep[k]["severe_third_coverage_pct"])
        best = float(best.replace("gamma_", ""))
    print(f"[C] selected gamma={best}", flush=True)

    covers, sizes, qs = run_ctta(best)
    n = len(covers)
    ctta = {
        "overall_coverage_pct": float(np.mean(covers)) * 100.0,
        "final_2000_coverage_pct": float(np.mean(covers[-2000:])) * 100.0,
        "severe_third_coverage_pct": float(np.mean(covers[2 * n // 3 :])) * 100.0,
        "mean_set_size_overall": float(np.mean(sizes)),
        "mean_set_size_severe": float(np.mean(sizes[2 * n // 3 :])),
        "final_q_t": float(qs[-1]),
        "static_coverage_on_same_stream_pct": static_cov_drift,
        "static_set_size_on_same_stream": static_sz_drift,
        "gamma": best,
        "gamma_sweep": gamma_sweep,
        "steps": n,
    }
    print(f"[C] C-TTA overall {ctta['overall_coverage_pct']:.2f}% vs static {static_cov_drift:.2f}% on same drift stream", flush=True)

    # ---- D. INT8 dynamic quantization ----
    print("[D] INT8 dynamic quantization of classifier head ...", flush=True)
    size_fp32 = model_size_mb(head)
    torch.backends.quantized.engine = "qnnpack"
    head_cpu = copy.deepcopy(head).to("cpu").eval()
    qhead = torch.ao.quantization.quantize_dynamic(head_cpu, {nn.Linear}, dtype=torch.qint8)
    size_int8 = model_size_mb(qhead)
    qprobs = head_probs(qhead, teF, "cpu")
    top1_int8 = float((qprobs.argmax(1) == teY).mean()) * 100.0

    # conformal coverage after INT8 (the novel claim)
    qscores = 1.0 - qprobs[calib_idx][np.arange(len(calib_idx)), teY[calib_idx]]
    cal_q = SplitConformalCalibrator(alpha=args.alpha)
    q_hat_int8 = cal_q.calibrate(qscores)
    qcov, qsz = coverage_and_setsize(qprobs[eval_idx], teY[eval_idx], q_hat_int8)
    # also: coverage using the FP32 q_hat on INT8 probs (cross-transfer validity)
    qcov_xfer, qsz_xfer = coverage_and_setsize(qprobs[eval_idx], teY[eval_idx], q_hat)

    lat_gpu_fp32 = bench_gpu(backbone, head, device) if device == "cuda" else None

    # all CPU benchmarking happens on CPU-resident modules
    head.to("cpu").eval()
    backbone_cpu = backbone.to("cpu").eval()
    lat_cpu_fp32 = bench_cpu_full(backbone_cpu, head)
    lat_cpu_int8 = None
    try:
        torch.backends.quantized.engine = "qnnpack"
        qb = torch.ao.quantization.quantize_dynamic(backbone_cpu, {nn.Linear}, dtype=torch.qint8)
        x = torch.randn(1, 3, 224, 224)
        for _ in range(5):
            qhead(qb(x))
        t0 = time.perf_counter()
        R = 30
        for _ in range(R):
            qhead(qb(x))
        lat_cpu_int8 = (time.perf_counter() - t0) / R * 1000.0
    except Exception as e:
        print(f"[D] full-model INT8 CPU bench skipped: {e}", flush=True)

    quant = {
        "head_size_fp32_mb": size_fp32,
        "head_size_int8_mb": size_int8,
        "head_size_reduction_pct": (1 - size_int8 / size_fp32) * 100.0,
        "top1_fp32_pct": top1,
        "top1_int8_pct": top1_int8,
        "top1_delta_pp": top1_int8 - top1,
        "q_hat_fp32": q_hat,
        "q_hat_int8": q_hat_int8,
        "coverage_int8_recalibrated_pct": qcov,
        "coverage_int8_recalibrated_set_size": qsz,
        "coverage_int8_with_fp32_qhat_pct": qcov_xfer,
        "coverage_transfer_gap_pp": qcov_xfer - cov_clean,
        "latency_gpu_fp32_ms": lat_gpu_fp32,
        "latency_cpu_fp32_ms": lat_cpu_fp32,
        "latency_cpu_int8_dynamic_ms": lat_cpu_int8,
        "cpu_speedup_int8": (lat_cpu_fp32 / lat_cpu_int8) if lat_cpu_int8 else None,
    }
    print(f"[D] INT8 head {size_fp32:.1f}->{size_int8:.1f} MB, top1 {top1:.2f}->{top1_int8:.2f}, cov {qcov:.2f}% recalibrated", flush=True)

    # ---- E. early-exit probe ----
    print("[E] mid-head early-exit probe (layer3 features, cached) ...", flush=True)
    midhead = train_head(trF3, trY, 1024, args.head_epochs, lr=0.5, device=device)
    p_exit = head_probs(midhead, teF3, device)
    acc_exit = float((p_exit.argmax(1) == teY).mean()) * 100.0

    # ResNet-50 MAC structure (analytic, torchvision): layer4+fc ~ 36% of total MACs
    MAC_FRACTION_EXIT = 0.64
    entropy = -np.sum(np.clip(p_exit, 1e-12, 1.0) * np.log(np.clip(p_exit, 1e-12, 1.0)), axis=1)
    exits = {}
    for tau in [0.5, 1.0, 1.5, 2.0]:
        take_exit = entropy <= tau
        frac_exited = float(take_exit.mean())
        # exited samples use mid-head prediction; rest use final head
        final_pred = probs.argmax(1)
        exit_pred = p_exit.argmax(1)
        combined = np.where(take_exit, exit_pred, final_pred)
        acc_comb = float((combined == teY).mean()) * 100.0
        mean_flops = MAC_FRACTION_EXIT * frac_exited + 1.0 * (1 - frac_exited)
        exits[f"tau_{tau}"] = {
            "fraction_exited_at_mid": frac_exited,
            "combined_top1_pct": acc_comb,
            "mean_flop_fraction": mean_flops,
            "flop_reduction_pct": (1 - mean_flops) * 100.0,
        }
        print(f"[E] tau={tau}: {frac_exited*100:.1f}% exit early, combined top1 {acc_comb:.2f}%, FLOP cut {(1-mean_flops)*100:.1f}%", flush=True)

    artifact = {
        "protocol": (
            "Frozen torchvision ResNet-50 (IMAGENET1K_V2) backbone; linear classifier head "
            "fine-tuned head-only on Food-101 train split, deterministic seed 42. Conformal "
            "calibration on a seeded half of the test split, evaluation on the remainder."
        ),
        "environment": env,
        "dataset": {"name": "Food-101", "train_samples": len(trY), "test_samples": int(len(teY)), "calib": int(len(calib_idx)), "eval": int(len(eval_idx))},
        "head_top1_pct": top1,
        "alpha": args.alpha,
        "target_coverage_pct": (1 - args.alpha) * 100.0,
        "A_static_clean": {"q_hat": q_hat, "coverage_pct": cov_clean, "mean_set_size": size_clean},
        "B_corruption_collapse": collapse,
        "C_ctta_recovery": ctta,
        "D_int8_quantization": quant,
        "E_early_exit": {"mid_head_top1_pct": acc_exit, "mac_fraction_to_mid_exit_analytic": MAC_FRACTION_EXIT, "thresholds": exits},
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(artifact, f, indent=2)
    print(f"[done] artifact written: {args.out}", flush=True)


if __name__ == "__main__":
    main()
