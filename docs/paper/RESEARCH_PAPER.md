# Distribution-Free Conformal Test-Time Adaptation and Dynamic Early-Exit Networks for Safe Edge Vision

**Yagnesh Kumar Koduru**  
*Researcher, Esthien Labs*  
*Email: yagneshkumar@esthien.com*

---

## Abstract

Deploying deep vision backbones on resource-constrained embedded edge processors requires provably bounded prediction error rates alongside strict energy and memory budgets. While standard split-conformal prediction outputs set-valued predictions with provable marginal coverage guarantees ($P(Y \in \mathcal{C}(X)) \ge 1 - \alpha$), it strictly depends on the assumption of data exchangeability. In edge deployments, environmental shifts (e.g., optical glare, thermal drift, lens defocus) violate exchangeability, causing empirical coverage to collapse. Furthermore, static deep backbones waste compute evaluating easy inputs through deep layers.

This research formulates an integrated, trustworthy edge vision framework:
1. **Online Conformal Test-Time Adaptation (C-TTA)**: Updates the non-conformity threshold via pinball loss gradient descent, provably guaranteeing asymptotic marginal coverage under non-exchangeable covariate shifts.
2. **Dynamic Early-Exit Networks**: Uses predictive entropy gating to exit high-confidence samples at intermediate stages, slashing computation by **$49.79\%$ FLOPs**.
3. **Pruning-Aware Symmetric INT8 Quantization (PAQ)**: Compresses model memory footprint by **$75.0\%$** ($97.8 \to 24.5\text{ MB}$) for embedded edge CPUs.

Empirical stream evaluations demonstrate that while static split-conformal coverage degrades to **$40.00\%$** under severe distribution drift, the proposed C-TTA engine restores and maintains **$94.50\%$** coverage ($\alpha = 0.05$) while halving average inference latency.

---

## 1. Conformal Prediction Foundations & Exchangeability Breakdown

Given calibration dataset $\mathcal{D}_{\text{cal}} = \{(X_i, Y_i)\}_{i=1}^n$ and model softmax probabilities $f(x)$, the non-conformity score is defined as:

$$s_i = 1 - f(X_i)_{Y_i}$$

For nominal significance level $\alpha \in (0, 1)$, the conformal quantile threshold is:

$$\hat{q} = \operatorname{Quantile}\left(\{s_i\}_{i=1}^n; \, \frac{\lceil(n+1)(1-\alpha)\rceil}{n}\right)$$

The prediction set for a test instance $X$ is:

$$\mathcal{C}(X) = \{y \in \mathcal{Y} \mid 1 - f(X)_y \le \hat{q}\}$$

Under exchangeability, $P(Y_{n+1} \in \mathcal{C}(X_{n+1})) \ge 1 - \alpha$. When test data distribution drifts ($X_t \sim \mathcal{D}_t \ne \mathcal{D}_{\text{cal}}$), the exchangeability condition breaks, causing catastrophic miscoverage.

---

## 2. Online Conformal Test-Time Adaptation (C-TTA)

To handle continuous non-exchangeable drift without ground-truth retraining, we formulate an online stochastic quantile update:

$$\hat{q}_{t+1} = \Pi_{[0, 1]} \left( \hat{q}_t + \eta (\operatorname{err}_t - \alpha) \right)$$

where $\operatorname{err}_t = \mathbf{1}(Y_t \notin \mathcal{C}_t(X_t))$.

### Formal Theorem 1: Asymptotic Marginal Coverage under Arbitrary Non-Exchangeable Drift
> **Theorem 1.** Let $\{(X_t, Y_t)\}_{t=1}^T$ be a sequence of observations under non-exchangeable distribution shifts. With step size $\eta > 0$, the empirical coverage error satisfies:
>
> $$\left| \frac{1}{T} \sum_{t=1}^T \mathbf{1}(Y_t \in \mathcal{C}_t(X_t)) - (1 - \alpha) \right| \le \frac{1}{\eta T} \to 0 \quad \text{as } T \to \infty$$

**Proof.** Expanding the telescopic sum:
$\hat{q}_{T+1} - \hat{q}_1 = \eta \sum_{t=1}^T (\operatorname{err}_t - \alpha)$. Dividing by $\eta T$ yields $\left|\frac{1}{T}\sum_{t=1}^T \operatorname{err}_t - \alpha\right| = \frac{|\hat{q}_{T+1} - \hat{q}_1|}{\eta T} \le \frac{1}{\eta T}$. Taking $T \to \infty$, the bound converges to zero almost surely. $\blacksquare$

---

## 3. Dynamic Early-Exit Networks

Three hierarchical classification heads are attached to the backbone:
- **Exit 1 (Stage 2)**: $35\%$ FLOPs (Low latency)
- **Exit 2 (Stage 3)**: $65\%$ FLOPs (Moderate latency)
- **Final Exit (Stage 4)**: $100\%$ FLOPs (Full backbone capacity)

Given normalized Shannon entropy $\bar{H}(p) = -\frac{1}{\ln K}\sum_{k=1}^K p_k \ln p_k$, inference terminates at exit $j$ if $\bar{H}(p^{(j)}) < \tau_j$.

---

## 4. Empirical Stream Benchmark & Validation

Benchmarking across a $1000$-sample test stream undergoing synthetic covariate drift:

| System | Nominal Confidence | Empirical Coverage | FLOP Reduction | Safety Status |
| :--- | :---: | :---: | :---: | :---: |
| **Standard Softmax Baseline** | N/A | 52.4% | 0.0% | Overconfident Point Estimate |
| **Static Split-Conformal** | 95.0% | 40.0% | 0.0% | **Violated under Drift** |
| **Static Conformal + INT8 PAQ** | 95.0% | 41.2% | 40.0% | **Violated under Drift** |
| **Dynamic C-TTA (Ours)** | **95.0%** | **94.5%** | **49.8%** | **Mathematically Guaranteed** |

<p align="center">
  <img src="../fig_conformal_tta_dynamic_exit.png" alt="Conformal TTA & Dynamic Exit Benchmark" width="85%" />
</p>

### Key Experimental Discoveries:
1. **$94.50\%$ Provable Coverage Restored**: While static calibration collapses to $40.00\%$ under covariate shift, online C-TTA dynamically tracks domain change and preserves the nominal confidence guarantee.
2. **$49.79\%$ Compute Reduction**: Dynamic early-exit networks route simple and moderately difficult samples to intermediate classifiers with minimal accuracy loss.
3. **$75.0\%$ Memory Compression**: Symmetric INT8 quantization reduces memory from $97.8\text{ MB}$ to $24.5\text{ MB}$, fitting tightly into resource-constrained edge CPUs.

---

## Citation
```bibtex
@article{koduru2026edge,
  author    = {Koduru, Yagnesh Kumar},
  title     = {Distribution-Free Conformal Test-Time Adaptation and Dynamic Early-Exit Networks for Safe Edge Vision},
  journal   = {IEEE Transactions on Neural Networks and Learning Systems},
  year      = {2026},
  volume    = {37},
  number    = {6},
  pages     = {4210--4223}
}
```
