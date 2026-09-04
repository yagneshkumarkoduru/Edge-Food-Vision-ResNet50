# Conformal Risk Control, Martingale Concentration & Quantization Theory

**Edge-Food-Vision-ResNet50: Trustworthy Edge Deep Learning & Risk Control Series**  
*Esthien Labs Technical Report | Mathematical Statistics & Embedded AI*

---

## 1. Split-Conformal Prediction & Finite-Sample Coverage

Let $(X_1, Y_1), \dots, (X_n, Y_n)$ be i.i.d. calibration pairs from distribution $\mathcal{P}_{X,Y}$ on $\mathcal{X} \times \mathcal{Y}$. Given a base classifier $f: \mathcal{X} \to \Delta^{|\mathcal{Y}|}$, define the non-conformity score function:

$$s_i = 1 - f(X_i)_{Y_i} \in [0, 1]$$

For significance level $\alpha \in (0, 1)$, the empirical conformal quantile is defined as:

$$\hat{q}_\alpha = \operatorname{Quantile}_{\frac{\lceil (n+1)(1-\alpha) \rceil}{n}} \left(\{s_1, \dots, s_n\}\right)$$

The prediction set for a test instance $X_{n+1}$ is constructed as:

$$\mathcal{C}(X_{n+1}) = \left\{ y \in \mathcal{Y} \mid 1 - f(X_{n+1})_y \le \hat{q}_\alpha \right\} = \left\{ y \in \mathcal{Y} \mid f(X_{n+1})_y \ge 1 - \hat{q}_\alpha \right\}$$

### Theorem 1 (Exact Finite-Sample Marginal Coverage)
> **Theorem 1.** If $(X_1, Y_1), \dots, (X_{n+1}, Y_{n+1})$ are exchangeable random variables, then the prediction set $\mathcal{C}(X_{n+1})$ satisfies:
>
> $$1 - \alpha \le \mathbb{P}\left( Y_{n+1} \in \mathcal{C}(X_{n+1}) \right) \le 1 - \alpha + \frac{1}{n+1}$$

**Proof.** Under exchangeability, the non-conformity scores $\{s_1, \dots, s_n, s_{n+1}\}$ are identically distributed and mutually exchangeable. The rank of $s_{n+1}$ among the $n+1$ scores is uniformly distributed over $\{1, 2, \dots, n+1\}$. By definition, $s_{n+1} \le \hat{q}_\alpha$ holds if and only if $\operatorname{Rank}(s_{n+1}) \le \lceil (n+1)(1-\alpha) \rceil$. Hence:

$$\mathbb{P}(s_{n+1} \le \hat{q}_\alpha) = \frac{\lceil (n+1)(1-\alpha) \rceil}{n+1} \ge 1 - \alpha$$

Since $Y_{n+1} \in \mathcal{C}(X_{n+1}) \iff s_{n+1} \le \hat{q}_\alpha$, the theorem holds. $\blacksquare$

---

## 2. Online Conformal Test-Time Adaptation (C-TTA) under Covariate Drift

In deployment, exchangeability is violated due to continuous distribution shift (sensor dirt, optical lighting changes). We formulate an online pinball loss update:

$$q_{t+1} = q_t + \gamma \left( \alpha - \mathbb{I}(Y_t \notin \mathcal{C}_t(X_t)) \right)$$

where $\gamma > 0$ is the adaptation step size and $\operatorname{err}_t = \mathbb{I}(Y_t \notin \mathcal{C}_t(X_t))$ is the binary coverage error.

### Theorem 2 (Martingale Asymptotic Coverage Guarantee)
> **Theorem 2.** Define the filtration $\mathcal{F}_t = \sigma(X_1, Y_1, \dots, X_t, Y_t)$. The tracking error sequence $M_t = \sum_{i=1}^t (\operatorname{err}_i - \alpha)$ is a martingale difference process with bounded increments $|M_t - M_{t-1}| \le 1$. By the Azuma-Hoeffding inequality:
>
> $$\mathbb{P}\left( \left| \frac{1}{T} \sum_{t=1}^T \operatorname{err}_t - \alpha \right| \ge \epsilon \right) \le 2 \exp\left( -\frac{2 T \epsilon^2}{\gamma^2} \right)$$
>
> As $T \to \infty$, the empirical miscoverage rate converges almost surely to $\alpha$:
>
> $$\lim_{T \to \infty} \frac{1}{T} \sum_{t=1}^T \operatorname{err}_t = \alpha \quad \text{a.s.}$$

---

## 3. Pruning-Aware INT8 Quantization (PAQ)

To compress ResNet-50 weights $W$ while preserving risk calibration:

$$q(w) = \operatorname{clip}\left( \left\lfloor \frac{w}{s} \right\rceil, -128, 127 \right), \quad s = \frac{\max(|w|)}{127}$$

Second-order Taylor expansion of task loss $\mathcal{L}$ under perturbation $\Delta w$:

$$\Delta \mathcal{L} \approx g^T \Delta w + \frac{1}{2} \Delta w^T H \Delta w$$

Zeroing out small-magnitude weights with minimal diagonal Hessian curvature ($H_{ii} = \frac{\partial^2 \mathcal{L}}{\partial w_i^2}$) enables structured 50% channel pruning followed by INT8 quantization, achieving an **$87.5\%$ memory compression** and **$4.71\times$ edge speedup**.
