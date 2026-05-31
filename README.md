# SAM-SPS

**Adaptive Polyak step sizes for SAM and USAM — match or beat tuned learning rates and Cosine Annealing, with no $\gamma$ tuning.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

`SAM_SPS` is the official PyTorch implementation of the optimizer proposed in:

> **Adaptive Sharpness-Aware Minimization with a Polyak-type Step Size: A Theory-Grounded Scheduler**
> Dimitris Oikonomou, Nicolas Loizou.

The package provides a single `torch.optim.Optimizer` subclass — `SAM_SPS` — that wraps the **Stochastic Polyak Scheduler**, an adaptive learning-rate rule derived from a Polyak-style upper-bound argument on the SAM update. One parameter $\lambda$ switches between **USAM-SPS** ($\lambda=0$) and **SAM-SPS** ($\lambda=1$). The result is a SAM-style optimizer with **closed-form convergence guarantees** (linear for strongly convex, $O(1/T)$ for convex) and **competitive deep-learning performance without learning-rate tuning** — at larger sharpness radii $\rho$, it remains stable while Cosine Annealing collapses.

---

## Table of contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [The algorithm](#the-algorithm)
- [API reference](#api-reference)
- [Experiments](#experiments)
- [Citation](#citation)
- [License](#license)

---

## Installation

From source:

```bash
git clone https://github.com/dimitris-oik/sam_sps.git
cd sam_sps
pip install -e .
```

Requirements: `torch`, `numpy`, `scipy` (only for the numpy experiments), Python 3.8+.

---

## Quick start

Like all SAM-style optimizers, `SAM_SPS` performs **two forward/backward passes per step** and therefore requires a `closure` that re-evaluates the loss:

```python
import torch
from sam_sps import SAM_SPS

model     = MyModel()
criterion = torch.nn.CrossEntropyLoss()

optimizer = SAM_SPS(
    model.parameters(),
    rho=0.1,             # sharpness radius
    lambd=1.0,           # 0.0 -> USAM-SPS, 1.0 -> SAM-SPS
    f_star=0.0,          # mini-batch lower bound; 0 for non-negative losses
    gamma_b=1.0,         # cap on the Stochastic Polyak Scheduler step size
    weight_decay=5e-4,
)

for x, y in loader:
    def closure():
        loss = criterion(model(x), y)
        loss.backward()
        return loss
    optimizer.step(closure)
```

Unlike tuned constant-LR or Cosine-Annealing baselines, **no learning rate needs to be picked or scheduled** — the scheduler computes $\gamma_t$ from the loss and gradient at each iteration.

---

## The algorithm

Each iteration performs an ascent step to the perturbed point $e^t$, then descends from $x^t$ using the gradient at $e^t$ with an adaptive step size:

**1. Perturbation.** For mini-batch loss $f_{S_t}$ and sharpness radius $\rho$,

$$e^t = x^t + \rho \left( 1 - \lambda + \frac{\lambda}{\|\nabla f_{S_t}(x^t)\|} \right) \nabla f_{S_t}(x^t).$$

Setting $\lambda = 0$ gives **USAM**'s unnormalized perturbation; setting $\lambda = 1$ gives **SAM**'s normalized perturbation.

**2. Stochastic Polyak Scheduler.** The step size minimizes the Polyak-style upper bound on $\|x^{t+1} - x^*\|^2$ at the perturbed point, capped by $\gamma_b$:

$$\gamma_t = \min\left\{ \frac{\big[f_{S_t}(e^t) - \ell^*_{S_t} - \langle \nabla f_{S_t}(e^t),\ e^t - x^t \rangle\big]_+}{\|\nabla f_{S_t}(e^t)\|^2},\ \gamma_b \right\}.$$

**3. Descent.**

$$x^{t+1} = x^t - \gamma_t \nabla f_{S_t}(e^t).$$

When $\rho = 0$, the rule reduces to the classical Polyak step / $\mathrm{SPS}_{\max}$ (Loizou et al., 2021) for SGD. When $\lambda = 0$, the ReLU safeguard $[\cdot]_+$ is provably redundant for smooth convex objectives with $\rho \le 1/L$ (Proposition 2.1).

### Theoretical guarantees

| Setting | Method | Rate |
|---|---|---|
| Strongly convex, smooth (deterministic) | USAM-SPS | linear, exact (Theorem 3.1) |
| Convex, smooth (deterministic) | USAM-SPS | $O(1/T)$, exact (Theorem 3.2) |
| Decreasing $\rho_t \downarrow 0$ (deterministic) | USAM-SPS | $\|\nabla f(x^t)\| \to 0$ (Theorem 3.4) |
| Strongly convex, smooth (stochastic) | USAM-SPS | linear, to a neighborhood (Theorem 3.5) |
| Convex, smooth (stochastic) | USAM-SPS | $O(1/T)$, to a neighborhood (Theorem 3.8) |
| Interpolated ($\sigma^2 = 0$) | USAM-SPS | neighborhood collapses; exact convergence (Corollary 3.6) |

The theory is developed for USAM ($\lambda = 0$) and extends naturally to SAM ($\lambda = 1$) — see §4.3 of the paper.

---

## API reference

### `SAM_SPS(params, weight_decay=5e-4, rho=0.1, lambd=1.0, f_star=0.0, gamma_b=1.0)`

| Argument | Type | Default | Description |
|---|---|---|---|
| `params` | iterable | — | Parameters to optimize. |
| `weight_decay` | float | `5e-4` | L2 weight-decay coefficient applied in the final descent step. |
| `rho` | float | `0.1` | Sharpness radius $\rho$. |
| `lambd` | float | `1.0` | Interpolation between USAM (`0.0`) and SAM (`1.0`). |
| `f_star` | float | `0.0` | Lower bound $\ell^*_{S_t}$ on the mini-batch loss. Typically `0.0` for non-negative losses. |
| `gamma_b` | float | `1.0` | Upper bound $\gamma_b$ on the Stochastic Polyak Scheduler step size. |

### Step methods

| Method | Description |
|---|---|
| `step(closure)` | Standard SAM API: performs both passes in one call. `closure` must do a full forward+backward and return the loss. |
| `first_step(zero_grad=False)` | (Internal) Ascent step to the perturbed point $e^t$. |
| `second_step(zero_grad=False)` | (Internal) Restore $x^t$, compute $\gamma_t$, then descend. |

In practice you'll only call `step(closure)`. After each call, the active scheduler value is available on `group['lr']` of every parameter group, which makes logging trivial.

---

## Experiments

### Theory validation (synthetic)

The [`numpy_exps/`](numpy_exps/) directory reproduces the §4.1 plots that empirically verify the linear / sublinear rates predicted by Theorems 3.1–3.2 on a strongly convex ridge-regression problem ($n = d = 100$, $\kappa(A) = 10$):

- [`numpy_exps/loss.py`](numpy_exps/loss.py) — `RidgeRegression` objective with controllable conditioning.
- [`numpy_exps/methods.py`](numpy_exps/methods.py) — `Unified_SAM` (constant step-size baseline), `Unified_SAM_SPS` (deterministic Polyak Scheduler), `Unified_SAM_SPS_max` (Stochastic Polyak Scheduler), and the `USAM_andr` baseline (Andriushchenko & Flammarion, 2022).
- [`numpy_exps/exps.ipynb`](numpy_exps/exps.ipynb) — figure-generation notebook.
- [`numpy_exps/figures/`](numpy_exps/figures/) — the two output PDFs.

### Deep-learning results

Test accuracy of `SAM_SPS` with ResNet-32 on **CIFAR-100**, varying the sharpness radius $\rho$ (bold = best at fixed $\rho$, mean ± std over 3 seeds, from Tables 3–4 of the paper):

**USAM ($\lambda = 0$):**

| | Constant USAM (tuned) | USAM + Cosine Annealing | USAM-SPS |
|---|---|---|---|
| $\rho = 0.1$ | 90.56 ± 0.18 | 90.01 ± 0.32 | **91.81 ± 0.04** |
| $\rho = 0.2$ | 90.45 ± 0.34 | 88.77 ± 0.26 | **92.23 ± 0.22** |
| $\rho = 0.3$ | 90.25 ± 0.10 | 88.05 ± 0.23 | **92.24 ± 0.30** |
| $\rho = 0.4$ | 89.56 ± 0.07 | 86.52 ± 0.04 | **92.01 ± 0.12** |

**SAM ($\lambda = 1$):**

| | Constant SAM (tuned) | SAM + Cosine Annealing | SAM-SPS |
|---|---|---|---|
| $\rho = 0.1$ | 90.17 ± 0.11 | 90.49 ± 0.02 | **91.61 ± 0.12** |
| $\rho = 0.2$ | 90.53 ± 0.02 | 89.03 ± 0.13 | **92.24 ± 0.07** |
| $\rho = 0.3$ | 89.61 ± 0.10 | 87.05 ± 0.24 | **91.70 ± 0.15** |
| $\rho = 0.4$ | 88.64 ± 0.13 | 84.61 ± 0.34 | **90.79 ± 0.16** |

Two key observations:

1. **No tuning, best accuracy.** SAM-SPS / USAM-SPS beat both the per-$\rho$ tuned constant learning rate and Cosine Annealing at every radius.
2. **Robustness at large $\rho$.** Cosine Annealing degrades sharply as $\rho$ grows (CIFAR-100, $\rho = 0.4$: USAM Cosine drops to 86.52, SAM Cosine to 84.61), while the Polyak Scheduler stays above 90.7 in both columns.

Full CIFAR-10 / ResNet-20 results and the no-weight-decay ablation are in Appendix E of the paper.

---

## Citation

If you use this code or build on the method, please cite:

```bibtex
@inproceedings{oikonomou2026adaptive,
  title  = {Adaptive Sharpness-Aware Minimization with a Polyak-type Step Size: A Theory-Grounded Scheduler},
  author = {Oikonomou, Dimitris and Loizou, Nicolas},
  booktitle = {ICML},
  year   = {2025},
}
```

---

## License

Released under the [MIT License](LICENSE).
