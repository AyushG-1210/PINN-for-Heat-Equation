# Physics-Informed Neural Network (PINN) for the 2D Heat Equation

A PyTorch implementation of a Physics-Informed Neural Network (PINN) to solve the 2D transient Heat Equation. This project investigates temporal error propagation in PINNs and identifies **Temporal Error Accumulation** as a key failure mode for long-time integration.

Full analysis and results are documented in the accompanying report: *"Analysis of Temporal Error Propagation in Physics-Informed Neural Networks (PINNs) for the 2D-Heat Equation"*.

---

## Table of Contents

- [Problem Formulation](#problem-formulation)
- [Project Structure](#project-structure)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Training Details](#training-details)
- [Results](#results)
- [Known Limitations](#known-limitations)
- [Proposed Improvements](#proposed-improvements)
- [References](#references)

---

## Problem Formulation

The governing equation is the 2D Heat Equation:

```
∂u/∂t = α (∂²u/∂x² + ∂²u/∂y²)
```

where `u(x, y, t)` is the temperature field and `α` is the thermal diffusivity.

The domain is a unit square `[0,1] × [0,1]` over non-dimensional time `t ∈ [0, 1]`.

**Boundary Conditions:** Dirichlet zero (`u = 0`) on all four edges.

**Initial Condition:** `u(x, y, 0) = sin(πx) · sin(πy)`

**Exact (Analytical) Solution:**

```
u(x, y, t) = exp(-2π²αt) · sin(πx) · sin(πy)
```

### Non-dimensionalization

The equation is non-dimensionalized using characteristic length `Lc = 1.0 m` and physical diffusivity `α_phys = 2.1 × 10⁻⁵ m²/s`. This maps the input domain to unit intervals `[0, 1]`, stabilizing gradient descent and preventing vanishing/exploding gradients from raw physical constants.

---

## Project Structure

```
.
├── pinn_heat_eqn.py     # Main training script
├── model.pt             # Saved model weights (generated after training)
└── README.md
```

---

## Architecture

The model is a fully connected network (FCN) with the following structure:

| Layer | Size |
|---|---|
| Input | 3 neurons `[x, y, t]` |
| Hidden (×4) | 64 neurons each |
| Output | 1 neuron `[u(x, y, t)]` |

- **Activation Function:** SiLU (Swish) — chosen for smooth, continuous second-order differentiability, which is required for computing the Laplacian via automatic differentiation. ReLU is unsuitable here as its second derivative is zero almost everywhere.
- **Weight Initialization:** Xavier normal for weights, zeros for biases.
- **Precision:** `torch.float64` (double precision) for numerical stability.

---

## Installation

**Requirements:**

- Python 3.8+
- PyTorch (with CUDA support recommended)
- NumPy
- Matplotlib

Install dependencies:

```bash
pip install torch numpy matplotlib
```

A CUDA-capable GPU is recommended. The script automatically falls back to CPU if CUDA is unavailable.

---

## Usage

Run the training script directly:

```bash
python pinn_heat_eqn.py
```

Training will run for 15,000 epochs. Plots of 3D spatiotemporal predictions and the relative L2 error are displayed every 1,000 iterations. The final trained model is saved as `model.pt`.

To load the saved model for inference:

```python
import torch
from pinn_heat_eqn import FCN

model = FCN(input=3, output=1, hidden=64, layers=4)
model.load_state_dict(torch.load('model.pt'))
model.eval()
```

---

## Training Details

### Loss Function

The composite loss is a weighted sum of three terms:

```
L = λ_IC · L_IC  +  λ_BC · L_BC  +  λ_PHYS · L_PDE
```

| Component | Symbol | Weight |
|---|---|---|
| Initial Condition | `λ_IC` | 3 |
| Boundary Condition | `λ_BC` | 1 |
| Physics Residual (PDE) | `λ_PHYS` | 12 |

The higher physics weight enforces PDE satisfaction more strongly relative to boundary and initial conditions.

### Sampling Strategy

| Point Type | Count |
|---|---|
| Collocation (physics residual) | 8,000 |
| Initial condition (`t = 0`) | 1,000 |
| Boundary condition | 200 |

Collocation and boundary points are sampled randomly over the domain each run.

### Optimizer & Scheduler

- **Optimizer:** Adam, `lr = 1e-3`
- **Scheduler:** `ReduceLROnPlateau` — reduces learning rate by factor 0.1 if loss does not improve for 320 consecutive steps

### Automatic Differentiation

Spatial and temporal derivatives are computed exactly using PyTorch's `torch.autograd.grad`, applying the chain rule through the network graph. This avoids the truncation errors introduced by finite difference or finite element discretization.

---

## Results

After 15,000 training epochs:

| Metric | Value |
|---|---|
| Total Loss | ~4.89 × 10⁻³ |
| IC Loss | ~1.32 × 10⁻³ |
| BC Loss | ~6.08 × 10⁻⁴ |
| Physics Loss | ~2.68 × 10⁻⁵ |

The 3D surface plots show physically correct thermal decay: a hot central region cools and the temperature field approaches equilibrium by `t = 1.0`, consistent with the analytical solution.

---

## Known Limitations

### Temporal Error Accumulation

The dominant failure mode of this model is **Temporal Error Accumulation**. The relative L2 error grows by a factor of ~10⁷ across the time domain (from ~10⁻¹ at `t = 0` to ~10⁶ at `t = 1.0`).

**Root Cause:** PINNs treat the entire space-time domain as a single global optimization problem. The optimizer cannot enforce causal ordering — small residual errors at early time steps are propagated and amplified forward in time, accumulating into large errors at later steps. The physics loss at later time steps is effectively overshadowed by the gradient contributions from already-accumulated errors.

This is a known structural limitation of vanilla PINNs for long-time integration and is not specific to this implementation.

---

## Proposed Improvements

1. **Temporal Domain Decomposition (TDD):** Divide `[0, T]` into sub-intervals `[t_i, t_{i+1}]` and solve each sequentially, using the solution at `t_i` as the initial condition for the next segment. This resets accumulated error to zero at each interface and is the most impactful fix for this failure mode.

2. **Adaptive Loss Weighting:** Replace static `λ` values with dynamic weights that evolve during training (e.g., based on gradient magnitudes or residual history). This allows the optimizer to respond to shifting loss landscapes rather than applying a fixed global balance.

---

## References

1. Source code: [github.com/AyushG-1210/PINN-for-Heat-Equation](https://github.com/AyushG-1210/PINN-for-Heat-Equation)
2. Ghaderi et al., "Equation Discovery, Parametric Simulation, and Optimization Using the Physics-Informed Neural Network (PINN) Method for the Heat Conduction Problem." arXiv:2510.25925 (2025).
3. Almusallam et al., "Physics-informed neural networks for solving the heat equation in thermal engineering." *Int. J. Tech. Phys. Probl. Eng* 17, no. 1 (2025): 375–382.
4. Lenau, A., Dimiduk, D. & Niezgoda, S.R., "Importance of Hyper-Parameter Optimization During Training of Physics-Informed Deep Learning Networks." *Integr Mater Manuf Innov* 14, 115–135 (2025). https://doi.org/10.1007/s40192-025-00394-6