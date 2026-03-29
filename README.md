# HypDiff: Stable Hyperbolic Diffusion for General Graph Generation

This repository is a refined and stabilized fork of [RingBDStack/HypDiff](https://github.com/RingBDStack/HypDiff). It implements **Hyperbolic Diffusion Models** specifically designed for generating complex graphs with hierarchical structures.

## Why this Fork?

The original HypDiff introduces a powerful framework for graph generation in hyperbolic space. However, diffusion on manifolds like the **Poincaré Ball** often faces challenges with numerical stability (e.g., points "escaping" the manifold boundary or NaN gradients).

### Key Enhancements in this Version:
* **Numerical Stability & Clipping:** Added robust handling for hyperbolic operations (Exponential and Logarithmic maps) to prevent numerical instability near the Poincaré ball boundary.
* **Modern Framework Compatibility:** Refactored the codebase to be fully compatible with recent versions of **PyTorch** and **Geoopt**, fixing deprecated calls from the original source.
* **Gradient Flow Optimization:** Implemented specific constraints and projection steps to ensure stable training during the reverse diffusion process.
* **Bug Fixes:** Resolved issues in the manifold-aware SDE/ODE solvers that caused instability in long-run sampling.

---

## Overview

**HypDiff** leverages the exponential volume growth of **Hyperbolic Geometry** to capture the hierarchical and power-law properties of real-world networks (social, biological, and citation graphs) more effectively than standard Euclidean models.

### Core Concepts
* **Manifold:** Poincaré Ball Model.
* **Metric:** Hyperbolic metric that naturally preserves tree-like structures.
* **Process:** Riemannian Diffusion that maintains data within the manifold boundaries throughout the forward and reverse steps.

---

## Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/IritKogan/HypDiff.git](https://github.com/IritKogan/HypDiff.git)
    cd HypDiff
    ```

2.  **Set up the environment:**
    It is highly recommended to use a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: This version requires `geoopt` for manifold-aware tensors.*

---

## Usage

### Training
To train the model on a graph dataset with stabilized hyperbolic parameters:
```bash
python train.py --dataset [DATASET_NAME] --manifold poincare --lr 1e-4 --epochs 200
```
### Sampling & Generation
To generate new graph structures using a trained checkpoint:
```bash
python sample.py --checkpoint checkpoints/stable_model.pt --n_samples 50
```
---

## Technical Implementation

This fork ensures that for any latent point **x** in the Poincaré Ball, the condition **||x|| < 1** is strictly maintained through:

* **Epsilon-Clamping:** Ensuring points do not hit the absolute boundary (radius = 1) where gradients become infinite.
* **Manifold Projection:** Regular projection steps (via `geoopt`) after gyrovector additions to maintain structural integrity.
* **Numerical Safety:** Implementation of stable Log-Sum-Exp and hyperbolic distance functions to avoid overflow.

---

## Credits & Acknowledgments

This project is based on the original research and code by **RingBDStack**.

* **Original Repository:** [RingBDStack/HypDiff](https://github.com/RingBDStack/HypDiff)
* **Authors of this Fork:** [Irit Kogan](https://github.com/IritKogan) & [Rotem Maravent](https://github.com/rotemmaravent)

---

## Contributing
If you encounter any issues with the hyperbolic stability or have suggestions for further optimization, feel free to open an issue or submit a pull request!
