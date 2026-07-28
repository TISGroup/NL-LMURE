# NL-LMURE

**Unbiased and Nonlocal Linear Regression for Video Denoising under Multiplicative Noise**

[![Journal](https://img.shields.io/badge/Journal-JMIV-blue)](https://link.springer.com/journal/10851)
<!-- [![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE) -->

PyTorch implementation of the paper:

> **Unbiased and Nonlocal Linear Regression for Video Denoising under Multiplicative Noise**  
> *Zipei Yan¹, Ting Wang², Chao Wang²\*, Jizhou Li¹\**  
> ¹ The Chinese University of Hong Kong &emsp; ² Southern University of Science and Technology  
> *Journal of Mathematical Imaging and Vision (JMIV), 2026*

---

## Overview

NL-LMURE is a **distribution-agnostic** nonlocal patch-based framework for video denoising under multiplicative noise $y = w \cdot x$ with $\mathbb{E}[w] = 1$, $\mathrm{Var}(w) = \tau^2$. Unlike prior methods that assume a specific noise PDF (e.g., Gamma), our framework derives the **Linear Multiplicative Unbiased Risk Estimator (LMURE)** — an unbiased estimate of the true MSE for *any* linear denoising function under the moment assumptions alone (Theorem 3, Eq. 9). By plugging a nonlocal linear functional form $\mathbf{F}(\mathbf{Y}) = \mathbf{Y}\mathbf{\Theta}$ into LMURE, we obtain a **closed-form solution** that frames denoising as a principled ridge regression (Theorem 4, Eq. 13):

$$\mathbf{\Theta}^* = \mathbf{I}_k - \frac{\tau^2}{\tau^2 + 1}(\mathbf{Y}^\top\mathbf{Y})^{-1}\mathbf{D}, \quad \mathbf{D} = \mathrm{diag}(\|\mathbf{Y}_{:,1}\|^2, \ldots, \|\mathbf{Y}_{:,k}\|^2)$$

### Key Contributions

- **Distribution-agnostic**: Applicable to *any* multiplicative noise satisfying $\mathbb{E}[w]=1$, $\mathrm{Var}(w)=\tau^2$ — including Gamma, Normal, Lognormal, Wald, and Beta Prime (Table 1 in the paper)
- **Closed-form optimization**: No iterative solver needed; the optimal regression weights admit a direct matrix solution
- **Unbiased patch matching**: A dedicated matching criterion (Theorem 1) that correctly recovers pairwise patch similarity under multiplicative noise (Fig. 9 in the paper)
- **Built-in variance estimation**: Estimates $\tau^2$ from the noisy observation via RANSAC-regularized linear regression (Section 4.5, Eq. 15)
- **Two-step internal adaptation**: A pilot estimate refines a second pass using only the matching indices, suppressing residual noise without new parameter tuning (Section 4.6)

---

## Table of Contents

- [Installation](#installation)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
  - [1. Download Data](#1-download-data)
  - [2. Simulate Noisy Data](#2-simulate-noisy-data)
  - [3. Run Denoising](#3-run-denoising)
- [Method](#method)
- [Supported Noise Distributions](#supported-noise-distributions)
- [Hyperparameters](#hyperparameters)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)
- [Contact](#contact)

---

## Installation

```bash
git clone https://github.com/TISGroup/NL-LMURE.git
cd NL-LMURE
conda create -n nl_lmure python=3.10
conda activate nl_lmure
pip install -r requirements.txt
```

**Core dependencies**: PyTorch, scikit-image, scikit-learn, einops, NumPy, SciPy, gdown.

---

## Project Structure

```
NL-LMURE/
├── denoise.py               # Core algorithm (block matching, LMURE, aggregation)
├── demo.py                  # Demo with PSNR/SSIM evaluation
├── simulate_noisy_data.py   # Multiplicative noise simulator (5 distributions)
├── download_data.py         # GoPro 540p dataset downloader
├── requirements.txt
├── README.md
└── data/
    ├── clean/               # Clean video frames (.tif)
    └── noisy/
        └── gamma/
            ├── v=0.01/      # τ = 0.1  (L = 100 looks)
            ├── v=0.04/      # τ = 0.2  (L =  25 looks)
            ├── v=0.25/      # τ = 0.5  (L =   4 looks)
            └── v=1/         # τ = 1.0  (L =   1 look)
```

---

## Quick Start

### 1. Download Data

We use the GoPro 540p dataset (from [FastDVDnet](https://github.com/m-tassano/fastdvdnet)), consisting of 4 RGB video sequences: *hypersmooth*, *motorbike*, *rafting*, *snowboard*.

```bash
python download_data.py
```

Clean data is saved as multi-frame `.tif` stacks in `./data/clean/`.

### 2. Simulate Noisy Data

The noise simulator supports all five multiplicative distributions studied in the paper (Section 5.1, Table 1). By default it generates **Gamma** noise at four noise levels:

$$v \in \{0.01, 0.04, 0.25, 1\} \quad\longleftrightarrow\quad \tau \in \{0.1, 0.2, 0.5, 1.0\}$$

corresponding to $L \in \{100, 25, 4, 1\}$-looks in coherent imaging systems.

```bash
python simulate_noisy_data.py
```

To switch noise types, edit `noise_type` in `simulate_noisy_data.py` to any of
`"gamma"`, `"normal"`, `"lognormal"`, `"wald"`, or `"beta_prime"`.

Noisy data is saved to `./data/noisy/{noise_type}/v={v}/noisy.tif`.

### 3. Run Denoising

```bash
python demo.py
```

The demo loads a clean/noisy pair, runs the full two-step NL-LMURE pipeline, and reports frame-wise **PSNR** and **SSIM** before and after denoising. Hyperparameters are pre-configured per noise level (see [Hyperparameters](#hyperparameters) below).

---

## Method

NL-LMURE operates within the nonlocal framework organized into four steps (Section 3.2 of the paper):

### Step 0 — Frame Grouping

We first select the $T$ most temporally similar frames using the **unbiased matching criterion** $L$ derived in Theorem 1, which correctly estimates the true $\ell_2$ discrepancy between underlying clean patches under *any* multiplicative noise satisfying (2).

### Step 1 — Unbiased Intra-Block Regression (LMURE)

Within each $B \times B$ spatiotemporal block, we extract non-overlapping $P_1 \times P_1$ patches and find the $K_1$ nearest neighbors via block matching. The stacked noisy patches $\mathbf{Y} \in \mathbb{R}^{m \times k}$ (where $m = C \cdot P_1^2$) are denoised by the linear operator $\hat{\mathbf{X}} = \mathbf{Y}\mathbf{\Theta}^\star$, with $\mathbf{\Theta}^\star$ given by the **LMURE closed-form solution** (Theorem 4, Eq. 13):

$$\mathbf{\Theta}^\star = \mathbf{I}_k - \frac{\tau^2}{\tau^2 + 1} (\mathbf{Y}^\top\mathbf{Y})^{-1} \mathbf{D},$$

where $\mathbf{D} = \mathrm{diag}(\Vert\mathbf{Y}_{:,1}\|^2, ..., \|\mathbf{Y}_{:,k}\Vert^2)$. This is derived by minimizing the **unbiased risk estimator** (Corollary 3.1, Eq. 11) that accurately estimates $\mathbb{E}\|\mathbf{X} - \hat{\mathbf{X}}\|_F^2$ without ground truth. Patches are aggregated via weighted-average reprojection with weights $w_j = 1 / \|\mathbf{\Theta}_{:,j}\|^2$ (Section 3.2).


### Step 2 — Internal Adaptation (Guided Regression)

Following the two-step design in NL-Means, BM3D, and NL-Ridge, we re-apply the pipeline using the Step 1 output $\hat{\mathbf{X}}$ as a **pilot estimate** (Section 4.6). The noise variance is re-estimated from $\hat{\mathbf{X}}$ via the RANSAC-based linear regression (Section 4.5, Eq. 15), and patch-matching indices are determined on the cleaner pilot. The second-stage weights are the least-squares solution $\mathbf{\Theta}^\star = \mathbf{Y}^\dagger \hat{\mathbf{X}}$, producing the final denoised output $\tilde{\mathbf{X}} = \mathbf{Y}\mathbf{\Theta}^\star$.

The full pipeline is summarized in **Algorithm 1** of the paper.

---

## Supported Noise Distributions

All noise models follow the multiplicative form $y = w \cdot x$ with $\mathbb{E}[w] = 1$, $\mathrm{Var}(w) = \tau^2$ (Eqs. 1–2 in the paper). The five distributions characterized in **Table 1** of the paper are:

| Distribution  | $w \sim$                         | `noise_type`    | Parameter mapping            |
|:--------------|:---------------------------------|:----------------|:-----------------------------|
| Gamma         | $\mathrm{Gamma}(\alpha,\beta)$   | `"gamma"`       | $\alpha = \beta = 1/\tau^2$  |
| Gaussian      | $\mathcal{N}(1, \tau^2)$         | `"normal"`      | $\mu = 1$                    |
| Log-normal    | $\mathrm{LogNormal}(\mu, \sigma^2)$ | `"lognormal"` | $\mu = -\sigma^2/2$          |
| Wald (Inv. Gaussian) | $\mathrm{Wald}(1, \lambda)$ | `"wald"`        | $\lambda = 1/\tau^2$         |
| Beta Prime    | $\mathrm{BP}(\alpha, \beta)$     | `"beta_prime"`  | $\alpha=1+2/\tau^2,\ \beta=2+2/\tau^2$ |

The Gamma case is particularly relevant for coherent imaging (SAR, ultrasound, OCT), where $\tau = 1/\sqrt{L}$ and $L$ is the number of looks.

---

## Hyperparameters

Hyperparameters follow **Table 2** of the paper. The spatial block size is fixed at $B = 37$ (Section 5.1).

| Parameter    | Description                                      | Paper notation | Typical       |
|:-------------|:-------------------------------------------------|:---------------|:--------------|
| `temp_depth` | Temporal depth (similar frames per block)        | $T$            | 2–4           |
| `block_size` | Spatial search window (odd)                      | $B$            | 37            |
| `patch_size1`| Patch size for Step 1 matching                   | $P_1$          | 7–9           |
| `topk1`      | Number of similar patches in Step 1              | $K_1$          | 16–22         |
| `patch_size2`| Patch size for Step 2 matching                   | $P_2$          | 5–7           |
| `topk2`      | Number of similar patches in Step 2              | $K_2$          | 22            |
| `variance`   | Noise variance $\tau^2$ (set `None` to auto-estimate) | $\tau^2$  | 0.01–1.0      |

**Preset configurations** used in `demo.py` (matching the paper's experiments):

| $\tau^2$ | $\tau$ | $L$ (looks) | $T$ | $P_1$ | $K_1$ | $P_2$ | $K_2$ |
|:------:|:----:|:---------:|:---:|:---:|:---:|:---:|:---:|
| 0.01   | 0.1  | 100       | 4   | 7   | 22  | 5   | 22  |
| 0.04   | 0.2  | 25        | 3   | 7   | 22  | 5   | 22  |
| 0.25   | 0.5  | 4         | 2   | 9   | 16  | 7   | 22  |
| 1.0    | 1.0  | 1         | 2   | 9   | 16  | 7   | 22  |

---

## Citation

If you find this work useful, please cite:

```bibtex
@article{yan2026nl_lmure,
  title   = {Unbiased and Nonlocal Linear Regression for Video Denoising under Multiplicative Noise},
  author  = {Yan, Zipei and Wang, Ting and Wang, Chao and Li, Jizhou},
  journal = {Journal of Mathematical Imaging and Vision},
  year    = {2026},
}
```

---

## Acknowledgments

This code builds upon the [NL-Ridge](https://github.com/sherbret/NL-Ridge) framework (Herbreteau & Kervrann, *SIAM J. Imaging Sci.*, 2025). The GoPro dataset comes from [FastDVDnet](https://github.com/m-tassano/fastdvdnet) (Tassano et al., *CVPR* 2020).

---

## Contact

For questions, please contact: **jzli AT ee DOT cuhk DOT edu DOT hk**
