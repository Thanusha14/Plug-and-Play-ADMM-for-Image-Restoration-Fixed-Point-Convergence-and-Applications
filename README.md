# Plug-and-Play-ADMM-for-Image-Restoration-Fixed-Point-Convergence-and-Applications
This project implements a flexible Plug-and-Play Alternating Direction Method of Multipliers (PnP-ADMM) framework in Python for solving inverse imaging problems such as image denoising and deblurring. The framework integrates advanced image denoisers into the optimization process without explicitly defining a prior model.

The Plug-and-Play approach replaces traditional regularization terms with powerful off-the-shelf denoisers, enabling high-quality image reconstruction while avoiding the need for manually designed image priors. This modular formulation allows different denoisers to be incorporated seamlessly, making the framework adaptable to various image restoration tasks.

---

## Features

- Supports **image denoising** and **Gaussian deblurring**
- Plug-and-play integration of multiple denoisers:
  - Total Variation (TV)
  - Non-Local Means (NLM)
  - BM3D
- Works with both **grayscale and color images**
- Quantitative evaluation using **PSNR**
- Adaptive ADMM parameter updates for improved convergence

---

## Method Overview

The PnP-ADMM algorithm alternates between:

1. A **data fidelity step**, solved analytically (or in the Fourier domain for deblurring)
2. A **denoising step**, where a classical denoiser is plugged into the optimization loop
3. **Dual variable updates** following the ADMM framework

This modular design makes it easy to replace or extend denoisers without modifying the core algorithm.

---

## Technologies Used

- Python
- NumPy
- scikit-image
- BM3D
- Matplotlib

---

## Usage

1. Set the input image path inside `PNP_admm.py`
2. Run the script:

```bash
python PNP_admm.py
