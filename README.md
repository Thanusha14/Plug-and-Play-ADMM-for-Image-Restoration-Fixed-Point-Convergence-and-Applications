# Plug-and-Play-ADMM-for-Image-Restoration-Fixed-Point-Convergence-and-Applications
Implementation of Plug-and-Play ADMM for image restoration, focusing on fixed-point convergence analysis and applications in image denoising and deblurring. The framework integrates advanced denoisers within the ADMM optimization process to achieve high-quality image reconstruction.

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
