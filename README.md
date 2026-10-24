# Neural Networks and Deep Learning — From Sequences to Images: TimesNet Variants for Weather Forecasting

Project developed with R. Pesce and M. Saccon, Awarded 30.

## Overview

This project is a **from-scratch PyTorch reimplementation of TimesNet** and a systematic study of its architectural variants on a long multi-year weather benchmark.

TimesNet turns forecasting into a computer vision problem. It finds the dominant periods of a time series with the FFT, reshapes the 1D sequence into 2D tensors that separate intra-period and inter-period variation, and processes them with Inception-based 2D convolutions. We compare four TimesNet variants against **PatchTST**, an attention-based model with no hardcoded periodicity assumption.

## Architectures

| Model | Description |
|-------|-------------|
| **Net1bis** | Base design: a linear layer produces a rough prediction, then a stack of TimesBlocks refines the extended sequence |
| **Net1** | Feature extraction first: TimesBlocks run on the input, then a linear layer projects into the future |
| **Net2** | Weight-sharing: one TimesBlock and one linear layer are applied iteratively, which reduces the parameter count |
| **Net3** | Cross-channel: each channel goes through its own TimesBlock pipeline before being merged |
| **Net4** | PatchTST baseline: self-attention over patches of the input sequence |

All TimesNet variants share the same building blocks: FFT-based period discovery, 1D-to-2D reshaping, a multi-scale `InceptionBlock` (kernels 1, 3, 5, 7) and a residual `TimesBlock`. They are trained with MSE loss and Adam, and hyperparameters are tuned with **Optuna**.

## Dataset

Weather data from the Max Planck Institute for Biogeochemistry's Jena Beutenberg station: 21 features, 10-minute resolution, from 2008 to 2025 (2026 is used as the test set).

* Input: 96 timesteps. Output: the following 96 timesteps (16 hours).
* Sliding windows with several shifts to enlarge the dataset (63098 training and 15775 validation samples).
* Per-sample normalization, de-normalized before computing the loss.
* Optional **PCA** to 9 components, which retain about 98% of the variance. Scaler and PCA are fitted on the training set only, to avoid leakage.

## Main results

* **Net1 vs Net1bis**: no significant difference, so the two are treated as equivalent and Net1 is preferred because it has fewer parameters.
* **Best model**: Net1 is best in the single-channel (temperature only) setting. Net2 is best in the multi-channel setting, which is the more relevant one given the correlation between features.
* **PCA**: training on 9 PCA components gives a lower validation error than using all 21 features, and learning is about 1.29x faster. Our interpretation is that PCA prevents the network from spending capacity on linear correlations between features.
* **Prediction length**: the error per timestep grows with the prediction horizon.
* **Input length**: increasing the input length does not improve performance.
* **Weighted loss**: weighting the first predicted timestep more heavily gives no gain.
* **Net3**: validation loss was about twice that of the other architectures during tuning, so it was not trained further.
* **Error growth**: the standard deviation of the prediction error grows approximately as the square root of the prediction time, like Brownian motion.

| Test loss | Net 1 | Net 2 | Net 4 (PatchTST) |
|-----------|-------|-------|------------------|
| Single-channel | 6.233 | 8.056 | 8.887 |
| Multi-channel (only T) | 7.090 | 6.049 | 7.918 |
| Multi-channel (total) | 14860 | 11770 | — |

## Repository structure

* Code for the models, training and analyses (notebooks and scripts).
* The report as a PDF (add it if you upload it).

## How to run

Install the required dependencies (`torch`, `numpy`, `scikit-learn`, `optuna`, `matplotlib`) and run the notebooks using Jupyter.

## References

* H. Wu et al., *TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis*, ICLR 2023.
* Y. Nie et al., *A Time Series is Worth 64 Words: Long-term Forecasting with Transformers* (PatchTST), ICLR 2023.
* C. Szegedy et al., *Going Deeper with Convolutions*, CVPR 2015.
* T. Akiba et al., *Optuna: A Next-generation Hyperparameter Optimization Framework*, KDD 2019.
