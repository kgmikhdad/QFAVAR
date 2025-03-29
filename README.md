# Bayesian FAVAR & QFAVAR Models for Monetary Policy Spillover Analysis

This repository contains two MATLAB implementations that use Bayesian estimation to analyze monetary policy spillovers using high‑frequency (24‑hour call rate) data from multiple countries. The codes implement two related models:

1. **FAVAR_GIRFs.m** – A standard Bayesian Factor-Augmented VAR (FAVAR) model with Generalized Impulse Response Functions (GIRFs).
2. **QFAVAR_GIRFs.m** – A quantile extension of the FAVAR model (QFAVAR) that allows the researcher to analyze the behavior of the system at different quantile levels.

Both models incorporate a global factor (constructed as the cross‑country average call rate) along with latent factors extracted from the data, and both use MCMC techniques to sample from the posterior distribution of the model parameters.

---

## Contents

- **FAVAR_GIRFs.m**: MATLAB script for estimating the standard Bayesian FAVAR model with GIRFs.
- **QFAVAR_GIRFs.m**: MATLAB script for estimating the quantile Bayesian Factor-Augmented VAR model.
- **call_rate_data.mat**: Example data file containing:
  - `x` – a T-by-n matrix of 24‑hour call rate data (e.g., for 14 countries)
  - `dates` – a T-by‑1 vector of dates
  - `country_names` – a 1-by‑n cell array of country names
- **functions/**: Folder with required helper functions:
  - `randn_gibbs.m`
  - `horseshoe_prior.m`
  - `FFBS.m`
  - `mlag2.m`
  - `armairf.m`
  - (Other supporting functions as needed)

---

## Overview

### FAVAR_GIRFs.m

- **Model Description:**  
  Implements a Bayesian FAVAR model in which the measurement equation links observed call rate data to latent factors and a global factor. The state equation is modeled as a VAR process with normally distributed shocks.
  
- **Key Features:**
  - MCMC estimation with Gibbs sampling.
  - Horseshoe priors for parameter shrinkage.
  - Computation of generalized impulse response functions (GIRFs) to map latent shocks to observable call rates.
  - Identification restrictions via normalization of factor loadings.

### QFAVAR_GIRFs.m

- **Model Description:**  
  Extends the FAVAR framework to a quantile context. In addition to a global factor, the model extracts quantile‐specific latent factors to examine the behavior of call rates across different quantiles (e.g., 10%, 50%, 90%).
  
- **Key Features:**
  - MCMC estimation using a quantile-specific measurement equation.
  - Adaptation of the Asymmetric Laplace Distribution (ALD) for quantile regression errors.
  - Separate extraction of latent factors for each quantile.
  - Computation of generalized impulse responses (GIRFs) for each quantile level.

---

## Requirements

- **MATLAB** (R2018b or later recommended)
- MATLAB toolboxes: Statistics and Machine Learning Toolbox (for PCA, gamma random number generation, etc.)
- The helper functions provided in the `functions/` folder must be in the MATLAB path.
- Data file (`call_rate_data.mat`) must be placed in the `data/` folder.

---

## Data Preparation

Your data file should include the following variables:
- **x**: A T-by-n matrix of 24‑hour call rate data (e.g., for 14 countries).
- **dates**: A T-by‑1 vector of dates (formatted as strings or numbers).
- **country_names**: A 1-by‑n cell array of country names.

The global factor is constructed automatically as the cross‑country average of `x`.

---

## How to Run

1. **Setup the Environment:**
   - Place the `FAVAR_GIRFs.m`, `QFAVAR_GIRFs.m`, and the `functions/` and `data/` folders in the same root directory.
   - Add the `functions/` and `data/` folders to your MATLAB path.

2. **Run the Scripts:**
   - Open MATLAB and navigate to the repository folder.
   - To run the standard model, type:
     ```matlab
     FAVAR_GIRFs
     ```
   - To run the quantile model, type:
     ```matlab
     QFAVAR_GIRFs
     ```

3. **Output:**
   - Both scripts will print progress updates in the MATLAB console.
   - MCMC draws and impulse response functions will be saved (e.g., `FAVAR_GIRFs_results.mat` and `QFAVAR_GIRFs.mat`).
   - Several figures (factor estimates, IRFs, GIRFs) will be generated and can be saved as image files.

---

## Parameter Settings

Both codes allow you to adjust several parameters:
- **Model Settings:**  
  - `r`: Number of latent factors.
  - `p`: Number of lags in the VAR (state equation).
  - `interX`, `interF`: Whether to include intercepts in the measurement or state equations.
  - `incldg`: Whether to include a global factor.
  - For QFAVAR: `quant` specifies the quantile levels (e.g., `[0.1, 0.5, 0.9]`).

- **MCMC Settings:**  
  - `nsave`: Number of posterior draws to store.
  - `nburn`: Number of initial draws to discard (burn-in).
  - `nthin`: Thinning parameter.
  - `iter`: Frequency of progress messages.

- **Priors and Hyperparameters:**  
  - Settings for horseshoe priors on measurement and VAR coefficients.
  - Error variance specifications.

Adjust these parameters as needed for your data and research question.

---

## Function Dependencies

Both scripts depend on several custom functions which should be placed in the `functions/` folder:
- **randn_gibbs.m**: Draws from the conditional posterior of regression coefficients.
- **horseshoe_prior.m**: Updates local and global shrinkage parameters.
- **FFBS.m**: Implements the forward-filtering backward-smoothing algorithm for state-space models.
- **mlag2.m**: Creates lag matrices for VAR estimation.
- **armairf.m**: Computes impulse response functions from ARMA representations.

Ensure these functions are available and correctly implemented in your MATLAB path.

---


