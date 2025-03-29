# Bayesian Factor-Augmented VAR Models for Monetary Policy Spillover Analysis

This repository contains MATLAB code that implements Bayesian Factor-Augmented VAR (FAVAR) models and its quantile counterpart (QFAVAR) for analyzing monetary policy spillovers using high-frequency (24‑hour call rate) data from multiple countries.

The code employs Markov Chain Monte Carlo (MCMC) estimation methods and computes Generalized Impulse Response Functions (GIRFs) to assess the dynamic responses of the system. It is particularly designed to study how monetary policy in one country may affect call rates—and, by extension, financial conditions—in other countries.

---

## Contents

- **FAVAR_GIRFs.m**: MATLAB script implementing the standard Bayesian FAVAR model with GIRFs.
- **QFAVAR_GIRFs.m**: MATLAB script implementing the quantile version of the Bayesian Factor-Augmented VAR model.
- **call_rate_data.mat**: (Example) Data file that should include:
  - `x` – a T-by-n matrix of 24‑hour call rate data (e.g., 14 countries)
  - `dates` – a T-by-1 vector of dates
  - `country_names` – a 1-by-n cell array with country names
- **functions/**: Folder containing the required helper functions:
  - `randn_gibbs.m`
  - `horseshoe_prior.m`
  - `FFBS.m`
  - `mlag2.m`
  - `armairf.m`
  - (Other supporting files as needed)

---

## Overview

This project implements two closely related models:
1. **FAVAR_GIRFs**: A standard Bayesian Factor-Augmented VAR model, where the measurement equation relates observed data to latent factors and a global factor, and the state equation follows a VAR process.
2. **QFAVAR_GIRFs**: An extension that allows for different quantile levels in the measurement equation, useful for assessing heterogeneity in the responses (e.g., the behavior at the lower, median, and upper tails).

Both scripts use MCMC techniques to sample from the posterior distribution of the model parameters and latent factors. The code then computes generalized impulse response functions to map the shocks in the latent state onto the observable variables.

---

## Requirements

- **MATLAB** (R2018b or later is recommended)
- The helper functions in the `functions/` folder must be in the MATLAB path.
- The data file `call_rate_data.mat` (or your own data in a similar format) must be placed in the `data/` folder.
- MATLAB toolboxes: Basic Statistics and Machine Learning Toolbox (for functions like `pca` and `gamrnd`) are recommended.

---

## Data Preparation

Your data file (`call_rate_data.mat`) should include:
- **x**: A T-by-n matrix of call rate data (e.g., 24‑hour call rates for 14 countries).
- **dates**: A T-by-1 vector of date strings or numeric date identifiers.
- **country_names**: A 1-by-n cell array of country names.

The global factor is automatically constructed as the cross‑country average of `x`.

---

## How to Run

1. **Setup the Environment:**
   - Ensure that the `functions` and `data` folders are in your MATLAB path.
   - Verify that your data file is in the correct format.

2. **Run the Script:**
   - Open MATLAB.
   - In the Command Window, navigate to the folder containing the scripts.
   - Run the desired script (e.g., `FAVAR_GIRFs.m` or `QFAVAR_GIRFs.m`) by typing its name.
   
3. **MCMC Output and Plots:**
   - The scripts will output progress updates in the MATLAB console.
   - Once completed, the MCMC draws and impulse response functions are saved (e.g., as `FAVAR_GIRFs_results.mat`).
   - Several plots (factor estimates, IRFs, GIRFs) will be generated and saved to files (e.g., `FAVAR_IRF_state_eq.jpg`).

---

## Parameter Settings

Within the scripts you can adjust:
- **Model Settings:** Number of latent factors (`r`), number of lags (`p`), inclusion of intercepts, and whether to use a global factor.
- **MCMC Settings:** Number of draws to store (`nsave`), burn-in period (`nburn`), thinning parameter (`nthin`), and iteration display frequency (`iter`).
- **Hyperparameters:** Settings for the horseshoe priors and error variance specifications.

Be sure to adjust these according to the specifics of your application and data.

---

## Function Dependencies

The scripts call several custom functions:
- **randn_gibbs.m**: For drawing from the conditional posterior of regression coefficients.
- **horseshoe_prior.m**: For updating local and global shrinkage parameters.
- **FFBS.m**: Implements the forward-filtering backward-smoothing algorithm for state-space models.
- **mlag2.m**: Constructs lag matrices for VAR estimation.
- **armairf.m**: Computes impulse response functions based on ARMA representations.

Ensure these functions are available and correctly implemented.

---

## Contact & License

If you have any questions or encounter issues, please contact [Your Name] at [your.email@example.com].

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

---

*Last updated: 29/03/2025*
