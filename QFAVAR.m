%% QFAVAR_MPS.m    Bayesian Quantile Factor-Augmented VAR Model for Monetary Policy Spillover
%% (MCMC estimation + Generalized IRFs for each quantile)
%
% The model is of the form
%
%          _       _        _             _    _        _     _          _
%         |  x_{t}  |      |L(tau)  G(tau) |  |F_{t}(tau)|   | e_{t}(tau) |
%         |         |  =   |               |  |          | + |            |
%         |_ g_{t} _|      |_ 0        I  _|  |_  g_{t} _|   |_     0    _|
%
%         _          _           _            _
%        | F_{t}(tau) |         | F_{t-1}(tau) |  
%        |            |  =  Phi |              |  + u_{t},
%        |_   g_{t}  _|         |_  g_{t-1}   _|
%
% where e_{t}(tau) ~ ALD(Sigma(tau)) and u_{t} ~ N(0, Q(tau)). Sigma(tau) 
% and Q(tau) are diagonal covariance matrices and tau is the quantile level.
%
% This updated code implements the methodology for analyzing monetary policy
% spillovers using 24‑hour call rate data from 14 countries.
%
% Assumptions:
%   - The data file "call_rate_data.mat" contains:
%         x            : T-by-14 matrix of 24‑hour call rates,
%         dates        : T-by-1 vector (or cell array) of dates,
%         country_names: 1-by-14 cell array of country names.
%   - A global factor is constructed as the cross‑country average.
%
% Make sure that the functions randn_gibbs.m, horseshoe_prior.m, VBQFA.m,
% FFBS.m, mlag2.m, and armairf.m are in your MATLAB path.
%
% Written by: [Your Name]
% Date: [Current Date]
% =========================================================================

clear all; clc; close all;

%% Add paths for functions and data
addpath('functions')
addpath('data')

%% USER INPUT & MODEL SETTINGS

% Data & model settings for Monetary Policy Spillover analysis
r         = 3;              % Number of latent quantile factors
p         = 2;              % Number of lags in the state equation
quant     = [0.1, 0.5, 0.9];  % Quantiles to estimate
interX    = 1;              % Include intercept in measurement equation
AR1x      = 0;              % (Not used here)
incldg    = 1;              % Include global factor in measurement equation
dfm       = 0;              % 0: estimate QFAVAR; 1: estimate QDFM
standar   = 2;              % Standardize both x and g variables
ALsampler = 1;              % ALD sampler: 1 for Khare & Robert (2012)
var_sv    = 0;              % VAR error variances: 0 (constant)
nhor      = 60;             % Horizon for IRFs, FEVDs, etc.

% MCMC settings
nsave     = 50000;          % Number of draws to store
nburn     = 2000;           % Number of draws to discard
ngibbs    = nsave + nburn;   % Total number of draws
nthin     = 50;
iter      = 20;             % Print progress every "iter" iterations

%% Load 24-Hour Call Rate Data for 14 Countries
% Expected variables in "call_rate_data.mat":
%   - x            : T x 14 matrix of call rates
%   - dates        : T x 1 vector (or cell array) of date strings/numbers
%   - country_names: 1 x 14 cell array of country names
load('call_rate_data.mat');  % Loads x, dates, country_names

[T, n] = size(x);            % T: number of time periods; n should be 14

% Compute lagged call rate data if AR1x is used (here AR1x = 0)
if AR1x
    xlag = lagmatrix(x, 1:p);
else
    xlag = [];
end

% Construct the global factor as the cross-country average call rate
ng = 1;                      % Number of global factors
g = mean(x, 2);

% Set variable names
names  = country_names;      % Country names (1 x n cell array)
namesg = {'Global Call Rate'};
tcode  = dates;              % Time codes

% Number of quantiles
nq = length(quant);

% For the state equation the number of regressors is:
% k = (number of state variables)*p + (optional intercept)
interF = 0;                  % No intercept in the VAR state equation (can be set to 1 if desired)
k = (r*nq + ng)*p + interF;

%% Preliminaries for QFAVAR

% Quantile-specific constants for the Asymmetric Laplace Distribution
k2_sq = 2 ./ (quant .* (1 - quant));
k1    = (1 - 2*quant) ./ (quant .* (1 - quant));
k1_sq = k1.^2;

% Horseshoe prior hyperparameters for measurement equation loadings.
% In this updated model, each country's measurement equation uses
% an intercept, a global loading, and r factor loadings.
load_dim = interX + ng + r;  % Total number of coefficients per equation
lambdaL   = 0.1 * ones(n, load_dim, nq);
tauL      = 0.1 * ones(n, nq);
nuL       = 0.1 * ones(n, load_dim, nq);
xiL       = 0.1 * ones(n, nq);

% Horseshoe prior hyperparameters for VAR coefficients (state equation)
lambdaPhi = 0.1 * ones(r*nq+ng, k);
tauPhi    = 0.1 * ones(r*nq+ng, 1);
nuPhi     = 0.1 * ones(r*nq+ng, k);
xiPhi     = 0.1 * ones(r*nq+ng, 1);

% Choose sampling algorithm for VAR parameters (depending on k and sample size)
est_meth = 1 + double(k > (T - p));

%% Initialize Matrices

xbar    = zeros(T, n, nq);
Lbar    = zeros(n, r, T, nq);   % To store factor loadings for the quantile part
Lbar2   = zeros(n*nq, ng, T);    % To store global loadings
L       = zeros(load_dim, n, nq);  % Measurement equation loadings: rows = [intercept; global; factors]
Sigma   = 0.1 * ones(n, nq);
z       = 0.1 * ones(T, n, nq);

% The state vector F will include the quantile-specific factors and the global factor.
% Its dimension is T x (r*nq + ng)
F       = zeros(T, r*nq + ng);
FL      = zeros(T, n, nq);

% VAR (state equation) matrices
Phi       = 0.1 * ones(k, r*nq + ng);
Omega     = 0.1 * ones(1, r*nq + ng);
Omega_t   = 0.1 * ones(T - p, r*nq + ng);
OMEGA     = 0.1 * ones(r*nq + ng, r*nq + ng, T);
h         = 0.1 * ones(T - p, r*nq + ng);
sig       = 0.1 * ones(r*nq + ng, 1);
Omegac    = zeros((r*nq + ng)*p, (r*nq + ng)*p, T);
Phic      = [Phi(interF+1:end, :)'; eye((r*nq+ng)*(p-1)) zeros((r*nq+ng)*(p-1), r*nq+ng)];
Omegac(1:r*nq+ng, 1:r*nq+ng, :) = repmat(diag(Omega), 1, 1, T);
QL        = ones(load_dim, n, nq);
QPhi      = ones(k, r*nq+ng);

intF      = zeros(T, (r*nq+ng)*p);

%% Factor Extraction

disp('Extracting factors via PCA and VBQFA...');

% Standard PCA extraction (baseline, optional)
[coeff, score, ~] = pca(zscore(x));
fpca = score(:, 1:r);

% Quantile factor extraction using VBQFA.
% Here we assume VBQFA takes as input the (standardized) data matrix x,
% the number of factors to extract, the number of iterations, the quantile,
% and additional options.
fqfa = zeros(T, r, nq);
for iq = 1:nq
    fqfa(:, :, iq) = VBQFA(zscore(x), r, 500, quant(iq), 0, 1);
end

% Combine the quantile factors and the global factor into the state vector F.
% The quantile factors are reshaped into T x (r*nq) and the global factor is appended.
F(:, 1:r*nq) = reshape(fqfa, T, r*nq);
F(:, r*nq + 1) = g;

%% MCMC Storage
F_draws     = zeros(T, r*nq, nsave/nthin);
L_draws     = zeros(load_dim, n, nq, nsave/nthin);
Phi_draws   = zeros(k, r*nq+ng, nsave/nthin);
Sigma_draws = zeros(n, nq, nsave/nthin);
OMEGA_draws = zeros(r*nq+ng, r*nq+ng, T, nsave/nthin);
z_draws     = zeros(T, n, nq, nsave/nthin);

firf_save   = zeros(nhor, r*nq+ng, r*nq+ng, nsave/nthin);
yirf_save   = zeros(nhor, n*nq, r*nq+ng, nsave/nthin);

%% ============================| START MCMC |==============================
format bank;
fprintf('Running QFAVAR MCMC for Monetary Policy Spillover Analysis\n');
fprintf('Iteration 000000\n');
savedraw = 0; tic;

for irep = 1:(nsave + nburn)
    % Display progress every "iter" iterations
    if mod(irep, iter) == 0
        fprintf('Iteration %6d\n', irep);
    end
    
    %% === Measurement Equation: Factor Extraction for Each Quantile ===
    for q = 1:nq
        % Extract quantile-specific factors for the current quantile level
        Fq = F(:, (q-1)*r+1 : q*r);
        
        for i = 1:n   % Loop over countries
            % Build the regressor matrix for the measurement equation:
            % [Intercept, Global Factor, Quantile-Specific Factors]
            F_all = [ones(T, interX), g, Fq];  % Dimension: T x (interX+ng+r)
            select = 1:(interX + ng + r);        % Use all coefficients
            
            F_select = F_all(:, select);
            
            % ----- Step 1: Sample Loadings L -----
            v = sqrt(Sigma(i, q) * k2_sq(q) * z(:, i, q));  % Variance for ALD errors
            x_tilde = (x(:, i) - k1(q) * z(:, i, q)) ./ v;    % Standardized LHS
            F_tilde = F_select ./ v;                          % Standardized RHS
            
            % Sample loadings using a Gibbs step (user-supplied function)
            L(select, i, q) = randn_gibbs(x_tilde, F_tilde, QL(select, i, q), T, length(select), 1);
            
            % Update the horseshoe prior for loadings
            [QL(select, i, q), ~, ~, lambdaL(i, select, q), tauL(i, q), nuL(i, select, q), xiL(i, q)] = ...
                horseshoe_prior(L(select, i, q)', length(select), tauL(i, q), nuL(i, select, q), xiL(i, q));
            
            % ----- Step 2: Sample Latent Indicators z -----
            FL(:, i, q) = F_all * L(:, i, q);
            if ALsampler == 1
                chi_z = sqrt(k1_sq(q) + 2 * k2_sq(q)) ./ abs(x(:, i) - FL(:, i, q));
                psi_z = (k1_sq(q) + 2 * k2_sq(q)) / (Sigma(i, q) * k2_sq(q));
                z(:, i, q) = min(1 ./ random('InverseGaussian', chi_z, psi_z, T, 1), 1e+6);
            elseif ALsampler == 2
                chi_z = ((x(:, i) - FL(:, i, q)).^2) / (Sigma(i, q) * k2_sq(q));
                psi_z = (k1_sq(q) + 2 * k2_sq(q)) / (Sigma(i, q) * k2_sq(q));
                for t = 1:T
                    z(t, i, q) = min(gigrnd(0.5, psi_z, chi_z(t), 1), 1e+6);
                end
            end
            
            % ----- Step 3: Sample Measurement Error Variance Sigma -----
            a1 = 0.01 + 3 * T / 2;
            sse = (x(:, i) - FL(:, i, q) - k1(q) * z(:, i, q)).^2;
            a2 = 0.01 + sum(sse ./ (2 * z(:, i, q) * k2_sq(q))) + sum(z(:, i, q));
            Sigma(i, q) = 1 ./ gamrnd(a1, 1 ./ a2);
        end  % End loop over countries
        
        % Normalize the latent factor loadings for identification.
        % Here we normalize each factor by its loading for the first country.
        for j = 1:r
            L(interX + ng + j, :, q) = L(interX + ng + j, :, q) ./ L(interX + ng + j, 1, q);
        end
        
        % Construct matrices for the state-space form.
        for i = 1:n
            Ftemp = ones(T, interX);  % Intercept regressor if applicable
            xbar(:, i, q) = (x(:, i) - Ftemp * L(1, i, q) - k1(q) * z(:, i, q)) ./ sqrt(k2_sq(q) * z(:, i, q));
            Lbar(i, 1:r, :, q) = (L(interX + ng + 1:end, i, q) ./ sqrt(k2_sq(q) * z(:, i, q)))';
            Lbar2((q - 1) * n + i, :, :) = L(interX + 1 : interX + ng, i, q) ./ sqrt(k2_sq(q) * z(:, i, q));
        end
    end  % End loop over quantiles
    
    % Construct the block matrix Lc combining the quantile-specific factor loadings.
    Lc = zeros(n * nq, r, T);
    for q = 1:nq
        Lc((q - 1) * n + 1 : q * n, :) = squeeze(Lbar(:, :, :, q))';
    end
    
    % Augment Lc with global loadings from Lbar2 to form the full measurement matrix Lfull.
    % Here Lfull has dimension (n*nq + ng) x r.
    Lfull = zeros(n * nq + ng, r, T);
    for t = 1:T
        Lc_t = [Lc(:, :, t); squeeze(Lbar2(:, :, t)];  % (n*nq+ng) x r
        Lfull(:, :, t) = Lc_t;
    end
    
    %% === Factor Sampling: Sample the State Vector F using FFBS ===
    % Here we stack the measurement equations across quantiles and append the global factor.
    Y_state = [reshape(xbar, T, []), g];  % Dimension: T x (n*nq + 1)
    % Sample the latent state F using a forward-filtering backward-smoothing (FFBS) algorithm.
    % The function FFBS should have a signature similar to: [F] = FFBS(Y, Lfull, intF, Phic, Sigma_aug, Omegac, state_dim)
    % Here the state dimension is (r*nq + ng).
    F = FFBS(Y_state, Lfull, intF, Phic, [Sigma(:); zeros(ng, 1)], Omegac, r * nq + ng);
    
    % Sign identification: rotate factors so that they match the baseline VBQFA factors.
    for q = 1:nq
        for j = 1:r
            Ctemp = corrcoef(F(:, (q - 1) * r + j), fqfa(:, j, q));
            F(:, (q - 1) * r + j) = F(:, (q - 1) * r + j) * sign(Ctemp(1, 2));
            L(interX + ng + j, :, q) = L(interX + ng + j, :, q) * sign(Ctemp(1, 2));
        end
    end
    
    %% === State Equation: VAR Dynamics for Factors and Global Factor ===
    % Construct the VAR regressors using lags of the full state vector F.
    Flag = mlag2(F, p);
    Fy = F(p+1:end, :);
    Fx = [ones(T - p, interF), Flag(p+1:end, :)];
    
    % Initialize residuals for the VAR.
    resid = zeros(T - p, r * nq + ng);
    A_ = eye(r * nq + ng);
    
    % ----- Step 5: Sample VAR Error Variances Omega -----
    se = (Fy - Fx * Phi).^2;
    if var_sv == 0
        b1 = 0.01 + (T - p) / 2;
        b2 = 0.01 + sum(se) / 2;
        Omega(1, :) = 1 ./ gamrnd(b1, 1 ./ b2);
        Omega_t = repmat(Omega, T - p, 1);
    else
        % (Stochastic volatility option not used here)
    end
    
    % ----- Step 6: Sample VAR Coefficients Phi -----
    for i = 1:(r * nq + ng)
        Fy_tilde = Fy(:, i) ./ sqrt(Omega_t(:, i));
        FX_tilde = [Fx, resid(:, 1:i-1)] ./ sqrt(Omega_t(:, i));
        VAR_coeffs = randn_gibbs(Fy_tilde, FX_tilde, [QPhi(:, i); 9 * ones(i - 1, 1)], T - p, k + i - 1, est_meth);
        Phi(:, i) = VAR_coeffs(1:k);
        A_(i, 1:i-1) = VAR_coeffs(k+1:end);
        [QPhi(:, i), ~, ~, lambdaPhi(i, :), tauPhi(i, 1), nuPhi(i, :), xiPhi(i, 1)] = ...
            horseshoe_prior(Phi(:, i)', k, tauPhi(i, 1), nuPhi(i, :), xiPhi(i, 1));
        resid(:, i) = Fy(:, i) - [Fx, resid(:, 1:i-1)] * VAR_coeffs;
    end
    
    Phic = [Phi(interF+1:end, :)'; eye((r*nq+ng)*(p-1)) zeros((r*nq+ng)*(p-1), r*nq+ng)];
    
    % Ensure stationarity of the VAR draws.
    while max(abs(eig(Phic))) > 0.999
        for i = 1:(r * nq + ng)
            Fy_tilde = Fy(:, i) ./ sqrt(Omega_t(:, i));
            FX_tilde = [Fx, resid(:, 1:i-1)] ./ sqrt(Omega_t(:, i));
            VAR_coeffs = randn_gibbs(Fy_tilde, FX_tilde, [QPhi(:, i); 9 * ones(i - 1, 1)], T - p, k + i - 1, 1);
            Phi(:, i) = VAR_coeffs(1:k);
            A_(i, 1:i-1) = VAR_coeffs(k+1:end);
            [QPhi(:, i), ~, ~, lambdaPhi(i, :), tauPhi(i, 1), nuPhi(i, :), xiPhi(i, 1)] = ...
                horseshoe_prior(Phi(:, i)', k, tauPhi(i, 1), nuPhi(i, :), xiPhi(i, 1));
            resid(:, i) = Fy(:, i) - [Fx, resid(:, 1:i-1)] * VAR_coeffs;
        end
        Phic = [Phi(interF+1:end, :)'; eye((r*nq+ng)*(p-1)) zeros((r*nq+ng)*(p-1), r*nq+ng)];
    end
    
    intF(:, 1:r*nq+ng) = (interF == 1) * repmat(Phi(1, :), T, 1);
    OMEGA(:, :, 1:p) = repmat(A_ * diag(Omega_t(1, :)) * A_', 1, 1, p);
    for t = 1:T - p
        OMEGA(:, :, t + p) = A_ * diag(Omega_t(t, :)) * A_';
    end
    Omegac(1:r*nq+ng, 1:r*nq+ng, :) = OMEGA;
    
    %% Save MCMC draws after burn-in
    if irep > nburn && mod(irep, nthin) == 0
        savedraw = savedraw + 1;
        F_draws(:, :, savedraw)       = F(:, 1:r*nq);
        L_draws(:, :, :, savedraw)     = L;
        Phi_draws(:, :, savedraw)       = Phi;
        Sigma_draws(:, :, savedraw)     = Sigma;
        OMEGA_draws(:, :, :, savedraw)  = OMEGA;
        z_draws(:, :, :, savedraw)      = z;
        
        %% Structural Inference: Compute IRFs
        
        % 1) Generalized IRFs for the state equation (for factors)
        ar_lags = Phi(interF+1:end, :)';
        ar0 = {ar_lags(:, 1:r*nq+ng)};
        if p > 1       
            for i = 2:p
                ar0 = [ar0, ar_lags(:, (i-1)*(r*nq+ng)+1 : i*(r*nq+ng))];
            end
        end
        [firf] = armairf(ar0, [], 'InnovCov', squeeze(OMEGA(:, :, end)), 'Method', 'generalized', 'NumObs', nhor);
        firf = permute(firf, [1, 3, 2]);
        
        % 2) GIRFs: Map factor IRFs to observed call rates.
        nshocks = r*nq + ng;
        yirf = zeros(nhor, n*nq, nshocks);
        % Construct the loading matrix for GIRF mapping for each quantile:
        LL = zeros(n*nq, ng + r);
        for q = 1:nq
            % Global loadings (row 2 of L, since row 1 is intercept)
            LL((q - 1) * n + 1 : q * n, 1:ng) = L(interX+1 : interX+ng, :, q)';
            % Latent factor loadings (rows 3:end)
            LL((q - 1) * n + 1 : q * n, ng+1 : ng+r) = L(interX+ng+1 : end, :, q)';
        end
        
        % Map the IRFs of the latent factors to the observed call rates.
        % (Here we assume that the ordering in firf corresponds to [quantile factors; global].)
        for j = 1:nshocks
            yirf(:, :, j) = firf(:, :, j) * LL';
        end
        
        % Save GIRFs for the current draw
        firf_save(:, :, :, savedraw) = firf;
        yirf_save(:, :, :, savedraw) = yirf;
    end
        
end

%% ============================| PLOTS |==============================================
% 1) Plot estimated quantile factors (averaged over MCMC draws)
F_est = squeeze(mean(F_draws, 3));
ddates = datetime(tcode, 'InputFormat', 'yyyyMMdd'); % Adjust the date format as needed

figure;
for j = 1:(r*nq)
   subplot(ceil((r*nq)/2), 2, j)
   plot(ddates, F_est(:, j), 'LineWidth', 2)
   grid on
   title(['Factor ', num2str(j)])
end

% 2) Plot IRFs of factors (state equation)
figure;
for j = 1:ng
    varshock = r*nq + j;      
    irfarray = squeeze(firf_save(:, :, varshock, :));
    for i = 1:(r*nq)
        subplot(ng, r*nq, (j-1)*(r*nq) + i)
        plot(1:nhor, mean(irfarray(:, i, :), 3), 'LineWidth', 2)
        grid on
        if i == 1
            ylabel(namesg{j})
        end
        title(['IRF Factor ', num2str(i)])
    end
end

% 3) Plot GIRFs for observed call rates by quantile block
for q = 1:nq
    figure;
    for i = 1:n
        subplot(ceil(n/2), 2, i)
        % For each country in quantile block q, average the GIRF across shocks for illustration.
        irf_country = mean(yirf_save(:, (q-1)*n+i, :, :), 4);
        plot(1:nhor, irf_country, 'LineWidth', 2)
        grid on
        title(names{i})
    end
    sgtitle(['GIRFs for Quantile ', num2str(quant(q))])
end

%% Save results
save('QFAVAR_MPS_results.mat', 'F_draws', 'L_draws', 'Phi_draws', 'Sigma_draws', 'OMEGA_draws', 'z_draws', 'firf_save', 'yirf_save');
