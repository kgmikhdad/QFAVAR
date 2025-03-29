%% FAVAR_GIRFs.m    Bayesian Factor-Augmented VAR Model for Monetary Policy Spillover
%%                  (MCMC estimation + Generalized IRFs)
%==========================================================================
% The model is of the form
%          _       _        _           _    _        _     _    _
%         |  x_{t}  |      |  L       G  |  |F_{t}|   | e_{t} |
%         |         |  =   |             |  |     | + |       |
%         |_ g_{t} _|      |_ 0       I _|  |_g_{t}|   |_  0  _|
%
%         _          _           _            _
%        | F_{t}     |         | F_{t-1}    |  
%        |           |  =  Phi |            |  + u_{t},
%        |_  g_{t}   |         |_  g_{t-1}  _|
%
% where e_{t} ~ N(0, Sigma) and u_{t} ~ N(0, Q), with Sigma and Q being
% covariance matrices.
%
% This updated code implements the FAVAR methodology for analyzing monetary 
% policy spillovers using 24‑hour call rate data from 14 countries.
%
% Assumptions:
%   - The file "call_rate_data.mat" contains:
%         x            : T-by-n matrix of call rates (n ≈ 14)
%         dates        : T-by-1 vector of dates
%         country_names: 1-by-n cell array of country names
%   - A global factor is constructed as the cross‑country average call rate.
%
% Ensure that the functions randn_gibbs, horseshoe_prior, FFBS, mlag2, and armairf 
% are available in your MATLAB path.
%
% Written by: [Your Name]
% Date: [Current Date]
%==========================================================================
clear all; clc; close all;
tic;

%% Add paths for functions and data
addpath('functions')
addpath('data')

%% USER INPUT & MODEL SETTINGS
r         = 3;              % Number of latent factors
p         = 2;              % Number of lags in the VAR (state equation)
interX    = 0;              % Include intercept in measurement equation (set to 0 here)
interF    = 0;              % Include intercept in state equation
AR1x      = 0;              % Own lags in measurement equation (not used)
incldg    = 1;              % Include global factor in measurement equation
dfm       = 0;              % 0: estimate FAVAR; 1: estimate DFM (here FAVAR)
var_sv    = 0;              % VAR variances: 0 (constant) or 1 (stochastic volatility)
standar   = 0;              % No standardization (assumed data are in appropriate scale)
nhor      = 60;             % Horizon for IRFs, FEVDs, etc.

% MCMC settings
nsave     = 100000;         % Number of draws to store
nburn     = 5000;           % Number of draws to discard
ngibbs    = nsave + nburn;   % Total number of draws
nthin     = 50;             % Save every nthin draw to reduce MCMC correlation
iter      = 100;            % Print progress every "iter" iterations

%% LOAD 24-HOUR CALL RATE DATA FOR MONETARY POLICY SPILLOVER
% Expected variables in "call_rate_data.mat":
%   x            : T-by-n matrix of 24-hour call rates (n should be ≈14)
%   dates        : T-by-1 vector of dates
%   country_names: 1-by-n cell array of country names
load('call_rate_data.mat');  % Loads x, dates, country_names
[T, n] = size(x);

% If needed, compute lags for x (here AR1x = 0, so not used)
if AR1x
    xlag = lagmatrix(x, 1:p);
else
    xlag = [];
end

% Construct the global factor as the cross-country average call rate
ng = 1;                        % Number of global factors
g = mean(x, 2);

% Set variable names
names  = country_names;        % 1-by-n cell array of country names
namesg = {'Global Call Rate'};
tcode  = dates;                % Time codes

% Number of VAR regressors in the state equation:
% k = (r + ng)*p + interF
k = (r + ng) * p + interF;

%% Preliminaries for Estimation
% Measurement equation: for each country, x_t = [Intercept, Global, Factors]*L + error.
load_dim = interX + ng + r;   % Total number of coefficients per equation
% Horseshoe priors for loadings L
lambdaL   = 0.1 * ones(n, load_dim);
tauL      = 0.1 * ones(n, 1);
nuL       = 0.1 * ones(n, load_dim);
xiL       = 0.1 * ones(n, 1);

% Horseshoe priors for VAR coefficients Phi (state equation)
lambdaPhi = 0.1 * ones(r + ng, k);
tauPhi    = 0.1 * ones(r + ng, 1);
nuPhi     = 0.1 * ones(r + ng, k);
xiPhi     = 0.1 * ones(r + ng, 1);

% Choose sampling algorithm for VAR parameters (depends on k and T)
est_meth = 1 + double(k > T);

%% Initialize Matrices
xbar      = zeros(T, n);
Lbar      = zeros(n, r, T);
Lbar2     = zeros(n, ng, T);
L         = zeros(load_dim, n);      % Measurement loadings matrix
Sigma     = 0.1 * ones(n, 1);          % Measurement error variances
Phi       = 0.1 * ones(k, (r + ng));     % VAR coefficients
Omega     = 0.1 * ones(1, r + ng);
Omega_t   = 0.1 * ones(T - p, r + ng);
OMEGA     = 0.1 * ones(r + ng, r + ng, T);
h         = 0.1 * ones(T - p, r + ng);
sig       = 0.1 * ones(r + ng, 1);
FL        = zeros(T, n);
Omegac    = zeros((r + ng) * p, (r + ng) * p, T);
Phic      = [Phi(interF+1:end, :)'; eye((r+ng)*(p-1)) zeros((r+ng)*(p-1), r+ng)];
Omegac(1:r+ng, 1:r+ng, :) = repmat(diag(Omega), 1, 1, T);
QL        = ones(load_dim, n);
QPhi      = ones(k, (r + ng));
intF      = zeros(T, (r + ng) * p);

%% Factor Extraction using PCA
disp('Extracting PCA factors...');
[coeff, score, ~] = pca(zscore(x));
fpca = score(:, 1:r);  % T x r factor estimates
F = fpca;              % Initialize factor estimates

%% Identification: Impose normalization restrictions on factor loadings
% For identification we normalize the loading for factor j using country j.
% (Assumes n >= r; here n=14 and r=3)
for j = 1:r
    L(interX + ng + j, j) = 1;
end

clc;

%% STORAGE FOR GIBBS DRAWS
F_draws     = zeros(T, r, nsave/nthin);
L_draws     = zeros(load_dim, n, nsave/nthin);
Phi_draws   = zeros(k, r + ng, nsave/nthin);
Sigma_draws = zeros(n, nsave/nthin);
OMEGA_draws = zeros(r + ng, r + ng, T, nsave/nthin);
firf_save   = zeros(nhor, r + ng, r + ng, nsave/nthin);
yirf_save   = zeros(nhor, n, r + ng, nsave/nthin);

%% ============================| START MCMC |==============================
format bank;
fprintf('Now running FAVAR MCMC for Monetary Policy Spillover\n');
fprintf('Iteration 000000\n');
savedraw = 0;
for irep = 1:(nsave + nburn)
    % Print progress every "iter" iterations
    if mod(irep, iter) == 0
        fprintf('Iteration %6d\n', irep);
    end
    
    %% === Measurement Equation: Factor Extraction ===
    % For each country i, sample loadings and measurement error variance Sigma.
    Lc = zeros(n, r + ng, T);
    Lfull = zeros(n + ng, r + ng, T);
    x_tilde = zeros(T, n);
    
    for i = 1:n
        % Build the regressor matrix for the measurement equation:
        % [Intercept, Global Factor, Latent Factors]
        F_all = [ones(T, interX), g, F(:, 1:r)];   % T x (interX+ng+r)
        select = 1:load_dim;                        % Use all coefficients
        
        F_select = F_all(:, select);
        
        % Step 1: Sample loadings L for country i
        x_tilde(:, i) = x(:, i) ./ sqrt(Sigma(i));  % Standardize LHS
        F_tilde = F_select ./ sqrt(Sigma(i));         % Standardize RHS
        
        % Sample loadings (using user-supplied Gibbs sampler)
        L(select, i) = randn_gibbs(x_tilde(:, i), F_tilde, QL(select, i), T, length(select), 1);
        
        % Update the horseshoe prior for loadings
        [QL(select, i), ~, ~, lambdaL(i, select), tauL(i), nuL(i, select), xiL(i)] = ...
            horseshoe_prior(L(select, i)', length(select), tauL(i), nuL(i, select), xiL(i));
        
        % Impose identification restriction: for factor j (j=1,...,r), set loading of country j equal to 1.
        if i <= r
            L(interX + ng + i, i) = 1;
        end
        
        % Step 2: Sample measurement error variance Sigma for country i
        FL(:, i) = F_all * L(:, i);
        a1 = 0.01 + T/2;
        sse = (x(:, i) - FL(:, i)).^2;
        a2 = 0.01 + sum(sse) / 2;
        Sigma(i) = 1 / gamrnd(a1, 1 / a2);
        
        % Remove intercept part from x (if applicable) for state equation
        Ftemp = [ones(T, interX), xlag(:, (AR1x * i))];  % (Empty if AR1x==0)
        xbar(:, i) = x(:, i) - Ftemp * L(1:interX, i);
    end
    
    % Construct Lfull for the state-space representation:
    % For each time t, set:
    %   Top block: [L(latent part)'  L(intercept+global part)'] 
    %   Bottom block: [zeros(ng, r)  eye(ng)]
    for t = 1:T
        % L(latent part): rows (interX+ng+1:end) of L (r x n), transposed gives n x r.
        % L(intercept+global part): rows 1:interX+ng of L (if interX=0 then just global) transposed gives n x (interX+ng).
        Lfull(:,:,t) = [ [L(interX+ng+1:end, :)',  L(1:interX+ng, :)']; ];
        % Append the measurement equation for the global factor (assumed measured without error)
        Lfull(:,:,t) = [Lfull(:,:,t); [zeros(ng, r), eye(ng)]];
    end
    
    %% === Factor Sampling: Sample the State Vector F via FFBS ===
    % Stack the measurement equations for x and the global factor:
    % Y_state = [xbar, g] with dimension T x (n+ng)
    Y_state = [xbar, g];
    % Sample F using a forward-filtering backward-smoothing algorithm.
    % (State dimension is r+ng.)
    F = FFBS(Y_state, Lfull, intF, Phic, [Sigma(:); zeros(ng, 1)], Omegac, r + ng);
    
    % Sign identification: Rotate factors to match the baseline PCA factors.
    for j = 1:r
        Ctemp = corrcoef(F(:, j), fpca(:, j));
        F(:, j) = F(:, j) * sign(Ctemp(1,2));
        L(interX+ng+j, :) = L(interX+ng+j, :) * sign(Ctemp(1,2));
    end
    
    %% === State Equation: VAR Dynamics ===
    % Construct VAR regressors using lags of [F(:,1:r), g]
    Flag = mlag2([F(:, 1:r), g], p);
    Fy = [F(p+1:end, 1:r), g(p+1:end, :)];
    Fx = [ones(T-p, interF), Flag(p+1:end, :)];
    resid = zeros(T-p, r + ng);
    A_ = eye(r + ng);
    
    % Step 5: Sample VAR error variances Omega
    se = (Fy - Fx * Phi).^2;
    if var_sv == 0
        b1 = 0.01 + (T-p)/2;
        b2 = 0.01 + sum(se) / 2;
        Omega(1,:) = 1 ./ gamrnd(b1, 1./b2);
        Omega_t = repmat(Omega, T-p, 1);
    elseif var_sv == 1
        fystar = log(se + 1e-6);
        for i = 1:(r+ng)
            [h(:,i), ~] = SVRW(fystar(:,i), h(:,i), sig(i,:), 4);
            Omega_t(:,i) = exp(h(:,i));
            r1 = 1 + (T-p-1)/2;
            r2 = 0.01 + sum(diff(h(:,i)).^2)'/2;
            sig(i,:) = 1 ./ gamrnd(r1/2, 2/r2);
        end
    end
    
    % Step 6: Sample VAR coefficients Phi
    for i = 1:(r+ng)
        Fy_tilde = Fy(:,i) ./ sqrt(Omega_t(:,i));
        FX_tilde = [Fx, resid(:,1:i-1)] ./ sqrt(Omega_t(:,i));
        VAR_coeffs = randn_gibbs(Fy_tilde, FX_tilde, [QPhi(:,i); 9*ones(i-1,1)], T-p, k+i-1, est_meth);
        Phi(:,i) = VAR_coeffs(1:k);
        A_(i,1:i-1) = VAR_coeffs(k+1:end);
        [QPhi(:,i), ~, ~, lambdaPhi(i,:), tauPhi(i,1), nuPhi(i,:), xiPhi(i,1)] = ...
            horseshoe_prior(Phi(:,i)', k, tauPhi(i,1), nuPhi(i,:), xiPhi(i,1));
        resid(:,i) = Fy(:,i) - [Fx, resid(:,1:i-1)] * VAR_coeffs;
    end
    Phic = [Phi(interF+1:end, :)'; eye((r+ng)*(p-1)) zeros((r+ng)*(p-1), r+ng)];
    
    % Ensure stationarity of VAR draws
    while max(abs(eig(Phic))) > 0.999
        for i = 1:(r+ng)
            Fy_tilde = Fy(:,i) ./ sqrt(Omega_t(:,i));
            FX_tilde = [Fx, resid(:,1:i-1)] ./ sqrt(Omega_t(:,i));
            VAR_coeffs = randn_gibbs(Fy_tilde, FX_tilde, [QPhi(:,i); 9*ones(i-1,1)], T-p, k+i-1, 1);
            Phi(:,i) = VAR_coeffs(1:k);
            A_(i,1:i-1) = VAR_coeffs(k+1:end);
            [QPhi(:,i), ~, ~, lambdaPhi(i,:), tauPhi(i,1), nuPhi(i,:), xiPhi(i,1)] = ...
                horseshoe_prior(Phi(:,i)', k, tauPhi(i,1), nuPhi(i,:), xiPhi(i,1));
            resid(:,i) = Fy(:,i) - [Fx, resid(:,1:i-1)] * VAR_coeffs;
        end
        Phic = [Phi(interF+1:end, :)'; eye((r+ng)*(p-1)) zeros((r+ng)*(p-1), r+ng)];
    end
    intF(:,1:r+ng) = (interF==1) * repmat(Phi(1,:), T, 1);
    OMEGA(:,:,1:p) = repmat(A_ * diag(Omega_t(1,:)) * A_', 1, 1, p);
    for t = 1:T-p
        OMEGA(:,:,t+p) = A_ * diag(Omega_t(t,:)) * A_';
    end
    Omegac(1:r+ng, 1:r+ng, :) = OMEGA;
    
    %% Save MCMC draws (after burn-in and thinning)
    if irep > nburn && mod(irep, nthin) == 0
        savedraw = savedraw + 1;
        F_draws(:,:,savedraw)       = F(:,1:r);
        L_draws(:,:,savedraw)       = L;
        Phi_draws(:,:,savedraw)     = Phi;
        Sigma_draws(:,savedraw)     = Sigma;
        OMEGA_draws(:,:,:,savedraw) = OMEGA;
        
        %% Structural Inference: Compute IRFs
        % 1) Generalized IRFs for the state equation (responses of [factors; global])
        ar_lags = Phi(interF+1:end, :)';
        ar0 = {ar_lags(:, 1:r+ng)};
        if p > 1       
            for i = 2:p
                ar0 = [ar0, ar_lags(:, (i-1)*(r+ng)+1:i*(r+ng))];
            end
        end
        [firf] = armairf(ar0, [], 'InnovCov', squeeze(OMEGA(:,:,end)), 'Method', 'generalized', 'NumObs', nhor);
        firf = permute(firf, [1, 3, 2]);
        
        % 2) GIRFs: Map the factor IRFs to observed call rates.
        nshocks = r + ng;
        yirf = zeros(nhor + AR1x, n, nshocks);
        LL = zeros(n, r + ng);
        % Stack loadings: first ng columns for the global factor, next r for latent factors.
        for i = 1:n
            LL(:,1:ng) = L(1:ng, :)';
            LL(:,ng+1:r+ng) = L(interX+ng+1:end, :)';
        end
        
        if AR1x == 1
            for j = 1:nshocks
                for h = 2:nhor+AR1x
                    yirf(h,:,j) = [firf(h-1, r+1:end, j), firf(h-1, 1:r, j)] * LL' + yirf(h-1,:,j) .* L(interX+AR1x, :);
                end
            end
            yirf = yirf(2:end,:,:);
        else
            for j = 1:nshocks
                yirf(:,:,j) = [firf(:, r+1:end, j), firf(:, 1:r, j)] * LL';
            end
        end
        % Save GIRFs for current draw
        firf_save(:,:,:,savedraw) = firf;
        yirf_save(:,:,:,savedraw) = yirf;
    end
end

%% =====================================| PLOTS |==============================================
% 1) Plot estimated factors (averaged over MCMC draws)
F_est = squeeze(mean(F_draws, 3));
figure;
for i = 1:r
   subplot(ceil(r/2), 2, i)
   plot(F_est(:, i), 'LineWidth', 2)
   grid on
   title(['Factor ', num2str(i)])
end

% 2) Plot IRFs of the state equation (responses of factors/global)
FigH = figure('Position', get(0, 'Screensize'));
for j = 1:ng
    varshock = r + j;
    irfarray = squeeze(firf_save(:, :, varshock, :));
    for i = 1:(r+1)
        subplot(ng, r+1, (j-1)*(r+1) + i)
        if i <= r
            plot(1:nhor, mean(irfarray(:, i, :), 3), 'LineWidth', 2)
            hold on
            shade(1:nhor, prctile(irfarray(:, i, :), 25, 3), 'w', 1:nhor, prctile(irfarray(:, i, :), 75, 3), 'w',...
                'FillType', [2 1], 'FillColor', {'black'}, 'FillAlpha', 0.2, 'LineStyle', "None")
            plot(1:nhor, zeros(1, nhor), 'r')
            hold off
            title(['Factor ', num2str(i)])
        else
            plot(1:nhor, mean(irfarray(:, r+j, :), 3), 'LineWidth', 2)
            title(namesg{j})
        end        
        if i == 1
            ylabel(namesg{j})
        end
    end
end
saveas(FigH, 'FAVAR_IRF_state_eq.jpg', 'jpeg');

% 3) Plot GIRFs for observed call rates
for ii = 1:length(namesg)
    varshock = r + ii;
    countries = names;  % Use country names for labeling
    irfarray = squeeze(yirf_save(:, :, varshock, :));
    % Arrange plots in a grid (for example, 2 rows x ceil(n/2) columns)
    nrows = 2; ncols = ceil(n / nrows);
    figure;
    for i = 1:n
        subplot(nrows, ncols, i)
        plot(mean(irfarray(:, i, :), 3), 'LineWidth', 2)
        hold on
        shade(1:nhor, prctile(irfarray(:, i, :), 25, 3), 'w', 1:nhor, prctile(irfarray(:, i, :), 75, 3), 'w',...
            'FillType', [2 1], 'FillColor', {'black'}, 'FillAlpha', 0.2, 'LineStyle', "None")
        plot(1:nhor, zeros(1, nhor), 'r')
        hold off
        grid on;
        title(countries{i})
    end
    sgtitle(namesg{ii})
end

%% Save results
save('FAVAR_GIRFs_results.mat', '-mat');
toc;
