function [yfore] = TVPSVMR(y,X,Xfore,nsave,nburn,tvp_reg,sv_reg,prior)
%% tvpsvqr: Function that does flexible estimation of Bayesian quantile regression
%  INPUTS
%    y      LHS variable of interest, measured for t=h+1:T, h is forecast horizon
%    X      RHS matrix (typically intercept, lags of y and lagged exogenous predictors, measured for t=1:T-h)
%  Xfore    Vector of values of X at time T (to forecast y_{T+h})
%  nsave    Number of Gibbs draws to store after convergence
%  nburn    Number of initial Gibbs draws to discard
% tvp_reg   0: Constant parameter regression; 1: TVP regression; 2: TVP only on intercept
% sv_reg    0: Constant variance; 1: Stochastic volatility
%  prior    Shrinkage prior to use. Choices are:
%                                   1: Normal-iGamma prior (Student t)
%                                   2: SSVS with Normal(0,tau_0) and Normal-iGamma components
%                                   3: Horseshoe prior
%==========================================================================
% Written by Dimitris Korobilis, University of Glasgow
% First version: 11 November 2020
% This version: 11 November 2020
%==========================================================================

[T,p] = size(X);

% ==============| create matrices useful for TVP estimation  
if tvp_reg == 0 % Constant parameter regression case
    x = X;
    Tp = p;
    model = 'CP';
elseif tvp_reg == 1 % TVP regression case
    % We need to create an expanded matrix x which has a lower triangular
    % structure with T rows and Tp columns for the TVPs. We also add one column
    % of ones for intercept (these are time-invariant).
    Tp = T*p;   
    H = speye(Tp,Tp) - sparse(p+1:Tp,1:(T-1)*p,ones(1,(T-1)*p),Tp,Tp);
    Hinv = speye(Tp,Tp)/H;
    x = full(SURform(X)*Hinv);
    model = 'TVP';
elseif tvp_reg == 2 % TVP only on intercept case
    Tp = T+(p-1);
    x = [tril(ones(T)), X(:,2:end)];
    model = 'TVI';
end

% ==============| Define priors
% prior for beta
Q    = .01*ones(Tp,1);
miu  = zeros(Tp,1);  
if prior == 1
    % Student_T shrinkage prior
    b0 = 0.01;
elseif prior == 2       
    % SSVS prior
    c0 = (0.01)^2;
    tau = 4*ones(Tp,1);
    pi0 = 0.25;
    b0 = 0.01;
elseif prior == 3   
    % Horseshoe prior
    lambda = 0.1*ones(Tp,1);     % "local" shrinkage parameters, one for each Tp+1 parameter element of betaD
    tau = 0.1*ones(1,1);           % "global" shrinkage parameter for the whole vector betaD
    nu = 0.1*ones(Tp,1);  
    xi = 0.1*ones(1,1);  
end

% ==============| Initialize vectors/matrices
beta = rand(Tp,1);
betaD = rand(Tp,1);
beta_mat = zeros(T,p,1);
sigma_t = 0.1*ones(T,1);
h = ones(T,1); 
sig = 0.1;

% Storage space for Gibbs draws (be careful these vectors can require too
% much memory if you overdo it with either n_q or nsave, or if you insert
% e.g. daily data with large T)
beta_draws  = zeros(T,p,nsave);
sigma_draws = zeros(T,nsave);
yfore       = zeros(nsave,1);

%% =========| GIBBS sampler starts here
iter = 500;             % Print every "iter" iteration
if sv_reg == 1
    model2 = 'SV';
elseif sv_reg == 0
    model2 = 'CV';
end
fprintf(['Model: MR with ' str2mat(model) ' and ' str2mat(model2)])
fprintf('\n')
fprintf('Iteration 0000')
for irep = 1:(nsave+nburn)
    % Print every "iter" iterations on the screen
    if mod(irep,iter) == 0
        fprintf('%c%c%c%c%c%c%c%c%c%c%c%c%c%c%s%4d',8,8,8,8,8,8,8,8,8,8,8,8,8,8,'Iteration ',irep)
    end   
    
    % =====| Step 2: Sample regression variance
    if sv_reg == 1   % Sample Stochastic Volatility
        yhat = y - x*betaD;                                     % regression residuals
        ystar  = log(yhat.^2 + 1e-6);                           % log squared residuals        
        [h, ~] = SVRW(ystar,h,sig,1);                           % log stochastic volatility using Chan's filter   
        sigma_t  = exp(h);                                      % convert log-volatilities to variances
        r1 = 1 + T - 1;   r2 = 0.01 + sum(diff(h).^2)';         % posterior moments of variance of log-volatilities
        sig = 1./gamrnd(r1./2,2./r2);                           % sample variance of log-volatilities from Inverse Gamma
    elseif sv_reg == 0   % Sample constant regression variance
        a1 = 0.1 + T/2;    sse = (y - x*betaD).^2;
        a2 = 0.1 + sum(sse);       
        sigma2 = 1./gamrnd(a1,1./a2);                            % Sample from inverse-Gamma
        sigma_t = repmat(sigma2,T,1);                            % Convert to a T x n_q matrix (but replicate same value for all t=1,...,T)
    end
    
    % =====| Step 3: Sample regression coefficients beta_t
    u = (1./sqrt(sigma_t));                       % This is the standard deviation of the Asymetric Laplace errors
    y_tilde = y  .*u;                             % Standardize y for GLS formula
    xnew = x.*u;                                  % Also standardize x           
    betaD = randn_gibbs(y_tilde,xnew,Q,T,Tp,2);    % sample regression coefficients
    if tvp_reg == 1
        beta = Hinv*betaD;                % betaD are TVPs in differences (beta_{t} - beta_{t-1}), convert to vector of TVPs in levels (beta_{t})
        beta_mat = reshape(beta,p,T)';    % beta_mat is T x (p+1) x n_q matrix of TVPs
    elseif tvp_reg == 2
        beta = [cumsum(betaD(1:T,:));betaD(T+1:end,:)];
        beta_mat = [beta(1:T,:) , repmat(betaD(T+1:end,:)',T,1)];
    elseif tvp_reg == 0
        beta = betaD;                     % In the constant parameter case, betaD are the beta coefficients we need
        beta_mat = repmat(beta',T,1);
    end
    
    %--draw prior variance
    if prior == 1
        [Q,~,miu] = student_T_prior(betaD,b0);
    elseif prior == 2
        [Q,~,miu,tau] = ssvs_prior(betaD,c0,tau,b0,pi0);
    elseif prior == 3
        [Q,~,miu,lambda,tau,nu,xi] = horseshoe_prior(betaD,Tp,tau,nu,xi);
    end
        
    if irep > nburn
        % Save draws of parameters
        beta_draws(:,:,irep-nburn) = beta_mat;
        sigma_draws(:,irep-nburn)  = sigma_t;
        
        % Forecast of quantiles from the QR model
        yfore(irep-nburn,:) = Xfore*squeeze(beta_mat(end,:))' + sqrt(sigma_t(end)).*randn;
    end    
end
fprintf('%c%c%c%c%c%c%c%c%c%c%c%c%c%c%c',8,8,8,8,8,8,8,8,8,8,8,8,8,8,8)
fprintf('\n')
end

