function [Q,invQ,miu,lambda,tau,nu,xi] = horseshoe_prior(beta,n,tau,nu,xi)


% sample lambda and nu
rate = (beta.^2)/(2*tau) + 1./nu;
lambda = min(1e+4,1./gamrnd(1,1./rate));    % random inv gamma with shape=1, rate=rate
nu = min(1e+3,1./gamrnd(1,1./(1 + 1./lambda)));    % random inv gamma with shape=1, rate=1+1/lambda_sq_12
% sample tau and xi	
rate = 1/xi + sum((beta.^2)./(2*lambda));
tau = min(1e+4,1/gamrnd((n+1)/2, 1/rate));    % inv gamma w/ shape=(p*(p-1)/2+1)/2, rate=rate
xi = min(1e+3,1/gamrnd(1,1/(1 + 1/tau)));    % inv gamma w/ shape=1, rate=1+1/tau_sq

% %% Horseshoe prior
% %% update lambda_j's in a block using slice sampling %%  
% eta = 1./(lambda.^2); 
% upsi = unifrnd(0,1./(1+eta));
% tempps = beta.^2/(2*sigma_sq*tau^2); 
% ub = (1-upsi)./upsi;
% 
% % now sample eta from exp(tempv) truncated between 0 & upsi/(1-upsi)
% Fub = 1 - exp(-tempps.*ub); % exp cdf at ub 
% Fub(Fub < (1e-4)) = 1e-4;  % for numerical stability
% up = unifrnd(0,Fub); 
% eta = -log(1-up)./tempps; 
% lambda = 1./sqrt(eta);
% 
% %% update tau %%
% tempt = sum((beta./lambda).^2)/(2*sigma_sq); 
% et = 1/tau^2; 
% utau = unifrnd(0,1/(1+et));
% ubt = (1-utau)/utau; 
% Fubt = gamcdf(ubt,(p+1)/2,1/tempt); 
% Fubt = max(Fubt,1e-8); % for numerical stability
% ut = unifrnd(0,Fubt); 
% et = gaminv(ut,(p+1)/2,1/tempt); 
% tau = 1/sqrt(et);

%% update estimate of Q and Q^{-1}
Q = (lambda.*tau);
invQ = 1./Q;

%% estimate of prior mean
miu = zeros(length(beta),1);
