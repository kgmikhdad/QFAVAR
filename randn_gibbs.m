function [Beta] = randn_gibbs(y,X,lambda,n,p,algorithm)


if algorithm == 1
    %% matrices %%
    Q_star=X'*X;
    Dinv = diag(1./lambda);       
    L=chol((Q_star + Dinv),'lower');
    v=L\(y'*X)';
    mu=L'\v;
    u=L'\randn(p,1);
    Beta = mu + u;
elseif algorithm == 2
    U = bsxfun(@times,lambda,X');
    %% step 1 %%
    u = normrnd(zeros(p,1),sqrt(lambda));
    %u = sqrt(lambda).*randn(p,1);	
    v = X*u + randn(n,1);
    %% step 2 %%
    v_star = ((X*U) + eye(n))\(y-v);
    Beta = (u + U*v_star);
end
