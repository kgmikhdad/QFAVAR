function r2 = rsquare_a(Y,X)
% produce the adjusted r2 in the regression of Y on X

[n,k]=size(X);
M0=eye(n)-ones(n,n)/n;

X=[ones(n,1) X];

Px= X*inv(X'*X)*X';
resid= (eye(n)-Px)*Y;



r2=1-(resid'*resid/(n-k-1))/(Y'*M0*Y/(n-1));



end
