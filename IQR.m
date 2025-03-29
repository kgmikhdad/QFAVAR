function [Fhat,Lhat] = IQR(X,r,tolerate,tau)

% X is the T by N matrix of observed variables 
% r is the number of estimated number of factors
% tolerate is the convergence threshold 
% Fhat is the T by r matrix of estimated factors
% Lhat is the N by r matrix of estimated factor loadings

[T,N]=size(X);

% choose the starting value for the IQR algorithm 
rng(2019);
F0=randn(T,r)';

obj_1=1; 
obj_0=0;

while abs(obj_1-obj_0)>tolerate;
    lambda0=[];
    for i=1:N
        lambda0=[lambda0 rq_fnm(F0',X(:,i),tau)];
    end;
    obj_0 = mean( mean ( ( tau - (X< F0'*lambda0 )).*(X-F0'*lambda0))) ; 
        
    F1=[];
    for j=1:T
        F1=[F1 rq_fnm(lambda0',X(j,:)',tau)];
    end;
    F0=F1;
    
    lambda1=[];
    for i=1:N
        lambda1=[lambda1 rq_fnm(F1',X(:,i),tau)];
    end;
    obj_1 = mean( mean ( ( tau - (X< F1'*lambda1 )).*(X-F1'*lambda1))) ; 
    
end

Fhat=F1';
Lambdahat=lambda1';

sigmaF=Fhat'*Fhat/T;
sigmaA=Lambdahat'*Lambdahat/N;

dum1= (sigmaF)^(0.5)*sigmaA*(sigmaF)^(0.5);
[dum2,dum3,dum4]=svd(dum1);
R= (sigmaF)^(-0.5)*dum2;
Fhat = Fhat*R;
Lambdahat=Lambdahat* (inv(R))'; 
Lhat=Lambdahat;