% PURPOSE: Bai and Ng (2002) "Determining the Number of Fcators in Approximate Factors Models", 
% Econometrica, 70,1, p 191-221
%
% -------------------------------------------------------
% Usage:    
% Where:   X = matrix (T,N) of observations 
% Options
%       - kmax : The maximum number of common factors used to compute the criterion functions for  
%                the estimation of r, the number of common factors. It is not specified kmax=min(N,T)
%
% -------------------------------------------------------
% RETURNS:
%           results.khat                 Estimated Numbers of Factor with IC1, IC2, IC3, PC1, PC2, PC3, AIC3 and BIC3
%           results.khat_IC              Estimated Numbers of Factor with IC1, IC2 and IC3
%           results.khat_PC              Estimated Numbers of Factor with PC1, PC2 and PC3 
%           results.khat_BIC3            Estimated Numbers of Factor with BIC3 (only BIC criteria function of N and T)
%           results.khat_AIC3            Estimated Numbers of Factor with AIC3 (only AIC criteria function of N and T) 
%                                        This criterium generaly tends to overstimate the number of factor k
%           results.kmax                 Maximum Number of Factors Authorized (for the case it is not specified)
%           results.IC                   IC1, IC2 and IC3 Information criteria for k=1,..,kmax 
%           results.PC                   PC1, PC2 and PC3 Information criteria for k=1,..,kmax 
%           results.BIC3                 BIC3 Information criterium for k=1,..,kmax 
%           results.AIC3                 AIC3 Information criterium for k=1,..,kmax 
%           results.Vkmax                Estimated Variance of Residuals with kmax factors

% -------------------------------------------------------
%
% C. Hurlin, 08 Juin 2004
% LEO, University of Orléans
%

function [results]=NbFactors(X,kmax);

if nargin==1, kmax=min(size(X)); end                            % Rule proposed by Bai and Ng to choose kmax

if isnan(kmax)==1, kmax=min(size(X)); end                       % Rule proposed by Bai and Ng to choose kmax

%----------------------------------------
%--- Computation of the V(kmax,Fkmax) ---
%----------------------------------------
[T,N]=size(X);                                                  % Sample Sizes

if T<N                                                          % Choice of normalization according the computional cost 
    
    [vectors,values] = eig((X*X'));                               % Eigenvalues and eigenvectors of XX'

    factors=sqrt(T)*vectors(:,T-kmax+1:T);                      % Estimated Factors with kmax Factors

    loadings=X'*factors/T;                                      % Estimated Matrix of Factor Loadings

    betahat=loadings*chol(loadings'*loadings/N);                % Rescaled Estimator of the Factor Loading 
    
else                                                            % Case T>N
   
    [vectors,values] = eig((X*X'));                               % Eigenvalues and eigenvectors of X'X

    loadings=sqrt(N)*vectors(:,N-kmax+1:N);                     % Estimated Matrix of Factor Loadings with kmax Factors

    factors=X*loadings/N;                                       % Estimated Factors

    betahat=(X'*X)*loadings/(N*T);                              % Rescaled Estimator of the Factor Loading                 

end
    
Z=X-X*betahat*inv(betahat'*betahat)*betahat';                   % Estimated Residuals 

var_Z_kmax = sum(sum(Z.^2))/(N*T);                              % Estimated Variance of Residuals with kmax factors

%----------------------------------
%--- Computation of the V(k,Fk) ---
%----------------------------------
V=zeros(kmax,1);                                                 % Vector of V(k,Fk) for k=1,..,kmax

for k=1:kmax                                                     % Loop on the number of factor k
    
    if T<N                                                       % Choice of normalization according the computional cost 
    
        [vectors,values] = eig((X*X'));                            % Eigenvalues and eigenvectors of XX'

        factors=sqrt(T)*vectors(:,T-k+1:T);                      % Estimated Factors with kmax Factors

        loadings=X'*factors/T;                                   % Estimated Matrix of Factor Loadings

        betahat=loadings*chol(loadings'*loadings/N);             % Rescaled Estimator of the Factor Loading 
    
    else                                                         % Case T>N
   
        [vectors,values] = eig((X*X'));                            % Eigenvalues and eigenvectors of X'X

        loadings=sqrt(N)*vectors(:,N-k+1:N);                     % Estimated Matrix of Factor Loadings with kmax Factors

        factors=X*loadings/N;                                    % Estimated Factors

        betahat=(X'*X)*loadings/(N*T);                           % Rescaled Estimator of the Factor Loading                 

    end
    
    Z=X-X*betahat*inv(betahat'*betahat)*betahat';                % Estimated Residuals 

    V(k) = sum(sum(Z.^2))/(N*T);                                 % V(k,Fk)    
    
end

%-----------------------------------
%--- Panel Information Criteria ----
%-----------------------------------
IC=repmat((0:1:kmax)',1,4);                                     % IC information Criteria (IC1, IC2 and IC3)

PC=repmat((0:1:kmax)',1,4);                                     % PC information Criteria (PC1, PC2 and PC3)

AIC3=repmat((0:1:kmax)',1,2);                                   % AIC information Criteria (AIC1, AIC2 and AIC3)

BIC3=repmat((0:1:kmax)',1,2);                                   % BIC information Criteria (BIC1, BIC2 and BIC3)

IC(1,2:4)=log(mean(sum(X.*X/T)))*ones(1,3);                     % IC information Criteria when r=0

PC(1,2:4)=mean(sum(X.*X/T))*ones(1,3);                          % PC information Criteria when r=0

AIC3(1,2)=mean(sum(X.*X/T));                                    % PC information Criteria when r=0

BIC3(1,2)=mean(sum(X.*X/T));                                    % PC information Criteria when r=0

%--------------------------
%--- PC and IC Criteria ---
%--------------------------
CNT=min(N,T);                                                   % Function Cnt^2

Penalty=[(N+T)/(N*T)*log((N*T)/(N+T)) ...                       % Penalty Terms for IC and PC 
        (N+T)/(N*T)*log(CNT) log(CNT)/CNT ];                    

Penalty=repmat(Penalty,kmax,1);                                 % Penalty Terms for IC and PC 

kk=repmat((1:1:kmax)',1,3);                                     % Matrix with increments

PC(2:end,2:end)=repmat(V,1,3)+var_Z_kmax*kk.*Penalty;           % IC information Criteria (IC1, IC2 and IC3)

IC(2:end,2:end)=log(repmat(V,1,3))+kk.*Penalty;                 % PC information Criteria (PC1, PC2 and PC3)

BIC3(2:end,2)=V+kk(:,1)*var_Z_kmax.*(N+T-kk(:,1))*log(N*T)/(N*T);       % BIC3 criterium

AIC3(2:end,2)=V+kk(:,1)*var_Z_kmax.*(N+T-kk(:,1))*2/(N*T);              % AIC3 criterium

%------------------------------------------
%--- Estimated Number of Common Factors ---
%------------------------------------------
[PCs,khat_PC] = min(PC(:,2:4));khat_PC=khat_PC-1;               % Estimated Numbers of Factor with IC

[ICs,khat_IC] = min(IC(:,2:4));khat_IC=khat_IC-1;               % Estimated Numbers of Factor with PC

[BIC3s,khat_BIC3] = min(BIC3(:,2));khat_BIC3=khat_BIC3-1;       % Estimated Numbers of Factor with BIC3

[AIC3s,khat_AIC3] = min(AIC3(:,2));khat_AIC3=khat_AIC3-1;       % Estimated Numbers of Factor with AIC3

khat = [khat_IC khat_PC khat_AIC3 khat_BIC3];                   % Estimated Numbers of Factor with different criteria

%================
%=== RESULTS ====
%================
results.khat=khat;                                               % Estimated Numbers of Factor with IC1, IC2, IC3, PC1, PC2, PC3 and BIC3

results.khat_IC=khat_IC;                                         % Estimated Numbers of Factor with IC1, IC2 and IC3

results.khat_PC=khat_PC;                                         % Estimated Numbers of Factor with PC1, PC2 and PC3 

results.khat_BIC3=khat_BIC3;                                     % Estimated Numbers of Factor with BIC3

results.kmax=kmax;                                               % Maximum Number of Factors Authorized (for the case it is not specified)

results.IC=IC;                                                   % IC1, IC2 and IC3 Information criteria for k=1,..,kmax 

results.PC=PC;                                                   % PC1, PC2 and PC3 Information criteria for k=1,..,kmax 

results.AIC3=AIC3;                                               % BIC3 Information criterium for k=1,..,kmax 

results.BIC3=BIC3;                                               % BIC3 Information criterium for k=1,..,kmax 

results.Vkmax=var_Z_kmax;                                        % Estimated Variance of Residuals with kmax factors

%
% End of Program
%

disp(' ')
disp('  =======================================================================================')
disp('  = Bai and Ng (2002) "Determining the Number of Factors in Approximate Factors Models" =')
disp('  =                   Econometrica 70(1), 191-221                                       =')
disp('  =======================================================================================')
disp(' ')

disp(sprintf('  Number of cross-section units (N) = %1.0f',size(X,2))), disp(' ')

disp(sprintf('  Time dimension (T) = %1.0f',size(X,1))), disp(' ')

disp(sprintf('  Maximum number of common factors = %1.0f',kmax)), disp(' ')

disp('  IC criteria: (nb of factors, IC1, IC2, IC3)')
disp(results.IC)

disp('  PC criteria: (nb of factors, PC1, PC2, PC3)')
disp(results.PC)

disp('  BIC3 criteria: (nb of factors, BIC3)')
disp(results.BIC3)

disp('  AIC3 criteria: (nb of factors, AIC3)')
disp(results.AIC3)

disp(sprintf('  According to IC1 criteria:  %1.0f common factor(s)',results.khat(1))), disp(' ')

disp(sprintf('  According to IC2 criteria:  %1.0f common factor(s)',results.khat(2))), disp(' ')

disp(sprintf('  According to IC3 criteria:  %1.0f common factor(s)',results.khat(3))), disp(' ')

disp(sprintf('  According to PC1 criteria:  %1.0f common factor(s)',results.khat(4))), disp(' ')

disp(sprintf('  According to PC2 criteria:  %1.0f common factor(s)',results.khat(5))), disp(' ')

disp(sprintf('  According to PC3 criteria:  %1.0f common factor(s)',results.khat(6))), disp(' ')

disp(sprintf('  According to BIC3 criteria: %1.0f common factor(s)',results.khat(7))), disp(' ')

disp(sprintf('  According to AIC3 criteria: %1.0f common factor(s)',results.khat(8))), disp(' ')
