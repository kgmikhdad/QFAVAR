function Omega = update_Omega(Y,Omega,B,iSigma,n,K)

%% ------------------------------------------------------------------------
%% Function to sample orthogonal factors as in Section 6 of Ma and Liu
%% (2022) On Posterior Consistency of Bayesian Factor Models in High
%% Dimensions, Bayesian Analysis 17(3): 901-929. DOI: 10.1214/21-BA1281
%% ------------------------------------------------------------------------

for k = 1:K
    s2 = 1 / sum(B(:, k)'*iSigma* B(:, k)); 

    mean = (Y - Omega * B' + Omega(:, k) * B(:, k)') * (iSigma*(B(:, k)) * s2);
    Omega_k_ = Omega(:, [1:k-1, k+1:end]);
    proj1 = Omega_k_ * pinv(Omega_k_' * Omega_k_) * Omega_k_';
    Omega_kw = Omega;
    Omega_kw(:,k) = mean;
    proj_mean = mean - proj1*mean ;
    w = sqrt(sum(proj_mean.^2));
    
    ini = (-(n - K - 2) * s2 + sqrt((n - K - 2)^2 * s2^2 + 4 * w^2 * n)) / (2 * w);
    std = abs(n - ini^2) / sqrt((n + ini^2) * (n - K - 2));
    d = ini;
    for metro_itr = 1:1000
        d_proposal = d + normrnd(0, std / 2);
        if d_proposal > -sqrt(n) && d_proposal < sqrt(n)
            if log(rand) < (n-K-2)/2*log(n-d_proposal^2)+w*d_proposal/s2 - (n-K-2)/2*log(n-d^2)+w*d/s2
                d = d_proposal;
                break;
            end
        end
    end

    A = zeros(n, n);
    A(:, 1:K) = Omega_kw;
    [Q, R] = qr(A);
    nor = mvnrnd(zeros(1, n - K), eye(n - K));
    Omega(:, k) = (d*proj_mean/w);% + (sqrt(sum(nor.^2))*nor*Q(K+1:end,:))'./sqrt(n - d^2); % + sqrt(n - d^2)./(sqrt(sum(nor.^2))*nor*Q(K+1:end,:))';
end

end


