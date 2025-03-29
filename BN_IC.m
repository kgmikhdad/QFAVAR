function [path,kstar,crits] = BN_IC(data,kmax)
    
    nn = size(data,2);
    T  = size(data,1);

    ssr = zeros(kmax,1);
    Cnt = min(kmax,T);
    IC1 = zeros(kmax,1);
    IC2 = zeros(kmax,1);
    IC3 = zeros(kmax,1);
    PC1 = zeros(kmax,1);
    PC2 = zeros(kmax,1);
    PC3 = zeros(kmax,1);
    AIC1 = zeros(kmax,1);
    AIC2 = zeros(kmax,1);
    AIC3 = zeros(kmax,1);
    BIC1 = zeros(kmax,1);
    BIC2 = zeros(kmax,1);
    BIC3 = zeros(kmax,1);
    
    [V,D] = eig((data'*data));
    [a,b] = sort(diag(D),'descend');
    Vsort = V(:,b(1:kmax));
    
    ll = Vsort*sqrt(nn);
    
    ff = data*ll/nn;
    fhat = ff*((ff'*ff)/T)^(0.5);
    
    sig = sum(sum((data-fhat*ll').^2))/(nn*T);
    
    c = ll;
    sc = fhat;

    for i = 1:kmax
        yhat   = sc(:,1:i)*c(:,1:i)';
        ss     = (data-yhat).^2;
        ssr(i) = sum(sum(ss))/(nn*T);
        
        IC1(i) = log(ssr(i)) + i*((nn+T)/(nn*T))*log((nn*T)/(nn+T));
        IC2(i) = log(ssr(i)) + i*((nn+T)/(nn*T))*log(Cnt);
        IC3(i) = log(ssr(i)) + i*log(Cnt)/(Cnt);
    
        PC1(i) = ssr(i) + i*sig*((nn+T)/(nn*T))*log((nn*T)/(nn+T));
        PC2(i) = ssr(i) + i*sig*((nn+T)/(nn*T))*log(Cnt);
        PC3(i) = ssr(i) + i*sig*log(Cnt)/(Cnt);
    
        AIC1(i) = ssr(i) + i*sig*(2/T);
        AIC2(i) = ssr(i) + i*sig*(2/nn);
        AIC3(i) = ssr(i) + i*sig*(2*(nn+T-i)/(nn*T));
    
        BIC1(i) = ssr(i) + i*sig*(log(T)/T);
        BIC2(i) = ssr(i) + i*sig*(log(nn)/nn);
        BIC3(i) = ssr(i) + i*sig*(((nn+T-i)*log(nn*T))/(nn*T));
    
    end
    
    
    crits = {"IC1", "PC1", "AIC1", "BIC1",...
             "IC2", "PC2", "AIC2", "BIC2",...
             "IC3", "PC3", "AIC3", "BIC3"};
    
    figure()
    for i = 1:length(crits)
        subplot(3,4,i)
        plot(eval(crits{i}))
    end
    
    kstar = zeros(length(crits),1);
    for i = 1:length(crits)
        kstar(i) = find(eval(crits{i})==min(eval(crits{i})));
    end
    
    path = [IC1 PC1 AIC1 BIC1 IC2 PC2 AIC2 BIC2 IC3 PC3 AIC3 BIC3];

end