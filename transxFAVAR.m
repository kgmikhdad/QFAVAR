function [Y,Ylag,dates] = transxFAVAR(Yraw,dates,tcode)

% Transform to stationarity
Y = NaN*Yraw;
for i = 1:size(Yraw,2)
    if tcode(i) == 1        % levels
        Y(:,i) = Yraw(:,i);
    elseif tcode(i) == 2    % first differences
        Y(2:end,i) = (Yraw(2:end,i) - Yraw(1:end-1,i)); 
    elseif tcode(i) == 5    % m-on-m growth rates
        Y(2:end,i) = 100*log(Yraw(2:end,i)./Yraw(1:end-1,i));
    elseif tcode(i) == 7    % annual growth rates
        Y(13:end,i) = 100*log(Yraw(13:end,i)./Yraw(1:end-12,i));
    elseif tcode(i) == 8    % remove linear trend
        Y(:,i)  = Yraw(:,i) - hpfilter(Yraw(:,i),1e+10);
    end
end

if sum(tcode==7)>0
    Y = Y(13:end,:); dates = dates(13:end);
else
    Y = Y(2:end,:); dates = dates(2:end);
end

Ylag  = Y(1:end-1,:);     % construct x(t-1) to be used in measurement equation
Y     = Y(2:end,:);       % construct x(t) to be used in the measurement equation
dates = dates(2:end);