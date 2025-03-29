function [con_tot, con_dto, con_dfrom, con_net, con_netpw, diags] = mfc_con_est_new(est_AR, est_sigma, p, H)%, var_names, do_table)
%mfc_con_est - produces the set of connectedness measures proposed by DY
%   obtained from the FEVD of a generalised VAR
%
% [con_tot, con_dto, con_dfrom, con_net, con_netpw, diags] = mfc_con_est(x, p, H)
% series_names)
%
% Inputs:
%   x           (T x N) array of data, with one time series of length T in
%               each column
%   p           lag order for the underlying VAR model from which the FEVD
%               is obtained
%   H           forecast horizon used for constructing FEVD and spillover
%               measures
%   var_names   cell array containing names of variables (optional)
%   tables      create table displaying spillovers if tables = 1
%
% Outputs:
%   con_tot     total connectedness index
%
%   con_dto     total connectedness index as a (1 x H) vector SO_dto
%               directional spillover from series i TO  all other series -
%               formatted as an (N x H) array, with element (n,h) containing
%               the spillover from series i TO all other series at a
%               horizon of h
%   con_dfrom   directional connectedness to series i FROM all other series -
%               formatted equivalently to SO_dto above
%   con_net     net connectedness
%
%
%% calculate and save corresponding VMA(oo) coefficient arrays

% VMA_est = vgxma(VAR_est, H);
% est_MA = VMA_est.MA;

N = size(est_sigma,1);

%preallocate cell array for MA coeffcient arrays, A
est_MA = cell(H+p,1); %note dimension - must have A_{-p} to A_0 for recursion

for i = 1:H+p
    if i < p
        est_MA{i} = zeros(N,N); %A_i is array of zeros for i < 0
    elseif i == p
        est_MA{i} = eye(N); %A_i is identity matrix for i = 0
    else
        for j = 1:p
            if j == 1
                est_MA{i} = est_AR{j}*est_MA{i-j};
            else
                est_MA{i} = est_MA{i} + est_AR{j}*est_MA{i-j};
            end 
        end
    end
end

%remove initial cells - NOTE: cell 1 now contains A_0, cell 2 contains A_1, etc
est_MA = est_MA(p:end);


%% calcuate generalised FEVD at horizons 1,...,H for all variables in VAR

%create array of selection vectors - i-th column of e_mat gives e_i
e_mat = eye(N);

%create array of sigma terms for premultiplying summation tern in numerator
%%% Q1: should these be VARIANCE terms or SDs? PS list them as vars, but DY as SDs
%%% Q2: should the index on the sigmas be ii or jj? 
tmp_sigma_diag = repmat((diag(est_sigma).^-1)', N, 1); %vars, index as jj - AS IN DY2014
% tmp_sigma_diag = repmat((diag(sqrt(est_sigma)).^-1), 1, N); %SDs, index as ii
% tmp_sigma_diag = repmat((diag(est_sigma).^-1), 1, N); %vars, index as ii - AS IN PS
% tmp_sigma_diag = repmat((diag(sqrt(est_sigma)).^-1)', N, 1); %%SDs, index as jj - AS IN DY2012

%preallocate arrays for nonstandardised FEVD (theta) and standardised FEVD
%(theta_tilde) for each h = 0,1,...,H-1
theta = cell(1,H+1);
theta_tilde = cell(1,H+1);

%preallocate cumulative numerator and denominator arrays - represent the
%summation terms in numerator and denominator for h = 1,2...,H
cum_numer = cell(1,H+1);
cum_denom = cell(1,H+1);

for h = 1:H+1
    
    %calculate new terms in summation numerator and denominator arrays for
    %current h to be added to the existing summation of terms
    loop_numer = zeros(N,N); %array, since a function of i and j
    loop_denom = zeros(N,1); %vector, since only a function of i
    
    for i = 1:N
        %calculate denominator term - only a function of i, not j
        loop_denom(i) = (e_mat(:,i)'*est_MA{h}*est_sigma*est_MA{h}'*e_mat(:,i));
        
        for j = 1:N
            %calculate numerator term - function of i and j
            loop_numer(i,j) = ((e_mat(:,i)'*est_MA{h}*est_sigma*e_mat(:,j))).^2;
        end
    end
    
    %cumulate numerator and denominator terms to obtain value of summation
    %terms for current value of h
    if h == 1
        cum_numer{h} = loop_numer;
        cum_denom{h} = repmat(loop_denom, 1, N); %note expansion from (N x 1) vector to (N x N) array
    else
        cum_numer{h} = loop_numer + cum_numer{h-1};
        cum_denom{h} = repmat(loop_denom, 1, N) + cum_denom{h-1};
    end
    
    %create theta array for current h
    theta{h} = (tmp_sigma_diag.*cum_numer{h})./cum_denom{h};
    
    %normalise theta s.t. each row sums to unity to obtain theta_tilde
    theta_tilde{h} = theta{h}./repmat(sum(theta{h}, 2), 1, N);
end

clear i j h cum_numer cum_denom e_mat tmp_sigma_diag loop_numer loop_denom


%% create total, directional and net connectedness indexes

%a) total connectedness index - scalar for each horizon h
con_tot = zeros(1, H);

for h = 1:H
    con_tot(h) = 100*(sum(sum(theta_tilde{h})) - sum(diag(theta_tilde{h})))/N;
end


%b) directional FROM connectedness - arranged as a (N x VAR_h) array, with
%each row containing the spillovers for a given series
con_dfrom = zeros(N, H);

for h = 1:H
    con_dfrom(:,h) = 100*(sum(theta_tilde{h}, 2) - diag(theta_tilde{h}))/N;
end


%c) directional TO connectedness
con_dto = zeros(N, H);

for h = 1:H
    con_dto(:,h) = 100*(sum(theta_tilde{h})' - diag(theta_tilde{h}))/N;
end


%d) net connectedness (total)
con_net = con_dto - con_dfrom;


%e) net connectedness, pairwise
con_netpw = cell(1, H);

for h = 1:H
    con_netpw{h} = 100.*(theta_tilde{h})/N;
end

%for h = 1:H
%    con_netpw{h} = 100.*(theta_tilde{h}' - theta_tilde{h})/N;
%end

%% save additional info into diagnostic structure if required

if nargout > 1
    diags.sigma = est_sigma;
    diags.ARcoeffs = est_AR;
    diags.MAcoeffs = est_MA;
    diags.theta = theta;
    diags.theta_tilde = theta_tilde;
end


%% create total connectedness table if required

% if do_table == 1
%     
%     table = cell(N+3,N+2);
%     
%     table(1,2:N+1) = var_names;
%     table{1,N+2} = 'Dir FROM';
%     
%     table(2:N+1,1) = var_names;
%     table{N+2,1} = 'Dir TO';
%     table{N+3,1} = 'Dir inc own'; 
% end

% end