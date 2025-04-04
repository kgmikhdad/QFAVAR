function [y] = transxf(x,tcode)
% =========================================================================
% DESCRIPTION:
% This function transforms a single series (in a column vector)as specified
% by a given transfromation code.
%
% -------------------------------------------------------------------------
% INPUT:
%           x       = series (in a column vector) to be transformed
%           tcode   = transformation code (1-7)
%
% OUTPUT:   
%           y       = transformed series (as a column vector)
%
% =========================================================================
% SETUP:
% Number of observations (including missing values)
n=size(x,1);
 
% Value close to zero 
small=1e-6;

% Allocate output variable
y=NaN*ones(n,1);

% =========================================================================
% TRANSFORMATION: 
% Determine case 1-7 by transformation code
switch(tcode);
    
  case 1, % Level (i.e. no transformation): x(t)
    y=x;

  case 2, % First difference: x(t)-x(t-1)
    y(2:n)=x(2:n,1)-x(1:n-1,1);
  
  case 3, % Second difference: (x(t)-x(t-1))-(x(t-1)-x(t-2))
    y(3:n)=x(3:n)-2*x(2:n-1)+x(1:n-2);

  case 4, % Natural log: ln(x)
    if min(x) < small; 
        y=NaN; 
    else
        y=log(x);
    end;
  
  case 5, % First difference of natural log: ln(x)-ln(x-1)
    if min(x) > small;
        x=log(x);
        y(2:n)=x(2:n)-x(1:n-1);
    end;
  
  case 6, % Second difference of natural log: (ln(x)-ln(x-1))-(ln(x-1)-ln(x-2))
    if min(x) > small;
        x=log(x);
        y(3:n)=x(3:n)-2*x(2:n-1)+x(1:n-2);
    end;
  
  case 7, % Annual differences
     if min(x) > small;
        x=log(x);
        y(13:n)=x(13:end) - x(1:end-12);
    end;     

end;