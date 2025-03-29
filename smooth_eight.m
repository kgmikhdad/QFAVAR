function object = smooth_eight(b,y,x,h,tau)

% y is the dependent variable
% x is the regressor
% h is the bandwidth
% tau is the quantile 


C1=5*7*9*11/2^13;
C2=7-35+462/5-858/7+715/9-221/11; 

dum = (y-x*b)/h;

object1 = ( tau -  ...
     (...
     1- ( C1 * ( C2+7*dum -35*(dum.^3) +462/5 *( dum.^5)  -858/7 *(dum.^7) +715/9 *(dum.^9)-221/11 *(dum.^11) )) .* (abs(dum)<1 ) -(dum>=1)          ...
       )...
   ).* (y-x*b);

%G = 1- (  0.5+ 105/64 * (dum -5/3*dum.^3 +7/5 *dum.^5  -3/7 *dum.^7)).* (abs(dum)<1 ) -(dum>=1);


object = sum (object1 ); 