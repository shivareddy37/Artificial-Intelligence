function [y] = drop(xx)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% DROP-WAVE FUNCTION
% 
% Adapted from Derek Bingham's code.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% INPUT:
%
% xx = [x1, x2 ; 
%       ...]
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

frac1 = 1 + cos(12*sqrt(xx(:,1).^2+xx(:,2).^2));
frac2 = 0.5*(xx(:,1).^2+xx(:,2).^2) + 2;
    
y = -frac1./frac2;

end