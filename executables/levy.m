function [y] = levy(xx)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% LEVY FUNCTION
%
% Adapted from Derek Bingham's code.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% INPUT:
%
% xx = [x1, x2, ..., xd ; 
%       ...]
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

d = size(xx,2);
y = zeros(size(xx,1),1);

for k = 1:size(xx,1)

    w=zeros(d,1);
    for ii = 1:d
        w(ii) = 1 + (xx(k,ii) - 1)/4;
    end
    
    term1 = (sin(pi*w(1)))^2;
    term3 = (w(d)-1)^2 * (1+(sin(2*pi*w(d)))^2);
    
    sum = 0;
    for ii = 1:(d-1)
        wi = w(ii);
        new = (wi-1)^2 * (1+10*(sin(pi*wi+1))^2);
        sum = sum + new;
    end
    
    y(k) = term1 + sum + term3;
end

end