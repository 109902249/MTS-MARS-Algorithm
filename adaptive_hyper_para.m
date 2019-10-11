function [alpha,beta,t]=adaptive_hyper_para(k,a1,a2,gamma_a,b1,b2,gamma_b,cur_best_H,gamma_t,t)
%--------------------------------------------------------------------------
% 'adap_hyper_para'
% calculates the adaptive hyperparameters required in the current iteration k
%--------------------------------------------------------------------------
% Output arguments
% ----------------
% alpha       : learning rate for updating the mean parameter function
% beta        : learning rate for updating the gradient estimator
% t           : current temperature
%
% Input arguments
% ---------------
% k           : iteration counter
% a1          : scale factor in alpha=a1/(k+a2)^gamma_a
% a2          : shift factor in alpha=a1/(k+a2)^gamma_a
% gamma_a     : power in alpha=a1/(k+a2)^gamma_a
% b1          : scale factor in beta=b1/(k+b2)^gamma_b;
% b2          : shift factor in beta=b1/(k+b2)^gamma_b;
% gamma_b     : power in beta=b1/(k+b2)^gamma_b;
% cur_best_H  : current best objective value found
% gamma_t     : power in lambda=1/(k+|hk*|)^gamma_t
% t           : previous temperature
%--------------------------------------------------------------------------
% This program is a free software.
% You can redistribute and/or modify it. 
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY, without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
%--------------------------------------------------------------------------

% alpha: learning rate for updating the mean parameter function
alpha=a1/(k+a2)^gamma_a;
% beta: learning rate for updating the gradient estimator
beta=b1/(k+b2)^gamma_b;
% lambda: step-size for the annealing temperature
lambda=1/(k+abs(cur_best_H))^gamma_t;
% current temperature
t=t*(1-lambda);

end