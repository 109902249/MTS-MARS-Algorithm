function cur_h=noisy_observation(cur_H,noise_case,K,noise_bound)
% 'noisy_observation' simulates the noisy observation
%--------------------------------------------------------------------------
% Output arguments
% ----------------
% cur_h       : current noisy observation
%
% Input arguments
% ---------------
% cur_H       : current true objective function value
% noise_case  : type of noise
% K           : noise parameter
% noise_bound : z(x) \in [-noise_bound,noise_bound] 
%--------------------------------------------------------------------------
% This program is a free software.
% You can redistribute and/or modify it. 
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY, without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
%--------------------------------------------------------------------------

switch noise_case
    case 'indep_noise'
        noise=normt_rnd(0,K,-noise_bound,noise_bound);
    case 'prop_noise'
        noise=normt_rnd(0,K/abs(cur_H),-noise_bound,noise_bound);
    case 'solu_dep_noise'
        noise=normt_rnd(0,K*abs(cur_H),-noise_bound,noise_bound);
end
cur_h=cur_H+noise;