%--------------------------------------------------------------------------
% Corresponding author: Qi Zhang
% Department of Applied Mathematics and Statistics
% Math Tower P137, Stony Brook University, Stony Brook, NY 11794-3600
% Email: zhangqi{dot}math@gmail{dot}com
%--------------------------------------------------------------------------
% 1. The MTS-MARS algorithm by Qi Zhang and Jiaqiao Hu [1] is implemented 
% for solving single-objective box-constrained expensive stochastic
% optimization problems
% 2. The MTS-MARS algorithm generalizes the TS-MARS algorithm [2] where the
% former version is designed for sovling single-objective box-constrained 
% expensive deterministic optimization problems
% 3. In this implementation, the algorithm samples candidate solutions from 
% a sequence of independent multivariate normal distributions that 
% recursively approximiates the corresponding Boltzmann distributions [3]
%--------------------------------------------------------------------------
% REFERENCES
% [1] Qi Zhang and Jiaqiao Hu (2019): Simulation Optimization Using Multi-time-scale
% Adaptive Random Search. Submitted to Asia-Pacific Journal of Operational 
% Research, under review.
% [2] Qi Zhang and Jiaqiao Hu (2017): A Two-time-scale Adaptive Search Algorithm
% for Global Optimization. The Proceedings of the 2017 Winter Simulation 
% Conference, pp. 2069-2079.
% [3] Jiaqiao Hu and Ping Hu (2011): Annealing adaptive search,
% cross-entropy, and stochastic approximation in global optimization.
% Naval Research Logistics 58(5):457-477.
%--------------------------------------------------------------------------
% This program is a free software.
% You can redistribute and/or modify it. 
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY, without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
%--------------------------------------------------------------------------

clearvars; close all;
%% PROBLEM SETTING
% The test function H(x) is the sum squares function in dimension 20 (d=20)
% with box-constrain [-20,20]^d
% and shifted to have an optimal(max) objective value -1
% Three types of observation noise z(x) are tested.
% 1. Independent noise: z(x)~TN(0,K)
% 2. Proportional noise: z(x)~TN(0,K/|H(x)|)
% 3. Solution-dependent noise: z(x)~TN(0,K|H(x)|)
% where TN is the truncated normal distribution
d=20; % dimension of the search region
left_bound=-20; % left bound of the search region
right_bound=20; % right bound of the search region
optimal_objective_value=-1; % optimal objective value
K=10; % noise parameter
noise_bound=10; % z(x) \in [-noise_bound,noise_bound] 
%--------------------------------------------------------------------------
% The test function: 
% 1. 'sumSquares'
fcn_name='sumSquares';
%--------------------------------------------------------------------------
% The noise case can be chosen from 
% 1. 'indep_noise'
% 2. 'prop_noise'
% 3. 'solu_dep_noise'
noise_case='indep_noise';
%--------------------------------------------------------------------------

%% HYPERPARAMETERS
budget=1e5; % total number of function evaluations assigned
% alpha: learning rate for updating the mean parameter function m(theta)
% alpha=a1/(k+a2)^gamma_a
a1=20; a2=2000; gamma_a=0.99;
% beta: learning rate for updating the gradient estimator
% beta=b1/(k+b2)^gamma_b
b1=50; b2=2000; gamma_b=0.51;
% random initial annealing temperature
t=abs(feval(fcn_name,left_bound+(right_bound-left_bound)*rand(d,1)))/6;
% lambda: step-size for the annealing temperature
% lambda=1/(k+|hk*|)^gamma_t
% where |hk*| is the current best objective function value
gamma_t=0.98;

%% INITIALIZATION
mu_old=left_bound+(right_bound-left_bound)*rand(d,1); % initial mean of the sampling distribution
var_old=((right_bound-left_bound)/2)^2*ones(d,1); % initial variance of the sampling distribution
G_x_old=zeros(d,1); G_x2_old=zeros(d,1); % initial gradient estimator Gamma(X) = (X,X^2)^T

% calculating the initial gradient estimator
[eta_x_old,eta_x2_old]=truncated_mean_para_fcn(left_bound,right_bound,mu_old,var_old);
% record current best true objective values found at each step
best_H=[]; best_H(1)=-inf;

%% MAIN LOOP
fprintf('Main loop begins.\n'); tic; % count main loop time
k=0; % iteration counter
num_evaluation=0; % budget consumption
while num_evaluation+1<=budget
    %% PROGRESS REPORT
    if mod(k,1000)==0 && k>0
        fprintf('iter: %5d, eval: %5d, cur best: %8.4f, true optimum: %8.4f \n',...
            k,num_evaluation,best_H(k),optimal_objective_value);
    end
    k=k+1;
    
    %% SAMPLING
    % given the sampling parameter theta_old=(mu_old,var_old)
    % generate a sample x from the independent multivariate normal density
    X_sample=normt_rnd(mu_old,var_old,left_bound,right_bound);              
    
    %% PERFORMANCE ESTIMATIONS
    cur_H=feval(fcn_name,X_sample); % current objective function value
    cur_h=noisy_observation(cur_H,noise_case,K,noise_bound); % current noisy observation
    num_evaluation=num_evaluation+1; best_H(k)=max(best_H(end),cur_H);    
    
    %% ADAPTIVE HYPERPARAMETERS
    [alpha,beta,t]=adaptive_hyper_para(k,a1,a2,gamma_a,b1,b2,gamma_b,best_H(k),gamma_t,t);
    
    %%  GRADIENT ESTIMATOR
    G_x_new=G_x_old+beta*( exp(cur_h/t).*X_sample-exp(cur_h/t).*G_x_old);
    G_x2_new=G_x2_old+beta*(exp(cur_h/t).*X_sample.* X_sample-exp(cur_h/t).*G_x2_old );
    
    %% MEAN PRARMETER FUNCTION
    eta_x_new=eta_x_old+alpha*(G_x_new-eta_x_old);
    eta_x2_new=eta_x2_old+alpha*(G_x2_new-eta_x2_old);
    
    %% SAMPLING PARAMETER UPDATING
    mu_new=eta_x_new;
    var_new=eta_x2_new-eta_x_new.^2 ;
    
    %% UPDATING
    G_x_old=G_x_new; G_x2_old=G_x2_new;
    eta_x_old=eta_x_new; eta_x2_old=eta_x2_new;
    mu_old=mu_new; var_old=var_new;
    
    %% VISUALIZATION
    if mod(k,1000)==0
        plot(best_H,'r-o'); % current best
        hold on
        cur_size=size(best_H);
        optimal_line=optimal_objective_value*ones(cur_size(2),1);
        plot(optimal_line,'k:','LineWidth',5); % true optimal value
        xlabel('Number of function evaluations')
        ylabel('Objective function value')
        title(sprintf('<%s function>   Iteration: %5d  Evaluation: %5d',fcn_name,k,num_evaluation));
        legend('MTS-MARS','True optimal value','Location','southeast');
        ylim([best_H(1)*1.1 1000]);
        grid on
        drawnow;
    end
end

%% FINAL REPORT
fprintf('iter: %5d, eval: %5d, cur best: %8.4f, true optimum: %8.4f \n',...
    k,num_evaluation,best_H(k),optimal_objective_value);
fprintf('Main loop ends \n');
tMainLoop=toc; % count main loop time
fprintf('Main loop takes %8.4f seconds \n',tMainLoop);
