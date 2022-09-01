clear all
close all
clc

addpath('./utils')
constants

rng(1)
global T_samp 

kernel = @(x1,x2,lengthscale) exp(-dist(x1,x2').^2 / (2*lengthscale^2));
dynamics = @(t,x,u) cstr(t,x,u);

%% PART1

% Gather a dataset to estimate the kernel hyperparams and the RKHS norms
N = 8;
D = 400*ones(1,N);
dataset = collect_data(D, @(t,x,u) dynamics(t,x,u), x_min, x_max, u_min, u_max, 0, 'grid');

% Estimate the kernel lengthscales
lengthscale_range = 2:0.1:5;
folds = 5;
lambda = 0.0000001;
[lengthscales, losses] = cv(folds, kernel, lambda, dataset, lengthscale_range);

% Estimate ground-truths RKHS norms
gammas = estimate_rkhs(dataset, @(x1,x2,lengthscale) kernel(x1,x2,lengthscale), lengthscales);

%% PART2

% Learn a 1-step ahead KRR to be the dynamics surrogate
dataset = collect_data(400, @(t,x,u) dynamics(t,x,u), x_min, x_max, u_min, u_max, delta_bar, 'grid');

for nx = 1:nx
    
    X{nx} = dataset{nx,1}(:,1:end-1);
    y{nx} = dataset{nx,1}(:,end);

    jitter = 1e-8;
    n_data = size(X{nx},1);
    K{nx} = kernel(X{nx},X{nx},lengthscales(nx,1)) + jitter*eye(n_data);
    alpha{nx} = (K{nx} + n_data*lambda*eye(n_data))\y{nx}; 
    
    kernel_casadi = @(x1,x2,lengthscale) exp(-diag((x1-repmat(x2,size(x1,1),1))*(x1-repmat(x2,size(x1,1),1))') / (2*lengthscale^2));
    krr{nx} = @(x) alpha{nx}'*kernel_casadi(X{nx},x,lengthscales(nx,1));

end

surrogate_model = @(z) [krr{1}(z); krr{2}(z)];
disp([newline 'Done learning a surrogate model!'])

%% PART 3

% Compute an optimal control sequence
% Choose an initial condition
x0 = [1.23; 0.81]; 
x0 = [1.365; 1.68];
x0 = [2.64; 1.33]; 
x0 = [2.7; 0.544]; 

[x_opti, u_opti] = ocp(surrogate_model, x0, N);
x_true = sim_sys_dt(dynamics,x_opti(:,1),u_opti,T_samp);

plot_trajs(x_opti, x_true, xs, x_min, x_max, 1);
disp([newline 'Done computing an optimal control sequence!'])

%% PART 4

% Collect data for uncertainty quantification 
D = 800*ones(1,N);
plts = false;
datasets = collect_data(D, @(t,x,u) dynamics(t,x,u), x_min, x_max, u_min, u_max, delta_bar, 'ocp', plts, surrogate_model);

% Verify how far apart the data are
min_features_dists(datasets);
min_queries_dists(datasets, x_opti, u_opti);

%% PART 5

% Augmentation factors for the norm and the noise bound
delta_aug = 1.2; 
gamma_aug = 1.2;

disp([newline 'Computing the sub-optimal bounds:'])

ubs_subopt = zeros(nx,N);
lbs_subopt = zeros(nx,N);

for step = 1:N    
    for state = 1:nx
    
        data = datasets{state,step};
        gamma = gammas(state,step);
        z = [x_opti(:,1)' u_opti(1:step)];
        
        [ub, lb] = subopt_bnd(z, data, @(x1,x2) kernel(x1,x2,lengthscales(state,step)), gamma*gamma_aug, delta_bar*delta_aug);
        ubs_subopt(state,step) = ub;
        lbs_subopt(state,step) = lb;
        
    end
    disp(['Step ' num2str(step) ' done...'])
end

plot_phase(x_opti, x_true, ubs_subopt, lbs_subopt, x_min, x_max, 2)
plot_errbars(x_opti, x_true, ubs_subopt, lbs_subopt, x_min, x_max, 3);

disp([newline 'Computing the optimal bounds:'])

ubs_opt = zeros(nx,N);
lbs_opt = zeros(nx,N);

for step = 1:N
    for state = 1:nx
    
        data = datasets{state,step};
        gamma = gammas(state,step);
        z = [x_opti(:,1)' u_opti(1:step)];
        
        [ub, lb] = opt_bnd(z, data, @(x1,x2) kernel(x1,x2,lengthscales(state,step)), gamma*gamma_aug, delta_bar*delta_aug);
        ubs_opt(state,step) = ub;
        lbs_opt(state,step) = lb;
        
    end
    disp(['Step ' num2str(step) ' done...'])
end

plot_phase(x_opti, x_true, ubs_opt, lbs_opt, x_min, x_max, 5)
plot_errbars(x_opti, x_true, ubs_opt, lbs_opt, x_min, x_max, 6)

% EOF