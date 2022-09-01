function gammas = estimate_rkhs(dataset, kernel, lengthscales)
%
% Returns two (x_dim, N) arrays: 
% gammas - the estimated RKHS norms for each x_dim and time step
% lengthscales - the estimated best hyperparameters

    jitter = 1e-5;

    nx = size(dataset, 1); % number of states
    nu = size(dataset{1,1}, 2) - (nx+1); % number of control inputs
    nsteps = size(dataset, 2);  % num of steps 
    
    for time_step = 1:nsteps

        for x_dim = 1:nx
        
            data = dataset{x_dim,time_step};
            norm_ests = [];
            
            lengthscale = lengthscales(x_dim,time_step);

            % gathering data
            N = size(data,1);
            X = data(:,1:end-1);
            y = data(:,end);

            % computing the kernel matrix
            K = kernel(X,X,lengthscale) + jitter*eye(N);

            % estimating the RKHS norm from below 
            norm = sqrt((y'/K)*y);
            gammas(x_dim,time_step) = norm;
            
        end
        
    end

    disp('Done estimating the RKHS norms!')

end