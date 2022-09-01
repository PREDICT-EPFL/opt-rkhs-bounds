function [lengthscales, losses] = cv(folds, kernel, lambda, dataset, lengthscale_range)
%CV Summary of this function goes here
%   Detailed explanation goes here

    jitter = 1e-5;

    nsteps = size(dataset,2);
    nx = size(dataset,1);

    for time_step = 1:nsteps

        for x_dim = 1:nx
        
            data = dataset{x_dim,time_step};
            
            % extracting data
            N = size(data,1);
            X = data(:,1:end-1);
            y = data(:,end);

            avg_losses = [];
            
            for lengthscale = lengthscale_range 

                for i = 1:folds
                
                    bin_size = floor(N/folds);
                    idxs = ((i-1)*bin_size+1):i*bin_size;
                    idxs_rem = setdiff(1:N,idxs);
                    
                    % computing the kernel model on the rest of the dataset
                    M = numel(idxs_rem);
                    K = kernel(X(idxs_rem,:),X(idxs_rem,:),lengthscale) + jitter*eye(M);
                    alpha = (K + N*lambda*eye(M))\y(idxs_rem); 
                    krr = @(x) kernel(X(idxs_rem,:),x,lengthscale)'*alpha;
                   
                    % estimating the loss
                    preds = krr(X(idxs,:));
                    gt = y(idxs);
                    loss(i) = (1/bin_size)*sum((preds-gt).^2);
                    
                end
                
                avg_losses = [avg_losses mean(loss)];
                
            end
            
            % save hyperparam that yielded the min loss
            min_loss = min(avg_losses);
            
            losses(x_dim,time_step) = min_loss;
            lengthscales(x_dim,time_step) = lengthscale_range(avg_losses == min_loss);
            
        end
        
    end

    disp('Done estimating the lengthscales!')
    
end

