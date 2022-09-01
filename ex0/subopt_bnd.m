function [ub,lb] = subopt_bnd(x, data, kernel, gamma, del_bar)
% For a given query point x, computes the uncertainty around f(x) sub-optimally
% based on the specified kernel, ground-truth RKHS estimate and noise bound
%
% Returns two scalars, so that  lb <= f(x) < ub

    function bound = compute_bound(x, dir)
        
        % Build nominal KRR model
        jitter = 1e-5;
        lambda = 1e-2;

        X = data(:,1:end-1);
        y = data(:,end);

        n_data = size(X,1);
        K = kernel(X,X) + jitter*eye(n_data);
        alpha = (K + n_data*lambda*eye(n_data))\y; 
        krr = @(x) alpha'*kernel(X,x);

        % Compute Delta
        nu = sdpvar(n_data,1);
        cost = 1/4*(nu'*K*nu) + nu'*y + del_bar*norm(nu,1);
        temp = optimize([], cost, sdpsettings('verbose', 0, 'solver', 'gurobi'));
        nu = value(nu);
        Delta = value(cost);

        % Compute the indivdual terms and the bounds
        s = @(x) y'*(K\kernel(X,x));
        s_norm = sqrt(y'*(K\y));
        P = @(x) sqrt(real(diag(kernel(x,x) - (kernel(x,X)/K)*kernel(X,x))));
        p = @(x) (del_bar * vecnorm((kernel(x,X)/K)', 1))';
        q = @(x) abs(s(x) - krr(x));

        S = @(x) P(x).*sqrt(gamma^2 + Delta) + abs(p(x)) + q(x);

        if strcmp(dir,'up'), bound = krr(x) + S(x);
        elseif strcmp(dir,'dw'), bound = krr(x) - S(x); end
    
    end
    
    ub = compute_bound(x, 'up');
    lb = compute_bound(x, 'dw');
    
end