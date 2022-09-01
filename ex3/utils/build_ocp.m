function [opti, X0, X, U] = build_ocp(surrogate_model, N)
%OCP Summary of this function goes here
%   Detailed explanation goes here

    % import target values and costs
    constants

    % reference
%     xs = [0; 0];
%     us = 0;

    % costs
%     Q = 10*eye(nx);
%     R = 0.5*eye(nu);
%     P = 50*eye(nx); 

    import casadi.*
    opti = casadi.Opti();
    
    X0 = opti.parameter(nx,1); 
    X = opti.variable(nx,N+1);
    U = opti.variable(nu,N);
    J = 0;

    % OCP formulation
    opti.subject_to(X(:,1) == X0);   
    for t = 1:N

        % dynamic constraints
        z = [X(:,t)' U(:,t)];   
        opti.subject_to(X(:,t+1) == surrogate_model(z));   
        
        % state and input box constraints
        opti.subject_to(x_min <= X(:,t) <= x_max);
        opti.subject_to(u_min <= U(:,t) <= u_max);

        % stage cost
        J = J + ((X(:,t)-xs)'*Q*(X(:,t)-xs) + (U(:,t)-us)'*R*(U(:,t)-us));

    end
    % terminal cost
    J = J + (X(:,N+1)-xs)'*P*(X(:,N+1)-xs);

    opti.minimize(J);
    ops = struct;
    ops.ipopt.tol = 1e-3;
    ops.ipopt.print_level = 0;
    ops.print_time = false;    
    opti.solver('ipopt', ops);

end