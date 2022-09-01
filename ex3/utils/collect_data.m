function dataset = collect_data(D, dynamics, xmin, xmax, umin, umax, delta_bar, method, plts, surrogate_model)
%
% Returns a {x_dim, N} cell array.
%
% Each cell is the data matrix associated with a system state dimension and
% a certain time-step. In other words, each datum (row of the matrix)
% contains the current state, the control sequence, and a (noise corrupted)
% component of the next state.
%
% The first cell is a [D, x_dim+(u_dim)+1] matrix.
% The second cell is a [D, x_dim+(u_dim*2)+1] matrix.
% ...
%
% method = 'rnd', 'grid', 'ocp'
%  
    global T_samp
    plot_aug = 1.5;
    N = numel(D);
    
    if ~exist('plts')
        plts = false;
    end
    
    % check the data vector order
    for step = 1:(numel(D)-1)
       if D(step) > D(step+1), error('The D vector elements must be in increasing order.'); end
    end
     
    if strcmp(method, 'rnd')
    
        D_max = max(D);
        
        % generate initial states and rnd control sequences
        [x0,u] = gen_feats(N, D_max, xmin, xmax, umin, umax, method);
        
        full_dataset = run_dynamics(@(t,x,u) dynamics(t,x,u), x0, u, delta_bar, plts);
        dataset = dataset_reduce(full_dataset, D);
    
    elseif strcmp(method, 'grid')
    
        D_max = max(D);
        
        % generate initial states and rnd control sequences
        [x0,u] = gen_feats(N, D_max, xmin, xmax, umin, umax, method);
        
        full_dataset = run_dynamics(@(t,x,u) dynamics(t,x,u), x0, u, delta_bar, plts);
        dataset = dataset_reduce(full_dataset, D);
    
        
    elseif strcmp(method, 'ocp')
            
        D_max = max(D);
        
        safety_margin = 0.92; % stay a little away from the constraints 
        
        % generate the state feats only
        [x0,~] = gen_feats(N, D_max, xmin*(1/safety_margin), xmax*safety_margin, umin, umax, method);
        
        % setup the OCP
        [opti, X0, X, U] = build_ocp(surrogate_model, N);
        for d = 1:D_max
            
            % generate the control feats through the OCP
            % old way 
            % [x_opti, u_opti] = ocp(surrogate_model, x0(:,d), N);
            opti.set_value(X0, x0(:,d));
            sol = opti.solve();
            u_opti = sol.value(U(:,:)); 
                        
            u(:,:,d) = u_opti;
            if mod(d,5) == 0, disp([num2str(d) ' data collected...']); end
        end
        
        full_dataset = run_dynamics(@(t,x,u) dynamics(t,x,u), x0, u, delta_bar, plts);
        dataset = dataset_reduce(full_dataset, D);
     
    end
    
    disp('Done collecting data!')
    
    function full_dataset = run_dynamics(dynamics, x0, u, delta_bar, plts)
        
        % simulate the system to get its evolution (labels)
        D_aug = size(x0,2)*ones(1,size(u,2));
        T = [0 N*T_samp];
        time = linspace(T(1),T(2),N+1);
        x_steps = zeros(N+1,2);

        for d = 1:D_aug

            [t,x] = ode45(@(t,x) dynamics(t,x,u(1,:,d)), T, x0(:,d));

            idx = find(t>=time(1), 1, 'first');
            x_steps(1,:) = x(idx,:);

            for n = 2:N+1

                idx = find(t>=time(n), 1, 'first');
                x_steps(n,:) = x(idx,:);
                
                % add noise
                del = rand(2,1) * 2*delta_bar - delta_bar;

                % assemble dataset
                full_dataset{1,n-1}(d,:) = [x_steps(1,:) u(1,1:n-1,d) (x_steps(n,1) + del(1))];
                full_dataset{2,n-1}(d,:) = [x_steps(1,:) u(1,1:n-1,d) (x_steps(n,2) + del(2))];
                
            end
            
        end
        
    end

    function dataset = dataset_reduce(full_dataset, D)
    
        nx = size(full_dataset, 1);
        
        for step = 1:N
            for state = 1:nx
            
                % drop aligned if possible
                idxs = 1:(floor(D_max/D(step))):D_max;
                dataset{state,step} = full_dataset{state,step}(idxs,:);
                
                % drop the remaining randomly
                rng(1)
                for i = 1:(size(dataset{state,step},1)-D(step))
                    idx = randi([1,size(dataset{state,step},1)]);
                    dataset{state,step}(idx,:) = [];
                end
                
            end
        end
    
    end

end