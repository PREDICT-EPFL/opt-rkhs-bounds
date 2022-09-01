function x_traj = sim_sys_dt(dynamics,x0,u,T_samp)
%SIM_SYS_DT Summary of this function goes here
%   Detailed explanation goes here
    
    N = size(u,2);
    T = [0 N*T_samp];
    time = linspace(T(1),T(2),N+1);
        
    [t,x] = ode45(@(t,x) dynamics(t,x,u), T, x0);
    idx = find(t>=time(1), 1, 'first');
    x_traj(1,:) = x(idx,:);
    for n = 2:N+1
        idx = find(t>=time(n), 1, 'first');
        x_traj(n,:) = x(idx,:);
    end

end

