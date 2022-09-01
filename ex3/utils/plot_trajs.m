function plot_trajs(x_pred, x_true, xs, x_min, x_max, fignum)
%PLOT_TIME Summary of this function goes here
%   Detailed explanation goes here

    if nargin == 6
        figure(fignum)
    else
        figure
    end

    plot(x_pred(1,:), x_pred(2,:), '-*', 'linewidth', 2); hold on;
    plot(x_true(:,1), x_true(:,2), '-*', 'linewidth', 2);
    legend('predicted', 'ground-truth', 'AutoUpdate', 'off')
    plot(xs(1), xs(2), 'kx', 'markersize', 10);
    grid on; set(gcf,'color','w');
    X_feas = Polyhedron([1 0; -1 0; 0 1; 0 -1],[x_max(1); -x_min(1); x_max(2); -x_min(2)]);
    plot(X_feas, 'color', 'red', 'linewidth', 2, 'linestyle', '--', 'alpha', 0);

end

