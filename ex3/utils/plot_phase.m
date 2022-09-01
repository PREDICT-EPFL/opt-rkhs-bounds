function plot_phase(x_opti, x_true, ubs, lbs, x_min, x_max, fignum)
%PLOT_PHASE Summary of this function goes here
%   Detailed explanation goes here

    N = size(x_opti,2)-1;

    % closed-loop trajectory
    if nargin == 7
        figure(fignum)
    else
        figure
    end
    
    %plot(x_opti(1,:),x_opti(2,:),'r-x','linewidth',2); hold on
    plot(x_true(:,1), x_true(:,2), 'k-x','linewidth',2); hold on
    %legend('model', 'ground-truth', 'AutoUpdate', 'off')
    legend('ground-truth', 'AutoUpdate', 'off')
    for i = 1:N
        box = Polyhedron([1 0; -1 0; 0 1; 0 -1],[ubs(1,i); -lbs(1,i); ubs(2,i); -lbs(2,i)]);
        plot(box, 'color', 'red', 'linewidth', 1, 'linestyle', '-', 'alpha', 0.05);
        hold on
    end
    axis([x_min(1) x_max(1) x_min(2) x_max(2)])
    grid on; set(gcf,'color','w');
    X_feas = Polyhedron([1 0; -1 0; 0 1; 0 -1],[x_max(1); -x_min(1); x_max(2); -x_min(2)]);
    plot(X_feas, 'color', 'red', 'linewidth', 1.5, 'linestyle', '--', 'alpha', 0)
    
    xlim([0.9 3.1]); ylim([0.45 2.05])

end

