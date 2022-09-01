function plot_errbars(x_opti, x_true, ubs, lbs, x_min, x_max, fignum)
%PLOT_TIME Summary of this function goes here
%   Detailed explanation goes here

    xx = 0:size(x_opti,2)-1;
    nx = size(x_opti,1);

    if nargin == 7
        figure(fignum)
    else
        figure
    end
    
    for state = 1:nx
        
        subplot(1,2,state)
        %plot(xx, x_opti(state,:), '-', 'marker', 'x', 'linewidth', 2, 'color', '#0072BD'); hold on
        plot(xx, x_true(:,state), '-', 'marker', 'x', 'linewidth', 2, 'color', 'k'); hold on
        %legend('model', 'ground-truth', 'AutoUpdate', 'off', 'Interpreter','latex')
        legend('ground-truth', 'AutoUpdate', 'off', 'Interpreter','latex')
        
        widths = (ubs(state,:) - lbs(state,:))/2;
        centers = (ubs(state,:) - lbs(state,:))/2 + lbs(state,:);
        e = errorbar(xx,[x_true(1,state) centers], [0 widths], 'linewidth', 2, 'color', '#0072BD', 'CapSize', 10, 'linewidth', 2);
        e.LineStyle = 'none';
        
        plot(xx, x_max(state)*ones(size(xx)), 'k--', 'linewidth', 2)
        plot(xx, x_min(state)*ones(size(xx)), 'k--', 'linewidth', 2)
        xlabel('time step'); xticks(1:numel(xx));
        ylabel(['x' num2str(state)]);
        
        h = findall(gcf,'Type','Line');
        for i = 1:numel(h)
            uistack(h(i),'top');
        end

    end

end

