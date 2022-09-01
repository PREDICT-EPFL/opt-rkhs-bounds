function plot_time(x_opti, x_true, ubs, lbs, x_min, x_max, fignum)
%PLOT_TIME Summary of this function goes here
%   Detailed explanation goes here

    xx = 1:size(x_opti,2);
    aug_ubs = [x_opti(:,1) ubs];
    aug_lbs = [x_opti(:,1) lbs];
    
    nx = size(x_opti,1);

    if nargin == 7
        figure(fignum)
    else
        figure
    end
    
    for state = 1:nx

        subplot(1,2,state)
        plot(xx, x_opti(state,:), 'r-', 'marker', 'o', 'linewidth', 2); hold on
        plot(xx, x_true(:,state), 'k-', 'marker', 'o', 'linewidth', 2);
        legend('model', 'ground-truth', 'AutoUpdate', 'off')
        fill([xx fliplr(xx)], [aug_lbs(state,:) fliplr(aug_ubs(state,:))], 'y', 'facealpha', 0.15, 'linewidth', 1.5);

        plot(xx, x_max(state)*ones(size(xx)), 'k--', 'linewidth', 2)
        plot(xx, x_min(state)*ones(size(xx)), 'k--', 'linewidth', 2)
        %ylim([x_min(state)*1.5 x_max(state)*1.5]);
        xlabel('time step'); xticks(1:numel(xx));
        ylabel(['x' num2str(state)]);

    end

end

