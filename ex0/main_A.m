clear all
close all
clc

rng(1)

f = @(x) x.^2 + 2*x;
xmin = -1.5; xmax = 0.5;

n_samp = 50;
xx = linspace(xmin,xmax,n_samp)';

plot(xx,f(xx));

lengthscale = 2;
kernel = @(x1,x2) exp(-dist(x1,x2').^2 / (2*lengthscale^2));

Gamma = 8;

X = [-1.2; 0; 0.2];
fX = f(X);

delta_bar = 0.1; aug = 1;
d = numel(X);
del = rand(d,1)*2*(delta_bar*aug) - (delta_bar*aug);

y = fX + del;

for i = 1:size(xx,1)
    [ub, lb] = opt_bnd(xx(i), [X y], kernel, Gamma, delta_bar);
    UB(i) = ub; LB(i) = lb;
    if mod(i,10) == 0, disp(i); end
end
disp('Done optimal!')

%%

close all
figure

fill([xx' fliplr(xx')], [UB fliplr(LB)], 'k', 'FaceColor', '#0072BD', 'EdgeColor', '#0072BD', 'facealpha', 0.1, 'linewidth', 1); hold on
plot(xx,f(xx), 'k--', 'linewidth', 1.5); hold on;

plot(X(1),y(1)-0.05,'o','markersize',6,'MarkerFaceColor', 'g', 'MarkerEdgeColor', 'g')
plot(X(2),y(2)-0.08,'o','markersize',6,'MarkerFaceColor', 'g', 'MarkerEdgeColor', 'g')
plot(X(3),y(3)+0.2,'o','markersize',6,'MarkerFaceColor', 'g', 'MarkerEdgeColor', 'g')

xlim([xmin xmax]); ylim([-2 2]);
xticks([]); yticks([]);
set(gcf,'Color','w','Position',[10 10 350 250]);
set(gca,'box','off','xcolor','w','ycolor','w')

hline_loc = -1.9;
yline(hline_loc ,'linewidth',1.5); hold on
plot([X(2) X(2)],[-2 1.7],'k--','linewidth',1)
plot([X(3) X(3)],[-2 1.7],'k--','linewidth',1)

text(-0.9,-0.7,'$f^\star$','fontsize',16,'Interpreter','latex');
text(-0.65,0.15,'$C(x)$','fontsize',16,'Interpreter','latex');
text(-0.6,-1.33,'$B(x)$','fontsize',16,'Interpreter','latex');
text(X(2)-0.035,hline_loc-0.25,'$x_i$','fontsize',16,'Interpreter','latex');
text(X(3)-0.035,hline_loc-0.25,'$x_j$','fontsize',16,'Interpreter','latex');
text(X(2)+0.02,fX(2)-0.28,'$y_i$','fontsize',16,'Interpreter','latex');
text(X(3)+0.02,fX(3)-0.39,'$y_j$','fontsize',16,'Interpreter','latex');

obs = findall(gca,'Type','Line','Marker','o');
for i = 1:numel(obs), uistack(obs(i),'top'); end

exportgraphics(gcf, ['ex0_A.pdf'], 'ContentType', 'vector')

%
% legend('sub-optimal','', 'optimal', 'AutoUpdate', 'off','NumColumns',2,'location','southeast')
% 
% grid on
% xlim([-10 10]); ylim([-60 40]);
% xticks(-10:5:10); yticks(-60:20:40);
% xl = xlabel('$z_2$','Interpreter','latex'); yl = ylabel('$f(z_1,z_2)$','Interpreter','latex'); 
% xl.Position(2) = xl.Position(2) - abs(xl.Position(2) * 0.04); 
% set(gcf,'Color','w','Position',[100 100 350 250]);
% 
% set(gca,'FontWeight','normal',...
%         'FontSize', 16,... 
%         'FontName','Serif',...
%         'TitleFontWeight','normal',...
%         'TitleFontSize',1,...
%         'Position', [0.166,0.152,0.73,0.75]);
%     
% if strcmp(sett, 'rnd')
%     title('Random sampling,           ','FontName', 'Serif', 'FontSize',16); 
%     text(3.3,45,'$\bar \delta = 1$','fontsize',16,'Interpreter','latex')
% elseif strcmp(sett, 'grid')
%     title('Grid sampling,           ','FontName', 'Serif', 'FontSize',16); 
%     text(2.4,45,'$\bar \delta = 1$','fontsize',16,'Interpreter','latex')
% end
% 
% exportgraphics(gcf, ['ex1_slice_' sett '_' num2str(delta_bar) '.pdf'], 'ContentType', 'vector')