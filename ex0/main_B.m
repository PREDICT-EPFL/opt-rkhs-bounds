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

X = [-1.2; 0.1327; 0.1327];
fX = f(X);

delta_bar = 0.05; aug = 1;
d = numel(X);
del = [0; delta_bar; -delta_bar];

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

plot(X(1),y(1)-0.01,'o','markersize',6,'MarkerFaceColor', 'g', 'MarkerEdgeColor', 'g')
plot(X(2),y(2)+0.2,'o','markersize',6,'MarkerFaceColor','g', 'MarkerEdgeColor', 'g')
plot(X(3),y(3)-0.2,'o','markersize',6,'MarkerFaceColor','g', 'MarkerEdgeColor', 'g')

xlim([xmin xmax]); ylim([-2 2]);
xticks([]); yticks([]);
set(gcf,'Color','w','Position',[10 10 350 250]);
set(gca,'box','off','xcolor','w','ycolor','w')

hline_loc = -1.9;
yline(hline_loc ,'linewidth',1.5); hold on
plot([X(2) X(2)],[-2 1.7],'k--','linewidth',1)

text(-0.9,-0.7,'$f^\star$','fontsize',16,'Interpreter','latex');
text(-0.8,0.48,'$C(x)$','fontsize',16,'Interpreter','latex');
text(-0.55,-1.33,'$B(x)$','fontsize',16,'Interpreter','latex');
text(X(2)-0.035,hline_loc-0.25,'$x_i$','fontsize',16,'Interpreter','latex');
text(X(2)+0.02,fX(2)+0.7,'$y_{i,1}$','fontsize',16,'Interpreter','latex');
text(X(3)+0.02,fX(3)-0.45,'$y_{i,2}$','fontsize',16,'Interpreter','latex');

obs = findall(gca,'Type','Line','Marker','o');
for i = 1:numel(obs), uistack(obs(i),'top'); end

exportgraphics(gcf, ['ex0_B.pdf'], 'ContentType', 'vector')
