clear all
close all
clc

rng(3)

a = 0.8; b = 8; c = 0.8;
f = @(z1,z2) 1 - a.*(z1.^2) + z2 + b*sin(c.*z2);

z_min = [-10; -10];
z_max = [10; 10];
nz = 2;

lengthscale = 5;
kernel = @(x1,x2) exp(-dist(x1,x2').^2 / (2*lengthscale^2));

Gamma = 1200;

d = 100;
Z = collect_data(d, z_min, z_max, 'rnd');

fZ = f(Z(:,1),Z(:,2));

delta_bar = 1;
del = rand(d,1)*2*delta_bar - delta_bar;

y = fZ + del;

%% 3D results

z_gran = [0.8 0.8];
[ZZ1, ZZ2] = meshgrid(z_min(1):z_gran(1):z_max(2),z_min(2):z_gran(2):z_max(2));
ZZ = [ZZ1(:) ZZ2(:)];

fZZ = f(ZZ1(:), ZZ2(:));

for i = 1:size(ZZ,1)
    [ub, lb] = opt_bnd(ZZ(i,:), [Z y], kernel, Gamma, delta_bar);
    UB(i) = ub; LB(i) = lb;
    if mod(i,10) == 0, disp(i); end
end
disp('Done optimal!')

for i = 1:size(ZZ,1)
    [ub, lb] = subopt_bnd(ZZ(i,:), [Z y], kernel, Gamma, delta_bar);
    SUB(i) = ub; SLB(i) = lb;
    if mod(i,10) == 0, disp(i); end
end
disp('Done suboptimal!')

%%
close all

sett = 'rnd';

surf(ZZ1,ZZ2,reshape(UB,size(ZZ1)), 'FaceColor', '#0072BD' ,'FaceAlpha', 0.3, 'EdgeColor', 'none'); hold on 
surf(ZZ1,ZZ2,reshape(fZZ,size(ZZ1)), 'FaceColor', 'k', 'FaceAlpha', 0.3, 'EdgeColor', 'none')

xlim([-10 10]); ylim([-10 10]); zlim([-100 50]);
xticks(-10:5:10); yticks(-10:5:10); zticks(-100:50:50);
set(gcf,'Color','w','Position',[100 100 360 250]);

if strcmp(sett, 'rnd')
    title('Random sampling,           ','FontName', 'Serif', 'FontSize',16); 
    text(-8,22.4,31,['$\bar \delta = ' num2str(delta_bar) '$'],'fontsize',16,'Interpreter','latex')
elseif strcmp(sett, 'grid')
    title('Grid sampling,           ','FontName', 'Serif', 'FontSize',16); 
    text(-8,19.8,38,['$\bar \delta = ' num2str(delta_bar) '$'],'fontsize',16,'Interpreter','latex')
end

xl = xlabel('$z_1$','Interpreter','latex'); yl = ylabel('$z_2$','Interpreter','latex'); zl = zlabel('$f(z_1,z_2)$','Interpreter','latex');
xl.Position(1) = xl.Position(1) + abs(xl.Position(1) * 0.1); 
xl.Position(2) = xl.Position(2) + abs(xl.Position(2) * 0.15);
yl.Position(1) = yl.Position(1) + abs(yl.Position(1) * 1.5); 
yl.Position(2) = yl.Position(2) + abs(yl.Position(2) * 6.5);
yl.Position(3) = yl.Position(3) - abs(yl.Position(3) * 0.1);
zl.Position(1) = zl.Position(1) - abs(zl.Position(1) * 1.4); 
zl.Position(2) = zl.Position(2) + abs(zl.Position(2) * 0.2);
zl.Position(3) = zl.Position(3) - abs(zl.Position(3) * 4);

set(gca,'FontWeight','normal',...
        'FontSize', 16,... 
        'FontName','Serif',...
        'TitleFontWeight','normal',...
        'TitleFontSize',1,...
        'view',[32,23]);

set(gca, 'Position', [0.16,0.124,0.66,0.78])
    
exportgraphics(gcf, ['ex1_' sett '_' num2str(delta_bar) '.pdf'], 'ContentType', 'vector')

%% 2D results

slice = -3;

ZZ2 = linspace(z_min(2),z_max(2),100);
ZZ1 = slice*ones(size(ZZ2));
ZZ = [ZZ1(:) ZZ2(:)];

fZZ = f(ZZ1(:), ZZ2(:));

for i = 1:size(ZZ,1)
    [ub, lb] = opt_bnd(ZZ(i,:), [Z y], kernel, Gamma, delta_bar);
    UB(i) = ub; LB(i) = lb;
    if mod(i,10) == 0, disp(i); end
end
disp('Done optimal!')

for i = 1:size(ZZ,1)
    [ub, lb] = subopt_bnd(ZZ(i,:), [Z y], kernel, Gamma, delta_bar);
    SUB(i) = ub; SLB(i) = lb;
    if mod(i,10) == 0, disp(i); end
end
disp('Done suboptimal!')

%%
close all

sett = 'rnd';

figure

xx = ZZ2(ZZ1(:) == slice);
gt = fZZ(ZZ1(:) == slice);
ub = UB(ZZ1(:) == slice);
lb = LB(ZZ1(:) == slice);
sub = SUB(ZZ1(:) == slice);
slb = SLB(ZZ1(:) == slice);

fill([xx fliplr(xx)], [ub fliplr(sub)], 'k', 'FaceColor', 'g', 'EdgeColor', 'g', 'facealpha', 0.15, 'linewidth', 1); hold on
fill([xx fliplr(xx)], [slb fliplr(lb)], 'k', 'FaceColor', 'g', 'EdgeColor', 'g', 'facealpha', 0.15, 'linewidth', 1); hold on
fill([xx fliplr(xx)], [lb fliplr(ub)], 'k', 'FaceColor', '#0072BD', 'EdgeColor', '#0072BD', 'facealpha', 0.15, 'linewidth', 1);
plot(xx,gt, 'k--', 'linewidth', 1.8); hold on;

legend('sub-optimal','', 'optimal', 'AutoUpdate', 'off','NumColumns',2,'location','southeast')

grid on
xlim([-10 10]); ylim([-60 40]);
xticks(-10:5:10); yticks(-60:20:40);
xl = xlabel('$z_2$','Interpreter','latex'); yl = ylabel('$f(z_1,z_2)$','Interpreter','latex'); 
xl.Position(2) = xl.Position(2) - abs(xl.Position(2) * 0.04); 
set(gcf,'Color','w','Position',[100 100 350 250]);

set(gca,'FontWeight','normal',...
        'FontSize', 16,... 
        'FontName','Serif',...
        'TitleFontWeight','normal',...
        'TitleFontSize',1,...
        'Position', [0.166,0.152,0.73,0.75]);
    
if strcmp(sett, 'rnd')
    title('Random sampling,           ','FontName', 'Serif', 'FontSize',16); 
    text(3.3,45,'$\bar \delta = 1$','fontsize',16,'Interpreter','latex')
elseif strcmp(sett, 'grid')
    title('Grid sampling,           ','FontName', 'Serif', 'FontSize',16); 
    text(2.4,45,'$\bar \delta = 1$','fontsize',16,'Interpreter','latex')
end

%exportgraphics(gcf, ['ex1_slice_' sett '_' num2str(delta_bar) '.pdf'], 'ContentType', 'vector')