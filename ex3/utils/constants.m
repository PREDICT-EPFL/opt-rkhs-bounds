jitter = 1e-8;

% CSTR bounds
x1_min = 1;
x1_max = 3;
x2_min = 0.5;
x2_max = 2;
u_min = 3;
u_max = 25; 

nx = 2; nu = 1;
x_min = [x1_min; x2_min];
x_max = [x1_max; x2_max];

% CSTR equilibrium
xs = [2.14; 1.09];
us = 14.19;

% OCP weights
Q = diag([0.2 1]); 
R = 0.1;
P = [14.46 13.56; 13.56 62.22];

T_samp = 30;
delta_bar = 0.05; 