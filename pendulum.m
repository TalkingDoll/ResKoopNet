clear
close all
clc

%%% UNCOMMENT THE FOLLOWING TO PERFORM COMPUTATIONS TO SIMULATE DATA %%%%%
%%
delta_t=0.5; % 0.5
ODEFUN=@(t,y) [y(2);-sin(y(1))];
options = odeset('RelTol',1e-16,'AbsTol',1e-16);

%% set up the computational grid for integration
M1=16; % 100
M2=M1;
% N1=5; N2=10;
%%
L=15; % cut off
x1=linspace(-pi,pi,M1+1);
x1=x1+(x1(2)-x1(1))/2;
x1=x1(1:end-1);
x2=linspace(-L,L,M2);
% x2 = zeros(1,M2);
[X1,X2] = meshgrid(x1(1:end-1),x2);
X1=X1(:); X2=X2(:);
M=length(X1); % number of data points

number_of_intervals = 1000;  % This means 11 points in total because we include both ends
tspan = linspace(0.000001, delta_t, number_of_intervals + 1);

% Initialize the array to store the solutions at all time points
% Dimensions: M x length(tspan) x 2
DATA_ALL = zeros(M, length(tspan), 2);
pf = parfor_progress(M);
pfcleanup = onCleanup(@() delete(pf));
parfor j=1:M
    Y0=[X1(j);X2(j)];
    [~,Y]=ode45(ODEFUN,tspan,Y0,options);
    Y(:, 1) = mod(Y(:, 1)+pi, 2*pi)-pi; % Adjusting the angle to be within -0 to 2*pi
    DATA_ALL(j,:,:) = Y;  % Store the states at all time points
    parfor_progress(pf);
    % return
end

%% reshape the data
% Create temporary variables to hold the original data
temp_DATA_X = DATA_ALL(:,1:end-1,:);
temp_DATA_Y = DATA_ALL(:,2:end,:);

% Assuming DATA_ALL is of size m x n x q
[m, n, q] = size(temp_DATA_X);

% Initialize DATA_X and DATA_Y for the reshaped data
DATA_X = zeros(m*n, q);
DATA_Y = zeros(m*n, q);

% Loop through each matrix and place it in DATA_X
for i = 1:m
    startRow = (i-1)*n + 1;
    endRow = i*n;
    DATA_X(startRow:endRow, :) = squeeze(temp_DATA_X(i, :, :));
end

% Loop through each matrix and place it in DATA_Y
for i = 1:m
    startRow = (i-1)*n + 1;
    endRow = i*n;
    DATA_Y(startRow:endRow, :) = squeeze(temp_DATA_Y(i, :, :));
end




%%
save('D:\new_data_June\data_pendulum_240.mat', 'DATA_X', 'DATA_Y');
