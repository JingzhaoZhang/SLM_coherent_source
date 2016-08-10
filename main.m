%clear all;
tag = '804_600by800_A';


%% Setup params
% All length value has unit meter in this file.
% The 3d region is behind lens after SLM. 


resolutionScale = 1; % The demagnification scale of tubelens and objective. f_tube/f_objective
lambda = 1e-6;  % Wavelength
focal_SLM = 0.2; % focal length of the lens after slm.
psSLM = 20e-6;      % Pixel Size (resolution) at the scattered 3D region
Nx = 800;       % Number of pixels in X direction
Ny = 600;       % Number of pixels in Y direction


psXHolograph = lambda * focal_SLM/ psSLM / resolutionScale / Nx;      % Pixel Size (resolution) at the scattered 3D region
psYHolograph = lambda * focal_SLM/ psSLM / resolutionScale / Ny;      % Pixel Size (resolution) at the scattered 3D region

useGPU = 1;     % Use GPU to accelerate computation. Effective when Nx, Ny is large (e.g. 600*800).


z = [-100 : 4: 100] * 3e-4 ;   % Depth level requested in 3D region.
nfocus = 25;                % z(nfocus) denotes the depth of the focal plane.
thresholdh = 2e9;          % Intensity required to activate neuron.
thresholdl = 0;             % Intensity required to not to activate neuron.

%% Point Targets
% radius = 0.5e-4; % Radius around the point.
% step = Nx/16 * psXHolograph;
% epsilon = 0;
% targets = [ step, step, z(10); step, 0, z(10); step, -step, z(10); 0, step, z(10);...
%     0, 0, z(10); 0, -step, z(10);step+epsilon, step, z(40); step+epsilon, 0, z(40); step+epsilon, -step, z(40); 0+epsilon, step, z(40);...
%     0+epsilon, 0, z(40); 0+epsilon, -step, z(40);-step-epsilon, step, z(25); -step-epsilon, 0, z(25); -step-epsilon, -step, z(25); 0-epsilon, step, z(25);...
%     0-epsilon, 0, z(25); 0-epsilon, -step, z(25);] ; % Points where we want the intensity to be high.
% zratio = 1;

% zratio = 5;
% radius = 3e-4;
% targets = [ 0, 0, z(5); 0, 0, z(45);];
% % 
% % 
% Masks = zeros(Nx, Ny, numel(z));
% if useGPU
%     Masks = gpuArray(Masks);
% end
% for i = 1 : numel(z)
%     Masks(:,:,i) = generatePointMask( targets, radius, z(i), Nx, Ny, psXHolograph,psYHolograph, zratio, useGPU);
% end
% maskfun = @(i1, i2) Masks(:,:,i1:i2);


%% Complex Target
load('largeAB');
step = 2e-3;
zrange1 = [z(25) - 0.01 * step, z(25)];
zrange2 = [-1, -1];

% zrange1 = [z(5) - 0.01 * step, z(5)];
% zrange2 = [z(45) - 0.01 * step, z(45)];


Masks = zeros(Nx, Ny, numel(z));
if useGPU
    Masks = gpuArray(Masks);
end
for i = 1 : numel(z)
    Masks(:,:,i) = generateComplexMask( z(i), Nx, Ny, maskA', zrange1, maskB', zrange2);
end
maskfun = @(i1, i2) Masks(:,:,i1:i2);


%% Kernel Function
HStacks = zeros(Nx, Ny, numel(z));
if useGPU
    HStacks = gpuArray(HStacks);
end
for i = 1 : numel(z)
    HStacks(:,:,i) = GenerateFresnelPropagationStack(Nx, Ny, z(i)-z(nfocus), lambda, psXHolograph,psYHolograph, useGPU);
end
kernelfun = @(i1, i2) HStacks(:,:,i1:i2);
% kernelfun = @(i) GenerateFresnelPropagationStack(Nx, Ny, z(i)-z(nfocus), lambda,psXHolograph,psYHolograph, focal_SLM, useGPU);


%% Pick Source Initialization method

% The starting point. reshape(x0(1:Nx*Ny), [Nx, Ny]) encodes the phase on
% SLM in rads. Normally initialized to zeros. reshape(x0(1+Nx*Ny:end), [Nx, Ny])
% encodes the source intensity. Need to be nonnegative.
x0 = zeros(Nx*Ny, 1);

%This sets a coherent light source.
% intensity = 1;
% source = sqrt(intensity) * ones(Nx, Ny);
% tag = [tag, '_uniform'];

%This sets a coherent light source.
std = Nx/2;
mu = [0 0];
Sigma = [std.^2 0; 0 std.^2];
x1 = [1:Nx] - Nx/2; x2 =  [1:Ny] - Ny/2;
[X1, X2] = meshgrid(x2,x1);
F = mvnpdf([X1(:) X2(:)],mu,Sigma);
source = reshape(F,length(x1),length(x2));
source = source/max(max(source));
total_energy = sum(sum(source.^2))
tag = [tag, '_gaussian'];
figure();imagesc(source);title('Gaussian source');drawnow;


%source = ones(Nx, Ny);
tic;


%% Optimization phase 1
% Scale the gradient of phase by 1 and the gradient of source by 0.
% This makes sure that only the phase is updated in each iteration.


f = @(x)SourceFunObj( x, source, z, Nx, Ny, thresholdh, thresholdl, maskfun, kernelfun, useGPU);


matlab_options = optimoptions('fmincon','GradObj','on', 'display', 'iter', ...
    'algorithm','interior-point','Hessian','lbfgs', 'MaxFunEvals', 200, 'MaxIter', 200,...
    'TolX', 1e-20, 'TolFun', 1e-12);
lb = -inf(Nx*Ny, 1);
ub = inf(Nx*Ny, 1);
nonlcon = [];
phase = fmincon(f,x0,[],[],[],[],lb,ub,nonlcon,matlab_options);


phase = reshape(phase, [Nx, Ny]);
phase = mod(phase, 2*pi) - pi;
hologram = floor(mod(phase, 2*pi)/2/pi * 255);
toc;
%% Optimization phase 2 
% The following part optimizes phase and source at the same time.
% ratio_phase = 1;
% ratio_source = 1; 
% 
% 
% coherent init
% phase_source1(Nx*Ny+1:end) = source1(:);
% 
% 
% f = @(x)SourceFunObj(x, z, Nx, Ny, thresholdh, thresholdl, maskfun, kernelfun, useGPU, ratio_phase, ratio_source);
% 
% phase_source = minFunc(f, phase_source, options);
% 
% matlab_options = optimoptions('fmincon','GradObj','on', 'display', 'iter', ...
%     'algorithm','interior-point','Hessian','lbfgs', 'MaxFunEvals', 150, 'MaxIter', 50,...
%     'TolX', 1e-20, 'TolFun', 1e-20);
% 
% phase_source2 = fmincon(f,phase_source1,[],[],[],[],lb,ub,nonlcon,matlab_options);
% 
% 
% phase2 = reshape(phase_source2(1:Nx*Ny), [Nx, Ny]);
% source2 = reshape(phase_source2(Nx*Ny+1:end), [Nx, Ny]);

toc;

%% plot
Ividmeas = zeros(Nx, Ny, numel(z));
usenoGPU = 0;
figure();
for i = 1:numel(z)
    HStack = GenerateFresnelPropagationStack(Nx,Ny,z(i) - z(nfocus), lambda, psXHolograph,psYHolograph, usenoGPU);
    imagez = fresnelProp(phase, source, HStack);
    Ividmeas(:,:,i) = imagez;
    imagesc(imagez);colormap gray;title(sprintf('Distance z %d', z(i)));colorbar;
    caxis([0, 10e7]);
    pause(0.1);
end
prctile(Ividmeas(:), 99.99)
prctile(Ividmeas(:), 99.9)
max(Ividmeas(:))
save(['optimization_' tag '.mat'], 'source', 'phase');
%save(['source_phase_result_' tag '.mat'], 'source1', 'phase1', 'source2', 'phase2', 'hologram');

%save(['simultaneous_result_' tag '.mat'],  'source2', 'phase2', 'hologram');



