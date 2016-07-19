%clear all;
tag = 'slmFocal_20_600by800_coaxisPoint';


%% Setup params
% All length value has unit meter in this file.
% The 3d region is behind lens after SLM. 


resolutionScale = 20; % The demagnification scale of tubelens and objective. f_tube/f_objective
lambda = 1e-6;  % Wavelength
focal_SLM = 0.2; % focal length of the lens after slm.
psSLM = 20e-6;      % Pixel Size (resolution) at the scattered 3D region
Nx = 600;       % Number of pixels in X direction
Ny = 800;       % Number of pixels in Y direction


psXHolograph = lambda * focal_SLM/ psSLM / resolutionScale / Nx;      % Pixel Size (resolution) at the scattered 3D region
psYHolograph = lambda * focal_SLM/ psSLM / resolutionScale / Ny;      % Pixel Size (resolution) at the scattered 3D region

useGPU = 1;     % Use GPU to accelerate computation. Effective when Nx, Ny is large (e.g. 600*800).


z = [-100 :4: 100] * 1e-6 ;   % Depth level requested in 3D region.
nfocus = 20;                % z(nfocus) denotes the depth of the focal plane.
thresholdh = 1e-2;          % Intensity required to activate neuron.
thresholdl = 0;             % Intensity required to not to activate neuron.

%% Point Targets
radius = 9.9 * 1e-6 ; % Radius around the point.
targets = [ 0, 0, z(10) * 1e6; -0,-0, z(40) * 1e6;] * 1e-6 ; % Points where we want the intensity to be high.
%targets = [ 0, 0, z(25) * 1e6;] * 1e-6 ;
Masks = zeros(Nx, Ny, numel(z));
if useGPU
    Masks = gpuArray(Masks);
end
for i = 1 : numel(z)
    Masks(:,:,i) = generatePointMask( targets, radius, z(i), Nx, Ny, psXHolograph,psYHolograph, useGPU);
end
maskfun = @(i1, i2) Masks(:,:,i1:i2);
%maskfun = @(zi)  generatePointMask( targets, radius, zi, Nx, Ny, psXHolograph,psYHolograph, useGPU);


%% Complex Target
% load('largeAB');
% zrange1 = [z(25) - 5e-6, z(25) + 5e-6];
% zrange2 = [550,580];
% % % maskfun = @(zi) generateComplexMask( zi, Nx, Ny, maskA, zrange1, maskB, zrange2);
% Masks = zeros(Nx, Ny, numel(z));
% if useGPU
%     Masks = gpuArray(Masks);
% end
% for i = 1 : numel(z)
%     Masks(:,:,i) = generateComplexMask( z(i), Nx, Ny, maskA, zrange1, maskB, zrange2);
% end
% maskfun = @(i1, i2) Masks(:,:,i1:i2);
% 

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
std = Nx/4;
mu = [0 0];
Sigma = [std.^2 0; 0 std.^2];
x1 = [1:Nx] - Nx/2; x2 =  [1:Ny] - Ny/2;
[X1, X2] = meshgrid(x2,x1);
F = mvnpdf([X1(:) X2(:)],mu,Sigma);
source = reshape(F,length(x1),length(x2));
figure();imagesc(source);title('Gaussian source');drawnow;

tic;


%% Optimization phase 1
% Scale the gradient of phase by 1 and the gradient of source by 0.
% This makes sure that only the phase is updated in each iteration.


f = @(x)SourceFunObj( x, source, z, Nx, Ny, thresholdh, thresholdl, maskfun, kernelfun, useGPU);


matlab_options = optimoptions('fmincon','GradObj','on', 'display', 'iter', ...
    'algorithm','interior-point','Hessian','lbfgs', 'MaxFunEvals', 150, 'MaxIter', 150,...
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
    imagesc(imagez);colormap gray;title(sprintf('Distance z %d', z(i)));
    colorbar;
    caxis([0, 5e-3]);
    pause(0.1);
end
save(['coherentsource_result_' tag '.mat'], 'source', 'phase');
%save(['source_phase_result_' tag '.mat'], 'source1', 'phase1', 'source2', 'phase2', 'hologram');

%save(['simultaneous_result_' tag '.mat'],  'source2', 'phase2', 'hologram');



