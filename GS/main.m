clear all;
tag = '804_800by600_A';


%% Setup params
% All length value has unit meter in this file.
% The 3d region is behind lens after SLM. 

addpath(genpath('minFunc_2012'))

resolutionScale = 1; % The demagnification scale of tubelens and objective. f_tube/f_objective
lambda = 1e-6;  % Wavelength
focal_SLM = 0.2; % focal length of the lens after slm.
psSLM = 20e-6;      % Pixel Size (resolution) at the scattered 3D region
Nx = 800;       % Number of pixels in X direction
Ny = 600;       % Number of pixels in Y direction


psXHolograph = lambda * focal_SLM/ psSLM / resolutionScale / Nx;      % Pixel Size (resolution) at the scattered 3D region
psYHolograph = lambda * focal_SLM/ psSLM / resolutionScale / Ny;      % Pixel Size (resolution) at the scattered 3D region

useGPU = 0;     % Use GPU to accelerate computation. Effective when Nx, Ny is large (e.g. 600*800).


z = [-100 : 4: 100] * 3e-4 ;   % Depth level requested in 3D region.
nfocus = 25;                % z(nfocus) denotes the depth of the focal plane.
thresholdh = 20000000;          % Intensity required to activate neuron.
thresholdl = 0;             % Intensity required to not to activate neuron.

%% Point Targets
% radius = 0.5e-4; % Radius around the point.
% step = Nx/16 * psXHolograph;
% epsilon = 0;
% targets = [ step, step, z(10); step, 0, z(10); step, -step, z(10); 0, step, z(10);...
%     0, 0, z(10); 0, -step, z(10);step+epsilon, step, z(40); step+epsilon, 0, z(40); step+epsilon, -step, z(40); 0+epsilon, step, z(40);...
%     0+epsilon, 0, z(40); 0+epsilon, -step, z(40);-step-epsilon, step, z(25); -step-epsilon, 0, z(25); -step-epsilon, -step, z(25); 0-epsilon, step, z(25);...
%     0-epsilon, 0, z(25); 0-epsilon, -step, z(25);] ; % Points where we want the intensity to be high.


% radius = 3e-4;
% targets = [ 0, 0, z(5); 0, 0, z(45);];





%% Complex Target
load('largeAB');
ims(:,:,1) = maskA' * 1;
imdepths = z(25);

% ims(:,:,1) = maskA' * 1;
% ims(:,:,2) = maskB' * 1;
% imdepths = [z(5), z(45)];

 
% for i = 1:12
% ims(:,:,i) = maskA' * 1;
% end
% imdepths = z(22:33);

%maskfun = @(zi) generateComplexMask( zi, Nx, Ny, maskA, zrange1, maskB, zrange2);
% 


%% Optimization
HStacks = zeros(Nx, Ny, numel(z));
if useGPU
    HStacks = gpuArray(HStacks);
end
for i = 1 : numel(z)
    HStacks(:,:,i) = GenerateFresnelPropagationStack(Nx, Ny, z(i)-z(nfocus), lambda, psXHolograph,psYHolograph, useGPU);
end
kernelfun = @(x) HStacks(:,:,x);
% kernelfun = @(i) GenerateFresnelPropagationStack(Nx, Ny, z(i)-z(nfocus), lambda,psXHolograph,psYHolograph, focal_SLM, useGPU);

%%
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
total_energy = sum(sum(source))
tag = [tag, '_gaussian'];

figure();imagesc(source);title('Gaussian source');drawnow;




%%


%load('gs_simulation.mat')
%load('../data/feasible_points.mat')
%[Ividmeas, phase]= getFeasiblePointTargets( source, targets, radius, z, nfocus, resolutionScale, lambda, focal_SLM, psSLM, Nx, Ny );
[Ividmeas, phase] = getFeasibleComplexTargets( ims, imdepths, z, nfocus, resolutionScale, lambda, focal_SLM, psSLM, Nx, Ny );

% 
% 

% %source = 10000*source1;
% im = source;
% maxiter = 30;
% figure();
% for n = 1:maxiter
%     for i = 1:numel(z)
%         HStack = kernelfun(i);
%         imagez = fftshift(fft2(im .* HStack));
%         target = Ividmeas(:,:,i) ;
%         %target = maskfun(z(i)) * 10;
%         imagez = sqrt(target) .* exp(1i * angle(imagez));
%         im =  ifft2(ifftshift(imagez))./HStack;
%         im = source.*exp(1i * angle(im));
%     end
%     %im = source.*exp(1i * angle(im));
%     if mod(n, 10)==0
%         display(n)
%         imagesc(angle(im));drawnow;pause(0.1);
%     end
% end
% 
% 
% source1 = source;
% phase = gather(angle(im));

%% Jingshan's GS

% intensity = 1/Nx/Ny;
% source = sqrt(intensity) * ones(Nx, Ny);
% %source = 10000*source1;
% im = source;
% maxiter = 200;
% figure();
% for n = 1:maxiter
%     for i = 1:numel(imdepths)
%         HStack = GenerateFresnelPropagationStack(Nx, Ny, imdepths(i)-z(nfocus), lambda, psXHolograph,psYHolograph, useGPU);        
%         imagez = fftshift(fft2(im .* HStack));
%         target = double(maskfun(imdepths(i)));
%         %target = maskfun(z(i)) * 10;
%         imagez = sqrt(target) .* exp(1i * angle(imagez));
%         im =  ifft2(ifftshift(imagez))./HStack;
%         im = source.*exp(1i * angle(im));
%     end
%     %im = source.*exp(1i * angle(im));
%     if mod(n, 50)==0
%         display(n)
%         imagesc(angle(im));drawnow;pause(0.1);
%     end
% end

% phase_jingshan = gather(angle(im));

%% plot
Ividmeas = zeros(Nx, Ny, numel(z));
usenoGPU = 0;
figure();
phase = mod(phase, 2*pi) - pi;
for i = 1:numel(z)
    HStack = GenerateFresnelPropagationStack(Nx,Ny,z(i) - z(nfocus), lambda, psXHolograph,psYHolograph, usenoGPU);
    imagez = fresnelProp(phase, source, HStack);
    Ividmeas(:,:,i) = imagez;
    imagesc(imagez);colormap gray;title(sprintf('Distance z %d', z(i)));
    colorbar;
    maxInten = sum(sum(source.^2))*500;
    %maxInten = max(max(max(Ividmeas)));
    caxis([0,maxInten])
    %caxis([0, 50000000]);
%     filename = sprintf('pointTarget%d.png', i);
%     print(['data/' filename], '-dpng')
    pause(0.1);
end
prctile(Ividmeas(:), 99.99)
prctile(Ividmeas(:), 99.9)
max(Ividmeas(:))
%save('gs_simulation.mat', 'Ividmeas', 'source1', 'phase')
%save(['source_phase_result_' tag '.mat'], 'source1', 'phase', 'source2', 'phase2', 'hologram');
phase = mod(phase, 2*pi) - pi;
save(['gs_cloudpoint_' tag '.mat'], 'phase', 'source');

