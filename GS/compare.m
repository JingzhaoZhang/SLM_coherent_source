
%%
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

%%
source = ones(Nx, Ny)/sqrt(Nx*Ny);

load('gs_result_slmFocal_20_600by800_coaxisPoints.mat');
Ividmeas1 = zeros(Nx, Ny, numel(z));
usenoGPU = 0;
figure();
for i = 1:numel(z)
    HStack = GenerateFresnelPropagationStack(Nx,Ny,z(i) - z(nfocus), lambda, psXHolograph,psYHolograph, usenoGPU);
    imagez = fresnelProp(phase1, source, HStack);
    Ividmeas1(:,:,i) = imagez;
    imagesc(imagez);colormap gray;title(sprintf('Distance z %d', z(i)));
    colorbar;
    caxis([0, 150]);
%     filename = sprintf('pointTarget%d.png', i);
%     print(['data/' filename], '-dpng')
    %pause(0.1);
end


load('phaseonly_result_slmFocal_20_600by800_coaxisPoints.mat');

%load('phaseonly_result_slmFocal_20_600by800_3points.mat');
Ividmeas2 = zeros(Nx, Ny, numel(z));
usenoGPU = 0;
%figure();
for i = 1:numel(z)
    HStack = GenerateFresnelPropagationStack(Nx,Ny,z(i) - z(nfocus), lambda, psXHolograph,psYHolograph, usenoGPU);
    imagez = fresnelProp(phase1, source, HStack);
    Ividmeas2(:,:,i) = imagez;
    imagesc(imagez);colormap gray;title(sprintf('Distance z %d', z(i)));
    colorbar;
    caxis([0, 150]);
%     filename = sprintf('pointTarget%d.png', i);
%     print(['data/' filename], '-dpng')
    %pause(0.1);
end




%%
x1 = 300;
y1 = 400;
x2 = 285;
y2 = 400;


v1 = squeeze(Ividmeas1(x1, y1, :)/max(Ividmeas1(x1, y1,:)));
v2 = squeeze(Ividmeas2(x1, y1,:)/max(Ividmeas2(x1, y1,:)));
figure();plot(1:numel(z), v1, 1:numel(z), v2);

v1 = squeeze(Ividmeas1(213, y2,:)/max(Ividmeas1(x1, y1,:)));
v2 = squeeze(Ividmeas2(213, y2,:)/max(Ividmeas2(x1, y1,:)));
figure();plot(1:numel(z), v1, 1:numel(z), v2);
