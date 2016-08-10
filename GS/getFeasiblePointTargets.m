function [ images, phase] = getFeasiblePointTargets(source, targets, radius, z, nfocus, resolutionScale, lambda, focal_SLM, psSLM, Nx, Ny )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
psXHolograph = lambda * focal_SLM/ psSLM / resolutionScale / Nx;      % Pixel Size (resolution) at the scattered 3D region
psYHolograph = lambda * focal_SLM/ psSLM / resolutionScale / Ny;      % Pixel Size (resolution) at the scattered 3D region
[n, ~] = size(targets);
images = zeros(Nx, Ny, numel(z));

cx=[1:Nx] - (floor(Nx/2)+1);
cy=[1:Ny] - (floor(Ny/2)+1);
[us, vs]=ndgrid(cx, cy);
us = us * psXHolograph; vs = vs * psYHolograph;

maxiter = 20;
usenoGPU = 0;
phase = zeros(Nx, Ny);


for i_target = 1:n
    display(sprintf('Processing target %d', i_target))
    coords = targets(i_target, :);
    zi = coords(3);
    dx2 = (us - coords(1)).^2;
    dy2 = (vs - coords(2)).^2;
    target = (dx2+dy2 <= radius^2) * 10;
    HStack = GenerateFresnelPropagationStack(Nx, Ny, zi-z(nfocus), lambda, psXHolograph,psYHolograph, usenoGPU);
    im = source;
    
    for i_iter = 1:maxiter
        imagez = fftshift(fft2(im .* HStack));
        %target = Ividmeas(:,:,i) ;
        imagez = sqrt(target) .* exp(1i * angle(imagez));
        im =  ifft2(ifftshift(imagez))./HStack;
        im = source.*exp(1i * angle(im));        
    end
    phase = phase + im;
    
    for i = 1:numel(z)
        HStack = GenerateFresnelPropagationStack(Nx,Ny,z(i) - z(nfocus), lambda, psXHolograph,psYHolograph, usenoGPU);
        imagez = fresnelProp(angle(im), source, HStack);        
        images(:,:,i) = images(:,:,i) + imagez;
    end
    
    
end
phase = angle(phase);
% figure()
% for i = 1:numel(z)
%     imagesc(images(:,:,i));colormap gray;colorbar;
%     caxis([0, 5e-4]);drawnow;pause(0.1);
% end

end

