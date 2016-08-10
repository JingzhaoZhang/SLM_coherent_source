function [ images, phase ] = getFeasibleComplexTargets( ims, imdepths, z, nfocus, resolutionScale, lambda, focal_SLM, psSLM, Nx, Ny )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
psXHolograph = lambda * focal_SLM/ psSLM / resolutionScale / Nx;      % Pixel Size (resolution) at the scattered 3D region
psYHolograph = lambda * focal_SLM/ psSLM / resolutionScale / Ny;      % Pixel Size (resolution) at the scattered 3D region
images = zeros(Nx, Ny, numel(z));


maxiter = 20;
source = ones(Nx,Ny)/Nx/Ny;
usenoGPU = 0;
phase = zeros(Nx, Ny);

for i_target = 1:numel(imdepths)
    display(sprintf('Processing target %d', i_target))
    target = ims(:,:,i_target);
    zi = imdepths(i_target);

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

