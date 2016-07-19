function [loss, df ] = SourceFunObj( phase, source, z, Nx, Ny, thresholdh, thresholdl, maskFun, fresnelKernelFun, useGPU)
%FUNOBJ Summary of this function goes here
%   Detailed explanation goes here




if useGPU
df = zeros(Nx, Ny, 'gpuArray');
phase = gpuArray(phase);
source = gpuArray(source);
else
df = zeros(Nx, Ny);
end

loss = 0;
phase = reshape(phase, [Nx, Ny]);
objectField = source.*exp(1i * phase);


for i = 1 : numel(z)
    
    HStack = fresnelKernelFun(i,i);
    mask = maskFun(i,i);    
    imagez = fftshift(fft2(objectField .* HStack));
    imageInten = abs(imagez.^2);
    maskh = mask .* (imageInten < thresholdh);
    maskl = (1-mask) .* (imageInten > thresholdl);
    
    diffh = maskh .* (imageInten - thresholdh);
    diffl = maskl .* (imageInten - thresholdl);
    
    temph = imagez.*diffh;
    temph = conj(HStack).*(Nx*Ny*ifft2(ifftshift(temph)));
    templ = imagez.*diffl;
    templ = conj(HStack).*(Nx*Ny*ifft2(ifftshift(templ)));
%     templ = Nx*Ny *abs(HStack.^2).* (objectField.*diffl);
%     temph = Nx*Ny *abs(HStack.^2).* (objectField.*diffh);

    loss = loss + sum(sum(diffh.^2 + diffl.^2)); 
    
    
    df = df +  temph + templ;
    %clear HStack mask imagez imageInten maskh maskl diffh diffl temph templ
end
%df = df .* (1i * intensity * exp(1i*phase));
dfphase = source.*(- real(df).*sin(phase) + imag(df) .* cos(phase));

%df = df .* (1i * intensity * exp(1i*phase));

df = gather(real(dfphase(:)));
% loss = real(loss);
% df(1:Nx*Ny) = real(dfphase(:)) * ratio1;
% df(Nx*Ny+1:end) = real(dfsource(:))* ratio2;
loss = gather(real(loss));
end

