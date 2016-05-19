function [loss, df ] = SourceFunObj( phase_source, z, nfocus, lambda, ps, focal_SLM, Nx, Ny, thresholdh, thresholdl, maskFun,  useGPU, ratio1, ratio2)
%FUNOBJ Summary of this function goes here
%   Detailed explanation goes here

phase = reshape(phase_source(1:Nx*Ny), [Nx, Ny]);
source = reshape(phase_source(Nx*Ny+1:end), [Nx, Ny]);
objectField = exp(1i * phase);

%objectField = phase;
loss = 0;


if useGPU
dfsource = zeros(Nx, Ny, 'gpuArray');
dfphase = zeros(Nx, Ny, 'gpuArray');
objectField = gpuArray(objectField);
else
dfsource = zeros(Nx, Ny);
dfphase = zeros(Nx, Ny);
end


for i = 1 : numel(z)
    mask = maskFun(z(i));   
    HStack = GenerateFresnelPropagationStack(Nx, Ny, z(i)-z(nfocus), lambda, ps, focal_SLM);
    fieldz = fftshift(fft2(objectField .* HStack));
    coherent_spectral = fft2(ifftshift(abs(fieldz.^2)));
    source_spectral = fft2(source);
    imagez = ifft2(source_spectral .* coherent_spectral);    
    

    maskh = mask .* (imagez < thresholdh);
    maskl = (1-mask) .* (imagez > thresholdl);
    
    

    
    diffh = maskh .* (imagez - thresholdh);
    diffl = maskl .* (imagez - thresholdl);
    
    
    temp_sourceh = 2 * ifft2(conj(coherent_spectral) .* fft2(diffh));
    temp_sourcel = 2 * ifft2(conj(coherent_spectral) .* fft2(diffl));

   
    temp_phaseh = fftshift(ifft2( conj(source_spectral).* fft2(diffh)));
    temp_phaseh = fieldz .* temp_phaseh;
    temp_phaseh = conj(HStack).*(Nx*Ny*ifft2(ifftshift(temp_phaseh)));
    temp_phasel = fftshift(ifft2( conj(source_spectral).* fft2(diffl)));
    temp_phasel = fieldz .* temp_phasel;
    temp_phasel = conj(HStack).*(Nx*Ny*ifft2(ifftshift(temp_phasel)));
%     templ = Nx*Ny *abs(HStack.^2).* (objectField.*diffl);
%     temph = Nx*Ny *abs(HStack.^2).* (objectField.*diffh);

    loss = loss + sum(sum(diffh.^2 + diffl.^2)); 
    
    dfphase = dfphase + temp_phaseh + temp_phasel;
    dfsource = dfsource + temp_sourceh + temp_sourcel;
    %clear HStack mask imagez imageInten maskh maskl diffh diffl temph templ
end
%df = df .* (1i * intensity * exp(1i*phase));

df = zeros(2*Nx*Ny, 1);
loss = gather(loss);
dfphase = -real(dfphase) .* sin(phase) + imag(dfphase).*cos(phase);
df(1:Nx*Ny) = gather(dfphase(:)) * ratio1;
%dfsource = ((dfsource + source) > 0) .* dfsource ;
df(Nx*Ny+1:end) = gather(dfsource(:)* ratio2);

end

