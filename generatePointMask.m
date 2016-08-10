function [ mask ] = generatePointMask( targets, radius, zi, Nx, Ny, psx, psy, zratio, useGPU)
% GENERATEMASK targets is a list of triple tuple [x,y,z; ...]. (0,0, z')
% is at the center. z' and vector z is within the same scale. All have
% unit meter.
% z is a sequence of z distances.
% ps is the pixel size.
% Returns the i th mask. i <= numel(z). The ball around targets has value
% 1.

[n, ~] = size(targets);
cx=[1:Nx] - (floor(Nx/2)+1);
cy=[1:Ny] - (floor(Ny/2)+1);

if useGPU
    cx = gpuArray(cx); cy = gpuArray(cy);
    mask = gpuArray.zeros(Nx, Ny);
else
    mask = zeros(Nx, Ny);
end
[us, vs]=ndgrid(cx, cy);
us = us * psx; vs = vs * psy;

for k = 1:n
    coords = targets(k , :);
    dz2 = zratio * (zi - coords(3))^2;
    dx2 = (us - coords(1)).^2;
    dy2 = (vs - coords(2)).^2;

        mask = mask + ((dx2 + dy2 + dz2) <= radius^2);

end

mask = mask > 0;

end

