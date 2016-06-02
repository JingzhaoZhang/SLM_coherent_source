function [ mask ] = generateComplexMask( zi, Nx, Ny, complexMask1, zrange1, complexMask2, zrange2)
% GENERATEMASK targets is a list of triple tuple [x,y,z; ...]. (0,0, z')
% is at the center. z' and vector z is within the same scale. All have
% unit meter.
% z is a sequence of z distances.
% ps is the pixel size.
% Returns the i th mask. i <= numel(z). The ball around targets has value
% 1.




if zi >= zrange1(1) && zi <= zrange1(2)
    mask = complexMask1;
elseif zi >= zrange2(1) && zi <= zrange2(2)
    mask = complexMask2;
else
    mask = zeros(Nx, Ny);
end



end

