folder = 'gif/0805/';
filename = 'optimization_804_600by800_compact18points_gaussian';
path = [folder filename '.tif'];
info_path = [folder filename '_info.mat'];
load(info_path)
z = Calibme.Zvector * 1e-6;
n = 121;
[Nx, Ny] = size(imread(path, 1));
Ividmeas = zeros(Nx, Ny, n);

f = figure();
for i = 1:n
    Ividmeas(:,:,i) = imread(path, i);
end

Ividmeas = Ividmeas/max(Ividmeas(:));

for i = 1:n
    imagesc(Ividmeas(:,:,i));colormap gray;
    caxis([0,1]);
    %colorbar;
    pause(0.02);
end
close(f);

save(['data/' filename '_stack'], 'Ividmeas', 'z')

Ividmeas = Ividmeas * 63;
map = colormap(gray);
gifname = [folder filename  '.gif'];
for i = 1:numel(z)
    
    if i == 1;
        imwrite(Ividmeas(:,:,i), map, gifname, 'LoopCount',Inf, 'DelayTime', 0.1);
    else
        imwrite(Ividmeas(:,:,i), map, gifname, 'WriteMode','append','DelayTime',0.1);
    end
end








