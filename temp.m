

%%
filename = 'optimization_804_600by800_compact18points_gaussian_stack';
load(['data/' filename '.mat']);

figure();
for i = 1:121
    imagesc(Ividmeas(:,:,i));colormap gray;caxis([0,1]);
end

figure()
slopeX = (471 - 533)/(89-34);
slopeY = (322 - 236)/(89-34);
Ividmeas1 = zeros(size(Ividmeas));
for i = 1:121
    i;
    Im = circshift(Ividmeas(:,:,i), [ -floor((i-34)*slopeY), -floor((i-34)*slopeX)]);
    imagesc(Im);colormap gray;caxis([0,1]);
    %pause(0.05);
    Ividmeas1(:,:,i) = Im;
end


%%
filename = 'gs_cloudpoint_804_800by600_compact18points_gaussian_stack';
load(['data/' filename '.mat']);

figure();
for i = 1:121
    imagesc(Ividmeas(:,:,i));colormap gray;
end

figure()
slopeX = (464-514)/(85-31);
slopeY = (322-253)/(85-31);
Ividmeas2 = zeros(size(Ividmeas));

for i = 1:121
    i;
    Im = circshift(Ividmeas(:,:,i), [ -floor((i-31)*slopeY), -floor((i-31)*slopeX)]);
    imagesc(Im);colormap gray;caxis([0,1]);
    pause(0.05);
    Ividmeas2(:,:,i) = Im;
end




%%
x1 = 362; 
y1 = 423; 
x2 = 257;
y2 = 397;


v1 = squeeze(Ividmeas1(x1, y1, :));
v1 = v1 - v1(end);
v1(v1 < 0) = 0;
v1 = v1/max(v1);
v2 = squeeze(Ividmeas2(x2, y2,:));
v2 = v2 - v2(end);
v2(v2 < 0) = 0;
v2 = v2/max(v2);
figure();plot(1:numel(z), v1, 1:numel(z), v2);legend('opt', 'GS');

