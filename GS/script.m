%%
Ividmeas1 = Ividmeas;

figure()
for i = 1:numel(z)
    imagesc(Ividmeas(:,:,i));colormap gray;colorbar;
    caxis([0, 5e-4]);drawnow;pause(0.1);
end
v = squeeze(Ividmeas(232, 400, :)/max(Ividmeas(232, 400,:)));
figure();plot(1:numel(z), v);


%%
figure();imagesc(Ividmeas1(:,:,25));colormap gray;colorbar;
figure();imagesc(Ividmeas2(:,:,25));colormap gray;colorbar;

%%
figure();imagesc(phase1);colorbar;