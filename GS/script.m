%%
filename = 'coherentsource_result_slmFocal_720_600by800_A_stack';
path = ['data/' filename '.mat'];
load(path)

[Nx, Ny, n] = size(Ividmeas);

%% Pick x in focal plane
figure();
for i = 1:n
imagesc(Ividmeas(:,:,i));
pause(0.1);
end

%%
x = 466;
figure();
imagesc(squeeze(Ividmeas(:,x,:)));caxis([0,1]);

%%
Ividmeas1 = Ividmeas;

%%
Ividmeas2 = Ividmeas;

%%
x1 = 92; 
y1 = 462; 
x2 = 104;
y2 = 457;
v1 = squeeze(Ividmeas1(x1, y1, :));
v1 = v1 - v1(end);
v1(v1 < 0) = 0;
v1 = v1/max(v1);
v2 = squeeze(Ividmeas2(x1, y1,:));
v2 = v2 - v2(end);
v2(v2 < 0) = 0;
v2 = v2/max(v2);
figure();plot(1:numel(z), v1, 1:numel(z), v2);legend('opt', 'GS');
