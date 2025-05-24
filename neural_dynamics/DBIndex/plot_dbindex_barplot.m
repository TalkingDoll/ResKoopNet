%% MDS
load('dbIndex_mds.mat')

figure; bb = bar(dbIndex_mds','FaceColor','flat');
xlabel('Mouse Index'); ylabel('Davies-Bouldin Index');

cmap = flipud(othercolor('Spectral8'));
cmap_short = cmap(floor(linspace(1,256,4)),:);

for k = 1:size(dbIndex_mds',2)
    bb(k).CData = cmap_short(k,:);
end
title('Multi-dimensional Scaling')
legend('NN-ResDMD', 'Hankel DMD', 'EDMD+RBF', 'Kernel ResDMD')
box off
ax = gca; ax.FontSize = 15;

%% UMAP
load('dbIndex_umap.mat')

figure; bb = bar(dbIndex_umap','FaceColor','flat');
xlabel('Mouse Index'); ylabel('Davies-Bouldin Index');

cmap = flipud(othercolor('Spectral8'));
cmap_short = cmap(floor(linspace(1,256,4)),:);

for k = 1:size(dbIndex_umap',2)
    bb(k).CData = cmap_short(k,:);
end
title('UMAP')
legend('NN-ResDMD', 'Hankel DMD', 'EDMD+RBF', 'Kernel ResDMD');
box off;
ax = gca; ax.FontSize = 15;

%% t-SNE
load('dbIndex_tsne.mat')

figure; bb = bar(dbIndex_tsne','FaceColor','flat');
xlabel('Mouse Index'); ylabel('Davies-Bouldin Index');

cmap = flipud(othercolor('Spectral8'));
cmap_short = cmap(floor(linspace(1,256,4)),:);

for k = 1:size(dbIndex_tsne',2)
    bb(k).CData = cmap_short(k,:);
end
title('t-SNE')
legend('NN-ResDMD', 'Hankel DMD', 'EDMD+RBF', 'Kernel ResDMD')
box off
ax = gca; ax.FontSize = 15;