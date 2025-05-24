

fid = fopen('dynamic_filenames.txt');
filenames = textscan(fid, '%s');
filenames = filenames{1};

input_path = '...';

dim_reduct_method = 'UMAP';

figure(1);
set(gcf, 'Position', [0,0,1800*2 1000])

cmap = flipud(othercolor('Spectral8'));
% Number of colors you want to extract
numColors = 6;
% Calculate indices to select colors
indices = round(linspace(1, length(cmap), numColors));
% Extract the colors
selectedColors = cmap(indices, :);


for mouse_idx = 1:size(filenames, 1) 

temp = load(fullfile(input_path,  filenames{mouse_idx}, 'Hankel_output.mat'));
Hankel_output = temp.HankelOutput;
load(strcat('...\sensorium_data',int2str(mouse_idx),'_Ndict_601_outputs.mat'))
load(fullfile(input_path,  filenames{mouse_idx}, 'oracle_trial_clustered.mat'))
Params = Output;

valid_efun_idx = find(abs(EDMD_outputs.evalues)>0.1);
[~, sort_efun_idx] = sort(abs(EDMD_outputs.evalues(valid_efun_idx)), 'descend');

EDMD_outputs.efuns_norm = normalize_efun(EDMD_outputs.efuns(:,sort_efun_idx));
temp = reshape(EDMD_outputs.efuns_norm, Params.numOracleTrials, Params.valid_len, size(EDMD_outputs.efuns_norm,2));

resDMD_feature = reshape(temp, Params.numOracleTrials, Params.valid_len*size(EDMD_outputs.efuns_norm,2))';
hankel_features = abs(Hankel_output.features_hankel);



switch dim_reduct_method
%%% MDS
    case 'MDS'
resDMD_feature = reshape(temp, Params.numOracleTrials, Params.valid_len*size(EDMD_outputs.efuns_norm,2))';
D_resDMD = pdist(resDMD_feature','correlation');
Y_resDMD = mdscale(D_resDMD, 2);

hankel_features = abs(Hankel_output.features_hankel);
D_hankel = pdist(hankel_features', 'correlation');
Y_hankel = mdscale(D_hankel, 2);
    
    case 'UMAP'
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% UMAP %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[Y_resDMD] = run_umap(resDMD_feature','metric','correlation','n_components', 2);
[Y_hankel] = run_umap(hankel_features','metric','correlation','n_components', 2);

    case 'tSNE'
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% t_SNE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[Y_resDMD] = tsne(resDMD_feature','NumDimensions',2,'Distance','correlation','Perplexity',15);
[Y_hankel] = tsne(hankel_features','NumDimensions',2,'Distance','correlation','Perplexity',15);

end

figure(1)
subplot(2, size(filenames, 1)+1, mouse_idx);
cluster_idx = unique(Params.clusteredTrials(2, :));
for i = 1:length(cluster_idx)
    idx = find(Params.clusteredTrials_sorted(2,:)==cluster_idx(i)); hold on;
    plot(Y_resDMD(idx,1), Y_resDMD(idx,2), '*', 'Color', selectedColors(i, :), 'LineWidth', 2); %view([-125,28])
end
ax = gca; ax.FontSize = 15;
xlabel('Dim. 1');
ylabel('Dim. 2');
zlabel('Dim. 3');
title(strcat('Mouse-', int2str(mouse_idx), ', resDMD'));
% title(strcat('resDMD'));


subplot(2, size(filenames, 1)+1, mouse_idx+size(filenames, 1)+1);
cluster_idx = unique(Params.clusteredTrials(2, :));
for i = 1:length(cluster_idx)
    idx = find(Params.clusteredTrials_sorted(2,:)==cluster_idx(i)); hold on;
    plot(Y_hankel(idx,1), Y_hankel(idx,2), '*',  'Color', selectedColors(i, :), 'LineWidth', 2); %view([-125,28])
end
ax = gca; ax.FontSize = 15;
xlabel('Dim. 1');
ylabel('Dim. 2');
zlabel('Dim. 3');
title(strcat('Hankel DMD'));
% title(strcat('Mouse-', int2str(mouse_idx), ', Hankel DMD'));

dbIndex.resDMD(mouse_idx) = daviesBouldin(Y_resDMD, Params.clusteredTrials_sorted(2,:));
dbIndex.hankel(mouse_idx) = daviesBouldin(Y_hankel, Params.clusteredTrials_sorted(2,:));

end

% legend({'Video 1','Video 2', 'Video 3', 'Video 4', 'Video 5', 'Video 6'})
% legend boxoff

subplot(2,6,[6,12])
bar(1:5, [dbIndex.resDMD; dbIndex.hankel]); 
xlabel('Mouse index')
ylabel('Davies-Bouldin Index')
legend({'resDMD', 'Hankel DMD'});
ylim([0,23])
ax = gca; ax.FontSize = 15; box off

set(gcf, 'Position', [1,1,1800, 600])
tightfig
