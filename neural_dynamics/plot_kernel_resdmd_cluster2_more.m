% Parameters (set these according to your data)
nfreq = 25; % Number of frequencies


% Read filenames from the text file
fid = fopen('dynamic_filenames.txt');
filenames = textscan(fid, '%s');
filenames = filenames{1};
fclose(fid);

dim_reduct_method = 'MDS';

cmap = flipud(othercolor('Spectral8'));
% Number of colors you want to extract
numColors = 6;
% Calculate indices to select colors
indices = round(linspace(1, length(cmap), numColors));
% Extract the colors
selectedColors = cmap(indices, :);

% Loop through each mouse's data
for mouse_idx = 1:5 % Modify as needed to include other indices


    %% load data
    temp = load(fullfile(input_path,  filenames{mouse_idx}, 'oracle_trial_clustered.mat'));
    Params = temp.Output;
    
    Ntrial = Params.numOracleTrials;
    % load(fullfile(input_path, filenames{mouse_idx}, 'responses_data_oracle.mat'), 'response_data');
    % [Nch, Ntime, Ntrial] = size(response_data);
    
    resDMD_feature = AveSpectrog{1, mouse_idx};
     %% plot temporal features
     if mouse_idx == 1
    
    resDMD_feature_reshaped = abs(efuns_mouse_trials1);

    all_resDMD_efuns = nan(299,299, 6, 10);
    trial_counter = ones(1,6);  

    for i = 1:Ntrial
        i_type = Params.clusteredTrials_sorted(2, i);
        all_resDMD_efuns(:,  :, i_type,trial_counter(i_type)) = squeeze(resDMD_feature_reshaped(:,:,i));
        trial_counter(i_type) = trial_counter(i_type)+1;
    end

    ave_resDMD_efuns = squeeze(nanmean(all_resDMD_efuns(:,:,:,:), 4));
    figure(1);
    subplot(8,1,1)
    imagesc(1:6); 
    ax=gca();ax.FontSize = 15; ylabel('State'); yticks([]); 
    ax.TickLength = [0 0];%ax.XTick = []; % Remove x-axis ticks;
    ax.XAxisLocation = 'top'; hold on;
    xline(1.5:1:6.5, 'k', 'LineWidth', 2)
    
    subplot(8,1,[2:8])
    imagesc(reshape(ave_resDMD_efuns, 299, 6*299));
    xline(300-1:300-1:1800, 'k', 'LineWidth', 2)
    colormap(flipud(othercolor('Spectral8')));
    ax=gca();ax.FontSize = 15; ylabel('Frequency (Hz)');xticks([])
     end

    Y_resDMD = AveSpectrog{2, mouse_idx};
    
    figure(20)
    subplot(3,5,mouse_idx)
    cluster_idx = unique(Params.clusteredTrials(2, :));
    for i = 1:length(cluster_idx)
    idx = find(Params.clusteredTrials_sorted(2,:)==cluster_idx(i)); hold on;
    plot(Y_resDMD(idx,1), Y_resDMD(idx,2), '*', 'Color', selectedColors(i, :), 'LineWidth', 2); %view([-125,28])
    end
    ax = gca; ax.FontSize = 15;
    xlabel('Dim. 1');
    ylabel('Dim. 2');
    zlabel('Dim. 3');
    title(strcat('Mouse-', int2str(mouse_idx), ', Kernel-ResDMD (MDS)'));
    % title(strcat('resDMD'));

    dbIndex.resDMD(mouse_idx, 1) = daviesBouldin(Y_resDMD, Params.clusteredTrials_sorted(2,:));


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% UMAP %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    [Y_resDMD] = run_umap(resDMD_feature','metric','correlation','n_components', 2);
    AveSpectrog{3, mouse_idx} = Y_resDMD;

        figure(20)
    subplot(3,5,mouse_idx+5)
    cluster_idx = unique(Params.clusteredTrials(2, :));
    for i = 1:length(cluster_idx)
    idx = find(Params.clusteredTrials_sorted(2,:)==cluster_idx(i)); hold on;
    plot(Y_resDMD(idx,1), Y_resDMD(idx,2), '*', 'Color', selectedColors(i, :), 'LineWidth', 2); %view([-125,28])
    end
    ax = gca; ax.FontSize = 15;
    xlabel('Dim. 1');
    ylabel('Dim. 2');
    zlabel('Dim. 3');
    title(strcat('Mouse-', int2str(mouse_idx), ', Kernel-ResDMD (UMAP)'));
    % title(strcat('resDMD'));

    dbIndex.resDMD(mouse_idx, 2) = daviesBouldin(Y_resDMD, Params.clusteredTrials_sorted(2,:));


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% t_SNE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    [Y_resDMD] = tsne(resDMD_feature','NumDimensions',2,'Distance','correlation','Perplexity',15);
        AveSpectrog{4, mouse_idx} = Y_resDMD;

            figure(20)
    subplot(3,5,mouse_idx+10)
    cluster_idx = unique(Params.clusteredTrials(2, :));
    for i = 1:length(cluster_idx)
    idx = find(Params.clusteredTrials_sorted(2,:)==cluster_idx(i)); hold on;
    plot(Y_resDMD(idx,1), Y_resDMD(idx,2), '*', 'Color', selectedColors(i, :), 'LineWidth', 2); %view([-125,28])
    end
    ax = gca; ax.FontSize = 15;
    xlabel('Dim. 1');
    ylabel('Dim. 2');
    zlabel('Dim. 3');
    title(strcat('Mouse-', int2str(mouse_idx), ', Kernel-ResDMD (t-SNE)'));
    % title(strcat('resDMD'));

    dbIndex.resDMD(mouse_idx, 3) = daviesBouldin(Y_resDMD, Params.clusteredTrials_sorted(2,:));

    % AveSpectrog{1, mouse_idx} = resDMD_feature;
    % AveSpectrog{2, mouse_idx} = Y_resDMD;
end

% At this point, combined_data is a (nfreq * nch * ntime) x ntrial matrix
