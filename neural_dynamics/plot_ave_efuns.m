all_resDMD_efuns = nan(299,6, 501, 10);
trial_counter = ones(1,6);

for i = 1:size(temp,1)
    i_type = Params.clusteredTrials_sorted(2, i);
    all_resDMD_efuns(:, i_type, :, trial_counter(i_type)) = squeeze(temp(i, :,:));
    trial_counter(i_type) = trial_counter(i_type)+1;
end

% valid_efun_idx = find(abs(EDMD_outputs.evalues)>0.1);
% all_resDMD_efuns = all_resDMD_efuns(:,:,valid_efun_idx,:);
% [~, sort_efun_idx] = sort(abs(EDMD_outputs.evalues(valid_efun_idx)), 'descend');

% ave_resDMD_efuns = squeeze(nanmean(all_resDMD_efuns(:,:,sort_efun_idx,:), 4));
ave_resDMD_efuns = squeeze(nanmean(all_resDMD_efuns(:,:,:,:), 4));
figure;
subplot(15,1,1)
imagesc(1:6); 
ax=gca();ax.FontSize = 15; ylabel('State'); yticks([]); 
ax.TickLength = [0 0];%ax.XTick = []; % Remove x-axis ticks;
ax.XAxisLocation = 'top'; hold on;
xline(1.5:1:6.5, 'k', 'LineWidth', 2)

subplot(15,1,[2:8])
imagesc(reshape(ave_resDMD_efuns, 6*299, length(valid_efun_idx))');
xline(299:299:1800, 'k', 'LineWidth', 2)
colormap(flipud(othercolor('Spectral8')));
ax=gca();ax.FontSize = 15; ylabel('Eigenfunctions');xticks([])
%xticks([150:299:1800-150]); xticklabels({'State 1', 'State 2', 'State 3', 'State 4', 'State 5', 'State 6'});%xticks(299:299:1800);
%title('resDMD')

%%
Hankel_feature_trial = nan(58, 50, 50);
for i = 1:size(temp,1)
    win = 50*(i-1)+1:50*i;
    Hankel_features_trial(i,:,:) = Hankel_output.features_hankel_mat(:, win);
end

all_Hankel_efuns = nan(50,6, 50, 10);
trial_counter = ones(1,6);

for i = 1:size(temp,1)
    i_type = Params.clusteredTrials_sorted(2, i);
    all_Hankel_efuns(:, i_type, :, trial_counter(i_type)) = Hankel_features_trial(i, :,:);
    trial_counter(i_type) = trial_counter(i_type)+1;
end
ave_Hankel_efuns = squeeze(nanmean(all_Hankel_efuns(:,:,:,:), 4));

subplot(15,1,[9:15])
imagesc(reshape(abs(ave_Hankel_efuns), 6*50, 50)');
xline(50:50:6*50, 'k', 'LineWidth', 2)
colormap(flipud(othercolor('Spectral8')));
ax=gca();ax.FontSize = 15; ylabel('Eigenfunctions');
%xticks(50:50:6*50);
xticks([150:299:1800-150]/6); xticklabels({'State 1', 'State 2', 'State 3', 'State 4', 'State 5', 'State 6'});%xticks(299:299:1800);
%title('Hankel DMD')

set(gcf, 'Position', [100, 100, 1800, 600])
% tightfig;
