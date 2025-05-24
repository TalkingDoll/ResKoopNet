
fid = fopen('dynamic_filenames.txt');
% Read the filenames into a cell array
filenames = textscan(fid, '%s');
filenames = filenames{1};

input_path = '...\';

for mouse_idx = 1:size(filenames, 1) 
tier_file_path = fullfile(input_path,  filenames{mouse_idx}, 'responses_data_oracle.mat');
loaded_data = load(tier_file_path,'response_data');
response_data = loaded_data.response_data;
[N_dim, len, N_trial] = size(response_data);

N_feature = 50;
features_hankel = nan(N_feature^2, N_trial);
features_hankel_mat = [];
for n_trial = 1:N_trial
    display(strcat('Start calculations for mouse:', int2str(mouse_idx), ', trial:', int2str(n_trial)));
    temp_mat = squeeze(response_data(:, :, n_trial));
    [HModes, HEvalues, Norms] = Hankel_DMD_Kaidi(temp_mat, len-N_feature, N_feature);
    features_hankel(:,n_trial) = HModes(:);
    features_hankel_mat = [features_hankel_mat, HModes'];
end

HankelOutput.N_feature = N_feature;
HankelOutput.features_hankel = features_hankel;
HankelOutput.features_hankel_mat = features_hankel_mat;

save(fullfile(input_path, filenames{mouse_idx}, 'Hankel_output.mat'),'HankelOutput');
end

