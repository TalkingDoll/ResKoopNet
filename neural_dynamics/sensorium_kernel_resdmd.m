
% Read filenames from the text file
fid = fopen('dynamic_filenames.txt');
filenames = textscan(fid, '%s');
filenames = filenames{1};
fclose(fid);

% define data path as input_path

% Loop through each mouse's data
for mouse_idx = 1:size(filenames, 1) 
    load(fullfile(input_path,  filenames{mouse_idx}, 'responses_data_oracle.mat'))
    [Nch, Ntime, Ntrial] = size(response_data);
    
    % Loop through each channel
    for n_trial = 1:Ntrial
        n_trial
        response_data_trial = squeeze(response_data(:,:, n_trial)); % Initialize for one channel
        
        X = response_data_trial (:,1:end-1);
        Y = response_data_trial (:, 2:end);
        
        [PSI_x, PSI_y, PSI_y2] = kernel_ResDMD(X,Y,X,Y);
    
        M = size(X, 1);
        G = (PSI_x'*PSI_x)/M;
        A = (PSI_x'*PSI_y)/M;
        L = (PSI_y'*PSI_y)/M;

        %% calculate Koopman eigenfunctions
        K = pinv(G)*A; % finite approximation of Koopman operator
        [V, D, U] = eig(K); % eigendecomposition of K
        evalues_unsorted = diag(D);
        [~, sort_idx] = sort(diag(abs(D)),'descend'); % here use modulus
        evalues = evalues_unsorted(sort_idx);
        
        N_dict = size(D, 1);

        B = PSI_x'*X(:,1:size(PSI_x,1))'/M;
        W = pinv(G)*B; % weight of dictionary basis
        % X_estim_psi = psi*W;
        
        phi_unsorted = PSI_x*V; % eigenfunctions
        phi = PSI_x*V(:, sort_idx);
        
        scaling_factor = diag(real(U'*V)); % scale left eigenvectors
        evectorL = nan(size(U,1));
        for n = 1:N_dict
        evectorL(:, n) = U(:, n)/scaling_factor(n);
        end
        Kpm_modes = real(evectorL'*W); % Koopman mode

        %%
        EDMD_outputs.N_dict = N_dict;
        
        EDMD_outputs.evalues_unsorted = evalues_unsorted;
        EDMD_outputs.evalues = evalues;
        
        EDMD_outputs.efuns_unsorted = phi_unsorted;
        EDMD_outputs.efuns = phi;
        EDMD_outputs.V = V(:, sort_idx);
        
        EDMD_outputs.kpm_modes = Kpm_modes(sort_idx,:);
        
        % Save the result for each channel
        save_filename = fullfile(input_path, filenames{mouse_idx}, ['kernel_resdmd_trial_' num2str(n_trial) '.mat']);
        save(save_filename, 'EDMD_outputs', '-v7.3')
    end
end