%% FIGURE 2: Classical Hopfield Network with Dynamic Thresholds
% Generates three-panel figure:
% Panel A: Recall quality vs noise for different γ values (fixed β)
% Panel B: Recall quality vs loading factor for different γ values (at different noise levels)
% Panel C: ΔQ heatmaps showing improvement over baseline

clear; clc; close all;

%% ========== PARAMETERS ==========
% Threshold parameters to test
gamma_values    = [-1.20, -1.00, -0.50, 0.00];   % Global energy scaling
bell_amp_values = [0.00, 10.0, 20.0];            % Heterogeneity amplitude

% Network parameters
network_size   = 144;            % 12x12 network
pattern_counts = [2, 4, 8, 12, 16, 20, 24, 28, 32];  % Different network loads
noise_levels   = 0:0.1:1.0;      % Noise levels to test
num_trials     = 10;            % Trials per condition

% Network topology
M        = 143;                  % Connectivity
SigmaM   = 3;                    % Spatial connection decay
weight_scale = 1.0;

% Dynamics parameters
max_iterations        = 1000;
convergence_threshold = 0.001;
temperature           = 0.1;
update_type           = 'async';

%% Initialize results storage
fprintf('=== FIGURE 2: CLASSICAL HOPFIELD CAPACITY ANALYSIS ===\n');
fprintf('Network size: %d neurons\n', network_size);
fprintf('Pattern counts: %s\n', mat2str(pattern_counts));
fprintf('Noise levels: 0.0 to 1.0 (step 0.1)\n');
fprintf('Trials per condition: %d\n', num_trials);

mean_results = zeros(length(noise_levels), length(pattern_counts), ...
                     length(gamma_values), length(bell_amp_values));
std_results  = zeros(length(noise_levels), length(pattern_counts), ...
                     length(gamma_values), length(bell_amp_values));

%% Run simulations
start_time = tic;
total_conditions = length(noise_levels) * length(pattern_counts) * ...
                   length(gamma_values) * length(bell_amp_values);
condition_count = 0;

for noise_idx = 1:length(noise_levels)
    noise_level = noise_levels(noise_idx);
    
    for p_idx = 1:length(pattern_counts)
        n_patterns = pattern_counts(p_idx);
        
        for gamma_idx = 1:length(gamma_values)
            gamma = gamma_values(gamma_idx);
            
            for bell_idx = 1:length(bell_amp_values)
                bell_amp = bell_amp_values(bell_idx);
                
                % Run trials
                trial_results = zeros(num_trials, 1);
                for trial = 1:num_trials
                    trial_seed = trial * 1000 + noise_idx * 100 + p_idx * 10 + ...
                                 gamma_idx * 5 + bell_idx;
                    
                    quality = hopfield_bell_threshold(network_size, noise_level, ...
                        false, M, SigmaM, n_patterns, max_iterations, ...
                        convergence_threshold, weight_scale, temperature, ...
                        update_type, gamma, bell_amp, trial_seed);
                    
                    trial_results(trial) = quality;
                end
                
                % Store statistics
                mean_results(noise_idx, p_idx, gamma_idx, bell_idx) = mean(trial_results);
                std_results(noise_idx, p_idx, gamma_idx, bell_idx) = std(trial_results);
                
                condition_count = condition_count + 1;
                if mod(condition_count, 50) == 0
                    elapsed = toc(start_time);
                    progress = 100 * condition_count / total_conditions;
                    fprintf('Progress: %.1f%% (%d/%d conditions), Elapsed: %.1f min\n', ...
                        progress, condition_count, total_conditions, elapsed/60);
                end
            end
        end
    end
end

fprintf('Completed in %.1f minutes\n', toc(start_time)/60);

%% ========== GENERATE FIGURE 2 ==========
loading_factors = pattern_counts / network_size;

% Find baseline (γ=0, β=0) for computing ΔQ
baseline_gamma_idx = find(gamma_values == 0);
baseline_bell_idx  = find(bell_amp_values == 0);

fig = figure('Position', [50, 50, 1800, 1400], 'Color', 'w');

%% PANEL A: Recall Quality vs Noise (specific γ,β combinations)
% Show three specific (γ, β) combinations across columns
% Based on example: (γ=0.0, β=0.0), (γ=-0.5, β=10.0), (γ=-1.0, β=20.0)
panel_A_configs = [
    4, 1;  % γ = 0.00, β = 0.0  (indices)
    3, 2;  % γ = -0.50, β = 10.0
    2, 3   % γ = -1.00, β = 20.0
];

for col = 1:3
    gamma_idx = panel_A_configs(col, 1);
    bell_idx  = panel_A_configs(col, 2);
    gamma_val = gamma_values(gamma_idx);
    bell_val  = bell_amp_values(bell_idx);
    
    subplot(3, 3, col);
    hold on; grid on; box on;
    
    % Plot all pattern counts with different colors
    colors = lines(length(pattern_counts));
    for p_idx = 1:length(pattern_counts)
        y_mean = mean_results(:, p_idx, gamma_idx, bell_idx);
        y_std  = std_results(:, p_idx, gamma_idx, bell_idx);
        
        errorbar(noise_levels, y_mean, y_std, '-o', ...
                 'LineWidth', 2, 'MarkerSize', 6, ...
                 'Color', colors(p_idx,:), ...
                 'DisplayName', sprintf('n = %d patterns', pattern_counts(p_idx)));
    end
    
    xlabel('Noise Level');
    ylabel('Recall Quality');
    title(sprintf('Recall Quality vs Noise (\\gamma = %.2f, \\beta = %.1f)', ...
                  gamma_val, bell_val));
    legend('Location', 'best', 'FontSize', 7);
    ylim([0, 1.05]);
    xlim([0, 1]);
    
    % Add reference line at 0.5
    plot([0, 1], [0.5, 0.5], 'k--', 'LineWidth', 1, 'HandleVisibility', 'off');
end

%% PANEL B: Recall Quality vs Loading Factor (at different noise levels)
% Show three different noise levels across columns
panel_B_noise_indices = [1, 3, 5];  % noise = 0.00, 0.20, 0.40
panel_B_bell_idx      = 1;          % Fixed β = 0.0

for col = 1:3
    noise_idx = panel_B_noise_indices(col);
    noise_val = noise_levels(noise_idx);
    
    subplot(3, 3, 3 + col);
    hold on; grid on; box on;
    
    % Plot each γ value
    colors = lines(length(gamma_values));
    for gamma_idx = 1:length(gamma_values)
        y_mean = squeeze(mean_results(noise_idx, :, gamma_idx, panel_B_bell_idx));
        y_std  = squeeze(std_results(noise_idx, :, gamma_idx, panel_B_bell_idx));
        
        errorbar(loading_factors, y_mean, y_std, '-o', ...
                 'LineWidth', 2, 'MarkerSize', 6, ...
                 'Color', colors(gamma_idx,:), ...
                 'DisplayName', sprintf('\\gamma = %.2f', gamma_values(gamma_idx)));
    end
    
    xlabel('Loading Factor \alpha = n/N');
    ylabel('Recall Quality');
    title(sprintf('Noise = %.2f', noise_val));
    legend('Location', 'best', 'FontSize', 8);
    ylim([0, 1.05]);
    
    % Add reference line at 0.5
    plot([min(loading_factors), max(loading_factors)], [0.5, 0.5], ...
         'k--', 'LineWidth', 1, 'HandleVisibility', 'off');
end

%% PANEL C: ΔQ Heatmaps (improvement over baseline)
% Show three different parameter combinations
% ΔQ_any = Q_best(γ,β) - Q_baseline(γ=0,β=0)
% ΔQ_γ = Q_best(γ,β=0) - Q_baseline  
% ΔQ_β = Q_best(γ,β≠0) - Q_baseline

% For each heatmap, find best γ or β and compute improvement
panel_C_configs = {
    'any',  'Q_{best}(\\gamma,\\beta) - Q_{baseline}';
    'gamma', 'Q_{best}(\\gamma,\\beta=0) - Q_{baseline}';
    'beta',  'Q_{best}(\\gamma,\\beta\\neq0) - Q_{baseline}'
};

for col = 1:3
    config_type = panel_C_configs{col, 1};
    config_label = panel_C_configs{col, 2};
    
    % Compute ΔQ for this configuration
    DeltaQ = zeros(length(noise_levels), length(pattern_counts));
    
    for noise_idx = 1:length(noise_levels)
        for p_idx = 1:length(pattern_counts)
            % Baseline quality
            Q_baseline = mean_results(noise_idx, p_idx, baseline_gamma_idx, baseline_bell_idx);
            
            % Best quality based on configuration
            switch config_type
                case 'any'
                    % Best over all γ and β
                    Q_best = max(max(mean_results(noise_idx, p_idx, :, :)));
                case 'gamma'
                    % Best over γ with β=0
                    Q_best = max(mean_results(noise_idx, p_idx, :, baseline_bell_idx));
                case 'beta'
                    % Best over β≠0 with any γ
                    non_zero_bell = setdiff(1:length(bell_amp_values), baseline_bell_idx);
                    Q_best = max(max(mean_results(noise_idx, p_idx, :, non_zero_bell)));
            end
            
            DeltaQ(noise_idx, p_idx) = Q_best - Q_baseline;
        end
    end
    
    subplot(3, 3, 6 + col);
    imagesc(loading_factors, noise_levels, DeltaQ);
    set(gca, 'YDir', 'normal');
    colorbar;
    colormap(jet);
    caxis([0, max(DeltaQ(:)) + 0.05]);
    
    xlabel('Load \alpha');
    ylabel('Noise \sigma');
    title(sprintf('\\DeltaQ = %s', config_label), 'Interpreter', 'tex');
end

% Add overall title
sgtitle('Figure 2: Classical Hopfield Network with Dynamic Thresholds', ...
        'FontSize', 14, 'FontWeight', 'bold');

%% Save figure
saveas(fig, '../../figures_out/main/fig02_classical_hopfield_network.png');
saveas(fig, '../../figures_out/main/fig02_classical_hopfield_network.fig');
fprintf('Figure saved to figures_out/main/\n');

%% Figures saved successfully
fprintf('\n=== Figure 2 Generation Complete ===\n');
fprintf('Figures saved to: figures_out/main/\n');

%% ========== HELPER FUNCTIONS (must be at end) ==========

function quality = hopfield_bell_threshold(N, noise_level, show_plots, M, SigmaM, ...
                                          n_patterns, max_iterations, convergence_threshold, ...
                                          weight_scale, temperature, update_type, ...
                                          gamma, bell_amp, trial_seed)
    % Main function that creates network and tests retrieval
    
    img_size = [round(sqrt(N)), round(sqrt(N))];
    
    % Create patterns
    patterns = create_patterns_fixed(N, n_patterns, img_size, trial_seed);
    
    % Create weight matrix
    W = create_weight_matrix_fixed(patterns, N, weight_scale, M, SigmaM, img_size);
    
    % Create threshold function with bell-shaped mechanism
    threshold_func = create_threshold_function_bell(patterns, W, N, gamma, bell_amp);
    
    % Test retrieval
    quality = test_retrieval_bell(N, patterns, W, threshold_func, img_size, ...
                                  noise_level, show_plots, max_iterations, ...
                                  convergence_threshold, temperature, update_type, trial_seed);
end

function threshold_func = create_threshold_function_bell(patterns, W, N, gamma, bell_amp)
    % Creates state-dependent threshold function with bell-shaped heterogeneity
    
    G_pat = W * patterns;
    G_opt = mean(G_pat(:));
    stdG = std(G_pat(:));
    
    threshold_func = @(s) local_thresholds_bell(s, W, G_opt, stdG, gamma, bell_amp, N);
end

function th = local_thresholds_bell(s, W, G_opt, stdG, gamma, bell_amp, N)
    % Compute bell-shaped thresholds for current state s
    
    S = s.' * W * s;  % Energy
    
    if stdG <= 0
        th = zeros(N, 1);
        return;
    end
    
    G = W * s;
    B = 1 / (2 * stdG^2);
    
    % Bell-shaped profile
    phi = exp(-B * (G - G_opt).^2);
    d = phi - mean(phi);
    d = d - (d.' * s) / N * s;  % Orthogonal to state
    d = bell_amp * d;
    
    % Parallel component (energy scaling)
    theta_par = (gamma * S / N) * s;
    
    th = theta_par + d;
end

function patterns = create_patterns_fixed(N, n_patterns, img_size, trial_seed)
    % Create random binary patterns
    
    patterns = zeros(N, n_patterns);
    
    for p = 1:n_patterns
        rng(trial_seed + p*1000 + 42);
        img = rand(img_size) < 0.3;
        binary_pattern = reshape(img, [], 1);
        patterns(:, p) = 2*binary_pattern - 1;
    end
end

function W = create_weight_matrix_fixed(patterns, N, weight_scale, M, SigmaM, img_size)
    % Create Hopfield weight matrix with spatial connectivity
    
    [X, Y] = meshgrid(1:img_size(2), 1:img_size(1));
    positions = [reshape(Y, [], 1), reshape(X, [], 1)];
    
    D = pdist2(positions, positions);
    P = exp(-(D.^2) / (2 * SigmaM^2));
    P = P - diag(diag(P));
    
    W = zeros(N, N);
    n_patterns = size(patterns, 2);
    
    for i = 1:N
        probs = P(i, :);
        [~, sorted_indices] = sort(probs, 'descend');
        neighbors = sorted_indices(1:min(M, N-1));
        
        for j = neighbors
            if probs(j) >= 0.01
                correlation = 0;
                for p = 1:n_patterns
                    correlation = correlation + patterns(i, p) * patterns(j, p);
                end
                W(i, j) = correlation / n_patterns;
            end
        end
    end
    
    W = W * weight_scale;
    W = (W + W') / 2;
    W = W - diag(diag(W));
end

function quality = test_retrieval_bell(N, patterns, W, threshold_func, img_size, ...
                                      noise_level, show_plots, max_iterations, ...
                                      convergence_threshold, temperature, update_type, trial_seed)
    % Test pattern retrieval with bell-shaped thresholds
    
    rng(trial_seed + 999);
    
    test_pattern_idx = 1;
    original_pattern = patterns(:, test_pattern_idx);
    
    % Add noise
    noisy_pattern = original_pattern;
    flip_indices = rand(N, 1) < noise_level;
    noisy_pattern(flip_indices) = -noisy_pattern(flip_indices);
    
    state = noisy_pattern;
    prev_state = state;
    
    converged = false;
    iteration = 0;
    
    while ~converged && iteration < max_iterations
        iteration = iteration + 1;
        thresholds = threshold_func(state);
        
        if strcmp(update_type, 'async')
            update_order = randperm(N);
            for idx = 1:N
                i = update_order(idx);
                net_input = W(i, :) * state - thresholds(i);
                
                if temperature > 0
                    prob = 1 / (1 + exp(-2 * net_input / temperature));
                    state(i) = 2 * (rand < prob) - 1;
                else
                    state(i) = sign(net_input);
                    if net_input == 0
                        state(i) = prev_state(i);
                    end
                end
            end
        else
            net_inputs = W * state - thresholds;
            if temperature > 0
                probs = 1 ./ (1 + exp(-2 * net_inputs / temperature));
                state = 2 * (rand(N, 1) < probs) - 1;
            else
                state = sign(net_inputs);
                state(net_inputs == 0) = prev_state(net_inputs == 0);
            end
        end
        
        if norm(state - prev_state) < convergence_threshold
            converged = true;
        end
        prev_state = state;
    end
    
    % Calculate quality
    overlap = (state' * original_pattern) / N;
    quality = (overlap + 1) / 2;
    
    correlation_quality = (corr(state, original_pattern) + 1) / 2;
    quality = max(quality, correlation_quality);
end
