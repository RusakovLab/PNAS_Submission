%% HOPFIELD NETWORK WITH BELL-SHAPED THRESHOLDS - CAPACITY ANALYSIS
% Integrates the bell-shaped threshold mechanism from Fig3Chat.m
% into capacity analysis framework for image recognition testing
%
% Key Parameters:
%   gamma    - Controls global energy scaling (< 0.5, negative = deeper wells)
%   bell_amp - Amplitude of bell-shaped heterogeneity (energy-neutral)

clear; clc; close all;

%% ========== USER PARAMETERS: BELL-SHAPED THRESHOLD MECHANISM ==========
gamma_values    = [-1.2, -1.0, -0.5, 0.0];   % Test multiple gamma values
bell_amp_values = [0, 1.1, 2.1, 3.1, 4.1];        % Test multiple bell amplitudes
% ========================================================================

%% EXPERIMENT CONTROL PARAMETERS
noise_levels = (0:0.1:1);
num_trials   = 100;     % Number of trials for statistical analysis
ShowPlot_    = false;  % Set to true to see individual heattrials

% CAPACITY TEST: Keep N fixed, vary n_patterns
network_size   = 144;            % 12x12 network for testing
pattern_counts = [2, 4, 8, 12, 16, 20, 24, 28, 32];
M        = 143;                  % Reduced connectivity for stability
SigmaM   = 3;                    % Smaller sigma for more local connectivity
weight_scale = 1.0;

%% SIMULATION PARAMETERS
max_iterations        = 1000;
convergence_threshold = 0.001;
temperature           = 0.1;     % Temperature for stochastic updates
update_type           = 'async';  % 'sync' or 'async' updates

%% Initialize results storage
% Results structure: [noise_levels, pattern_counts, gamma_idx, bell_idx, trials]
all_results  = zeros(length(noise_levels), length(pattern_counts), ...
                     length(gamma_values), length(bell_amp_values), num_trials);
mean_results = zeros(length(noise_levels), length(pattern_counts), ...
                     length(gamma_values), length(bell_amp_values));
std_results  = zeros(length(noise_levels), length(pattern_counts), ...
                     length(gamma_values), length(bell_amp_values));

%% Run capacity analysis with bell-shaped thresholds
fprintf('=== BELL-SHAPED THRESHOLD CAPACITY ANALYSIS ===\n');
fprintf('Network size: %d neurons\n', network_size);
fprintf('Number of trials: %d\n', num_trials);
fprintf('Testing gamma values: %s\n', mat2str(gamma_values));
fprintf('Testing bell_amp values: %s\n', mat2str(bell_amp_values));
fprintf('Testing pattern counts: %s\n', mat2str(pattern_counts));
fprintf('Testing noise levels: %s\n', mat2str(noise_levels));

% Progress tracking
total_experiments = length(noise_levels) * length(pattern_counts) * ...
                    length(gamma_values) * length(bell_amp_values) * num_trials;
experiment_count = 0;
start_time = tic;

for noise_idx = 1:length(noise_levels)
    noise_level = noise_levels(noise_idx);
    fprintf('\n=== NOISE LEVEL: %.1f ===\n', noise_level);
    
    for p_idx = 1:length(pattern_counts)
        n_patterns    = pattern_counts(p_idx);
        loading_factor = n_patterns / network_size;
        
        fprintf('\n--- Testing %d patterns (α = %.3f) ---\n', n_patterns, loading_factor);
        
        for gamma_idx = 1:length(gamma_values)
            gamma = gamma_values(gamma_idx);
            
            for bell_idx = 1:length(bell_amp_values)
                bell_amp = bell_amp_values(bell_idx);
                fprintf('  γ=%.2f, bell_amp=%.1f: ', gamma, bell_amp);
                
                % Run multiple trials
                trial_results = zeros(num_trials, 1);
                for trial = 1:num_trials
                    trial_seed = trial * 1000 + noise_idx * 100 + p_idx * 10 + ...
                                 gamma_idx * 5 + bell_idx;
                    
                    quality = hopfield_bell_threshold(network_size, noise_level, ...
                        ShowPlot_ && trial == 1 && p_idx <= 3, M, SigmaM, ...
                        n_patterns, max_iterations, convergence_threshold, weight_scale, ...
                        temperature, update_type, gamma, bell_amp, trial_seed);
                    
                    trial_results(trial) = quality;
                    experiment_count = experiment_count + 1;
                    
                    % Progress indicator
                    if mod(trial, max(1, floor(num_trials/5))) == 0
                        fprintf('.');
                    end
                end
                
                % Store results
                all_results(noise_idx, p_idx, gamma_idx, bell_idx, :) = trial_results;
                mean_results(noise_idx, p_idx, gamma_idx, bell_idx) = mean(trial_results);
                std_results(noise_idx, p_idx, gamma_idx, bell_idx)  = std(trial_results);
                
                fprintf(' Mean=%.3f±%.3f\n', ...
                    mean_results(noise_idx, p_idx, gamma_idx, bell_idx), ...
                    std_results(noise_idx, p_idx, gamma_idx, bell_idx));
            end
        end
    end
    
    elapsed   = toc(start_time);
    remaining = (total_experiments - experiment_count) * elapsed / max(experiment_count,1);
    fprintf('Progress: %d/%d (%.1f%%), Est. remaining: %.1f min\n', ...
        experiment_count, total_experiments, 100*experiment_count/total_experiments, ...
        remaining/60);
end

fprintf('\nCompleted %d experiments in %.1f minutes\n', experiment_count, toc(start_time)/60);

%% Save results to file
timestamp = datestr(now, 'yyyy-mm-dd_HH-MM-SS');
filename  = sprintf('bell_hopfield_capacity_%s.mat', timestamp);

save(filename, 'mean_results', 'std_results', 'all_results', ...
     'noise_levels', 'pattern_counts', 'gamma_values', 'bell_amp_values', ...
     'num_trials', 'network_size');

fprintf('\nResults saved to: %s\n', filename);

%% Create comprehensive comparison plots
loading_factors = pattern_counts / network_size;

% Plot 1: Effect of gamma for fixed bell_amp
figure('Position', [100, 100, 1400, 900]);
bell_idx_to_plot = 1;  % Plot bell_amp index 3 (adjust as needed)
bell_amp_plot = bell_amp_values(bell_idx_to_plot);

for noise_idx = 1:min(6, length(noise_levels))
    subplot(2, 3, noise_idx);
    hold on; grid on;
    
    colors = lines(length(gamma_values));
    for gamma_idx = 1:length(gamma_values)
        errorbar(loading_factors, ...
                mean_results(noise_idx, :, gamma_idx, bell_idx_to_plot), ...
                std_results(noise_idx,  :, gamma_idx, bell_idx_to_plot), ...
                '-o', 'LineWidth', 2, 'Color', colors(gamma_idx,:), ...
                'DisplayName', sprintf('\\gamma = %.2f', gamma_values(gamma_idx)));
    end
    
    xlabel('Loading Factor \alpha = n_{patterns}/N');
    ylabel('Recall Quality');
    title(sprintf('Noise = %.1f, bell\\_amp = %.1f', noise_levels(noise_idx), bell_amp_plot));
    legend('Location', 'best');
    ylim([0, 1.05]);
end

sgtitle('Effect of \gamma on Network Capacity (Fixed bell\_amp)');
saveas(gcf, sprintf('gamma_effect_%s.png', timestamp));

% Plot 2: Effect of bell_amp for fixed gamma
figure('Position', [150, 150, 1400, 900]);
gamma_idx_to_plot = 4;  % Plot gamma index 2 (adjust as needed)
gamma_plot = gamma_values(gamma_idx_to_plot);

for noise_idx = 1:min(6, length(noise_levels))
    subplot(2, 3, noise_idx);
    hold on; grid on;
    
    colors = lines(length(bell_amp_values));
    for bell_idx = 1:length(bell_amp_values)
        errorbar(loading_factors, ...
                mean_results(noise_idx, :, gamma_idx_to_plot, bell_idx), ...
                std_results(noise_idx,  :, gamma_idx_to_plot, bell_idx), ...
                '-o', 'LineWidth', 2, 'Color', colors(bell_idx,:), ...
                'DisplayName', sprintf('bell\\_amp = %.1f', bell_amp_values(bell_idx)));
    end
    
    xlabel('Loading Factor \alpha = n_{patterns}/N');
    ylabel('Recall Quality');
    title(sprintf('Noise = %.1f, \\gamma = %.2f', noise_levels(noise_idx), gamma_plot));
    legend('Location', 'best');
    ylim([0, 1.05]);
end

sgtitle('Effect of bell\_amp on Network Capacity (Fixed \gamma)');
saveas(gcf, sprintf('bell_amp_effect_%s.png', timestamp));

% Plot 3: Heatmap of best configurations
figure('Position', [200, 200, 1200, 400]);

% noise 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
%       1,  2,   3,   4,   5,   6,   7,   8,   9,  10,   11  
%pattern_counts = [2, 4, 8, 12, 16, 20, 24, 28, 32];
% 0.0138	0.0277	0.0555	0.08333	0.1805	0.1388	0.166	0.194	0.222



noise_idx_heatmap = 1;  % Noise level to show (index 5 = 0.4)
p_idx_heatmap     = 4;  % Pattern count to show

heatmap_data = squeeze(mean_results(noise_idx_heatmap, p_idx_heatmap, :, :));

imagesc(heatmap_data);
colorbar;
colormap(jet);
caxis([0, 1]);

xlabel('bell\_amp index');
ylabel('\gamma index');
title(sprintf('Recall Quality Heatmap: Noise=%.1f, Patterns=%d', ...
    noise_levels(noise_idx_heatmap), pattern_counts(p_idx_heatmap)));

xticks(1:length(bell_amp_values));
xticklabels(arrayfun(@num2str, bell_amp_values, 'UniformOutput', false));
yticks(1:length(gamma_values));
yticklabels(arrayfun(@num2str, gamma_values, 'UniformOutput', false));

% Add text annotations
for i = 1:length(gamma_values)
    for j = 1:length(bell_amp_values)
        text(j, i, sprintf('%.2f', heatmap_data(i,j)), ...
            'HorizontalAlignment', 'center', 'Color', 'white', 'FontWeight', 'bold');
    end
end

saveas(gcf, sprintf('parameter_heatmap_%s.png', timestamp));

%% ===== NEW: Recall quality vs. noise (γ / bell_amp sweeps) =====
% Choose a representative loading factor (pattern count) to slice on:
p_idx_noiseplot = 5;  % e.g., pattern_counts(5) => alpha ≈ 0.111
alpha_plot = pattern_counts(p_idx_noiseplot) / network_size;

% Figure A: vary gamma at fixed bell_amp
figure('Position', [220, 220, 1400, 900]);
bell_idx_to_plot = 1;                      % pick a bell_amp index to show
bell_amp_plot = bell_amp_values(bell_idx_to_plot);

hold on;
colors = lines(length(gamma_values));
for gamma_idx = 1:length(gamma_values)
    mu = squeeze(mean_results(:, p_idx_noiseplot, gamma_idx, bell_idx_to_plot));   % [noise]
    sd = squeeze(std_results(:,  p_idx_noiseplot, gamma_idx, bell_idx_to_plot));
    errorbar(noise_levels, mu, sd, '-o', 'LineWidth', 2, ...
             'Color', colors(gamma_idx,:), ...
             'DisplayName', sprintf('\\gamma = %.2f', gamma_values(gamma_idx)));
end
grid on; box on;
xlabel('Noise level'); ylabel('Recall Quality');
title(sprintf('Quality vs Noise (fixed bell\\_amp = %.1f, \\alpha = %.3f)', bell_amp_plot, alpha_plot));
legend('Location','southwest');
ylim([0, 1.05]);
saveas(gcf, sprintf('quality_vs_noise_fixed_bellamp_%s.png', timestamp));

% Figure B: vary bell_amp at fixed gamma
figure('Position', [240, 240, 1400, 900]);
gamma_idx_to_plot = 4;                      % pick a gamma index to show
gamma_plot = gamma_values(gamma_idx_to_plot);

hold on;
colors = lines(length(bell_amp_values));
for bell_idx = 1:length(bell_amp_values)
    mu = squeeze(mean_results(:, p_idx_noiseplot, gamma_idx_to_plot, bell_idx));   % [noise]
    sd = squeeze(std_results(:,  p_idx_noiseplot, gamma_idx_to_plot, bell_idx));
    errorbar(noise_levels, mu, sd, '-o', 'LineWidth', 2, ...
             'Color', colors(bell_idx,:), ...
             'DisplayName', sprintf('bell\\_amp = %.1f', bell_amp_values(bell_idx)));
end
grid on; box on;
xlabel('Noise level'); ylabel('Recall Quality');
title(sprintf('Quality vs Noise (fixed \\gamma = %.2f, \\alpha = %.3f)', gamma_plot, alpha_plot));
legend('Location','southwest');
ylim([0, 1.05]);
saveas(gcf, sprintf('quality_vs_noise_fixed_gamma_%s.png', timestamp));

%% Generate summary report
txt_filename = sprintf('bell_hopfield_summary_%s.txt', timestamp);
fid = fopen(txt_filename, 'w');

fprintf(fid, 'BELL-SHAPED HOPFIELD NETWORK CAPACITY ANALYSIS\n');
fprintf(fid, '==============================================\n\n');
fprintf(fid, 'Experiment Parameters:\n');
fprintf(fid, '- Network size: %d neurons\n', network_size);
fprintf(fid, '- Number of trials: %d\n', num_trials);
fprintf(fid, '- Gamma values: %s\n', mat2str(gamma_values));
fprintf(fid, '- Bell_amp values: %s\n', mat2str(bell_amp_values));
fprintf(fid, '- Pattern counts: %s\n', mat2str(pattern_counts));
fprintf(fid, '- Timestamp: %s\n\n', timestamp);

% Find best configurations
fprintf(fid, '\nBEST CONFIGURATIONS:\n');
fprintf(fid, '====================\n\n');

for noise_idx = [1, 5, 9]  % Show results for noise 0.0, 0.4, 0.8
    fprintf(fid, 'Noise Level: %.1f\n', noise_levels(noise_idx));
    for p_idx = [3, 5, 7]  % Show results for different pattern counts
        n_patterns = pattern_counts(p_idx);
        
        % Find best gamma and bell_amp combination
        results_slice = squeeze(mean_results(noise_idx, p_idx, :, :));
        [max_quality, max_idx] = max(results_slice(:));
        [best_gamma_idx, best_bell_idx] = ind2sub(size(results_slice), max_idx);
        
        fprintf(fid, '  Patterns=%d (α=%.3f): Quality=%.3f at γ=%.2f, bell_amp=%.1f\n', ...
            n_patterns, n_patterns/network_size, max_quality, ...
            gamma_values(best_gamma_idx), bell_amp_values(best_bell_idx));
    end
    fprintf(fid, '\n');
end

fclose(fid);
fprintf('Summary saved to: %s\n', txt_filename);

%% MAIN HOPFIELD FUNCTION WITH BELL-SHAPED THRESHOLDS
function quality = hopfield_bell_threshold(N, noise_level, show_plots, M, SigmaM, ...
                                          n_patterns, max_iterations, convergence_threshold, ...
                                          weight_scale, temperature, update_type, ...
                                          gamma, bell_amp, trial_seed)
    
    grid_size = sqrt(N);
    if floor(grid_size) ~= grid_size
        grid_size = round(sqrt(N));
        N = grid_size^2;
    end
    img_size = [grid_size, grid_size];

    % Create patterns
    patterns = create_patterns_fixed(N, n_patterns, img_size, trial_seed);

    % Create weight matrix
    W = create_weight_matrix_fixed(patterns, N, weight_scale, M, SigmaM, img_size);

    % Calculate bell-shaped thresholds using mechanism from Fig2Chat.m
    [thresholds_func, G_opt, stdG] = create_bell_threshold_function(W, patterns, gamma, bell_amp, N); %#ok<ASGLU>

    % Test retrieval with bell-shaped thresholds
    quality = test_retrieval_bell(N, patterns, W, thresholds_func, img_size, ...
                                 noise_level, show_plots, max_iterations, ...
                                 convergence_threshold, temperature, update_type, trial_seed);
end

%% BELL-SHAPED THRESHOLD MECHANISM (from Fig2Chat.m)
function [threshold_func, G_opt, stdG] = create_bell_threshold_function(W, patterns, gamma, bell_amp, N)
    % Creates a function handle that returns bell-shaped thresholds for any state
    % This implements the minima-preserving threshold mechanism from Fig2Chat.m
    
    % Calculate statistics for bell-shape from all patterns
    G_pat = W * patterns;
    G_opt = mean(G_pat(:));  % Center bell-shape around average pattern conductance
    
    % Calculate standard deviation across all possible inputs (approximated by patterns)
    stdG = std(G_pat(:));
    
    % Return function handle that computes thresholds for any state s
    threshold_func = @(s) local_thresholds_bell(s, W, G_opt, stdG, gamma, bell_amp, N);
end

function th = local_thresholds_bell(s, W, G_opt, stdG, gamma, bell_amp, N)
    % Returns bell-shaped thresholds that preserve minima
    % Implementation from Fig2Chat.m: local_thresholds_minima_preserving
    
    S = s.' * W * s;  % State energy contribution
    
    if stdG <= 0
        th = zeros(N, 1);
        return;
    end
    
    G = W * s;  % Synaptic input for each neuron
    B = 1 / (2 * stdG^2);
    
    % Bell-shaped profile centered at G_opt
    phi = exp(-B * (G - G_opt).^2);
    
    % Make zero-mean across neurons
    d = phi - mean(phi);
    
    % Make orthogonal to s (energy-neutral constraint)
    d = d - (d.' * s) / N * s;
    
    % Apply user amplitude knob
    d = bell_amp * d;
    
    % Parallel component (controls global energy rescale)
    theta_par = (gamma * S / N) * s;
    
    % Final threshold: parallel (energy-scaling) + orthogonal (heterogeneity)
    th = theta_par + d;
end

%% PATTERN AND NETWORK CREATION FUNCTIONS
function patterns = create_patterns_fixed(N, n_patterns, img_size, trial_seed)
    patterns = zeros(N, n_patterns);
    
    for p = 1:n_patterns
        rng(trial_seed + p*1000 + 42);
        
        % Create random binary patterns with 30% activity, convert to ±1
        img = rand(img_size) < 0.3;
        binary_pattern = reshape(img, [], 1);
        patterns(:, p) = 2*binary_pattern - 1;
    end
    
    if n_patterns > 1
        % Check pattern correlations
        correlations = zeros(n_patterns, n_patterns);
        for i = 1:n_patterns
            for j = 1:n_patterns
                correlations(i,j) = corr(patterns(:,i), patterns(:,j));
            end
        end
        avg_correlation = mean(abs(correlations(~eye(n_patterns))));
        if avg_correlation > 0.3
            fprintf('Warning: High pattern correlation: %.3f\n', avg_correlation);
        end
    end
end

function W = create_weight_matrix_fixed(patterns, N, weight_scale, M, SigmaM, img_size)
    % Create classical Hopfield weight matrix with spatial connectivity
    [X, Y] = meshgrid(1:img_size(2), 1:img_size(1)); %#ok<NASGU>
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
    W = (W + W') / 2;  % Symmetry
    W = W - diag(diag(W));  % No self-connections
end

%% RETRIEVAL TESTING WITH BELL-SHAPED THRESHOLDS
function quality = test_retrieval_bell(N, patterns, W, threshold_func, img_size, ...
                                      noise_level, show_plots, max_iterations, ...
                                      convergence_threshold, temperature, update_type, trial_seed)
    
    rng(trial_seed + 999);
    
    % Test pattern 1
    test_pattern_idx = 1;
    original_pattern = patterns(:, test_pattern_idx);
    
    % Create noisy version
    noisy_pattern = original_pattern;
    flip_indices = rand(N, 1) < noise_level;
    noisy_pattern(flip_indices) = -noisy_pattern(flip_indices);
    
    % Initialize network state
    state = noisy_pattern;
    prev_state = state;
    
    % Run Hopfield dynamics with bell-shaped thresholds
    converged = false;
    iteration = 0;
    
    while ~converged && iteration < max_iterations
        iteration = iteration + 1;
        
        % Get thresholds for CURRENT state (key difference: state-dependent)
        thresholds = threshold_func(state);
        
        if strcmp(update_type, 'async')
            % Asynchronous updates
            update_order = randperm(N);
            for idx = 1:N
                i = update_order(idx);
                
                % Calculate net input with bell-shaped threshold
                net_input = W(i, :) * state - thresholds(i);
                
                if temperature > 0
                    prob = 1 / (1 + exp(-2 * net_input / temperature));
                    if rand < prob
                        state(i) = 1;
                    else
                        state(i) = -1;
                    end
                else
                    if net_input > 0
                        state(i) = 1;
                    elseif net_input < 0
                        state(i) = -1;
                    end
                end
            end
        else
            % Synchronous updates
            net_inputs = W * state - thresholds;
            
            if temperature > 0
                probs = 1 ./ (1 + exp(-2 * net_inputs / temperature));
                state = 2 * (rand(N, 1) < probs) - 1;
            else
                state = sign(net_inputs);
                state(net_inputs == 0) = prev_state(net_inputs == 0);
            end
        end
        
        % Check convergence
        if norm(state - prev_state) < convergence_threshold
            converged = true;
        end
        
        prev_state = state;
    end
    
    if show_plots
        fprintf('  Converged after %d iterations\n', iteration);
    end
    
    % Calculate retrieval quality
    overlap = (state' * original_pattern) / N;
    quality = (overlap + 1) / 2;  % Normalize to [0,1]
    
    correlation_quality = (corr(state, original_pattern) + 1) / 2;
    quality = max(quality, correlation_quality);
    
    if show_plots
        fprintf('  Overlap: %.3f, Correlation: %.3f, Quality: %.3f\n', ...
                overlap, corr(state, original_pattern), quality);
        
        figure;
        subplot(1,3,1);
        imshow(reshape((original_pattern + 1)/2, img_size), []);
        title('Original Pattern');
        
        subplot(1,3,2);
        imshow(reshape((noisy_pattern + 1)/2, img_size), []);
        title(sprintf('Noisy Input (%.0f%%)', noise_level*100));
        
        subplot(1,3,3);
        imshow(reshape((state + 1)/2, img_size), []);
        title(sprintf('Retrieved (Q=%.2f)', quality));
        
        pause(1);
    end
end
