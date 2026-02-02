%% LIF Hopfield Network with Non-Uniform (Bell-Shaped) Thresholds
% Recall quality as a function of noise, loading factor, gamma (γ) and bell_amp (β)

clear; close all; clc;

%% ---------------- Parameters ----------------
N = 144;                                   % Number of neurons
pattern_counts = [2, 4, 8, 12, 16, 20, 24, 28, 32];
noise_levels   = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
num_trials     = 10;                        % Trials per condition

% LIF neuron parameters (kept from your uniform model)
tau_m    = 10;      % ms
V_rest   = 0;       % normalized
V_thresh = 1;       % nominal threshold (not used directly; we use theta)
V_reset  = 0;       % normalized
dt       = 0.5;     % ms
T_sim    = 200;     % ms total
T_relax  = 50;      % ms: initial relax with clamping decay
update_every_ms = 5;                         % update state every 5 ms
update_every   = max(1, round(update_every_ms/dt)); % Corrected to max(1,...)

% Bell-shaped threshold parameters (NEW)
gamma_values    = [-2.0, -1.0, -0.5, 0.0];  % energy scaling (parallel component)
bell_amp_values = [0, 1, 2, 3, 4];       % heterogeneity amplitude (β)

%% ------------- Storage for results -------------
% recall_quality(p_idx, noise_idx, gamma_idx, bell_idx)
recall_quality = zeros(length(pattern_counts), length(noise_levels), ...
                       length(gamma_values), length(bell_amp_values));

fprintf('Running simulations with non-uniform thresholds (γ, β)...\n');

%% ----------------- Main loops ------------------
for p_idx = 1:length(pattern_counts)
    P = pattern_counts(p_idx);
    alpha = P / N;
    fprintf('Patterns: %d  (alpha = %.3f)\n', P, alpha);

    for n_idx = 1:length(noise_levels)
        noise = noise_levels(n_idx);

        for g_idx = 1:length(gamma_values)
            gamma = gamma_values(g_idx);

            for b_idx = 1:length(bell_amp_values)
                bell_amp = bell_amp_values(b_idx);

                quality_sum = 0;

                for trial = 1:num_trials
                    rng(1000 + 100*p_idx + 10*n_idx + g_idx + b_idx + 10000*trial);

                    % ----------- Patterns and weights -----------
                    % Random patterns in {-1,+1}
                    patterns = 2 * randi([0, 1], N, P) - 1;

                    % Hebbian weights (no self-connections)
                    W = zeros(N, N);
                    for i = 1:N
                        for j = 1:N
                            if i ~= j
                                W(i, j) = (1/N) * sum(patterns(i, :) .* patterns(j, :));
                            end
                        end
                    end

                    % Select a random pattern to test
                    test_idx = randi(P);
                    orig = patterns(:, test_idx);

                    % Add flip noise
                    noisy = orig;
                    num_flips = round(noise * N);
                    if num_flips > 0
                        flip_idx = randperm(N, num_flips);
                        noisy(flip_idx) = -noisy(flip_idx);
                    end

                    % ----------- Bell-shape statistics (from patterns) -----------
                    % Use all patterns to estimate G statistics
                    G_pat  = W * patterns;                 % N x P
                    G_opt  = mean(G_pat(:));
                    stdG   = std(G_pat(:));
                    if stdG == 0, stdG = eps; end

                    % ----------- Initialize dynamics -----------
                    state = noisy;                          % current ±1 state
                    V     = V_rest * ones(N, 1);            % membrane voltage

                    num_steps = round(T_sim / dt);

                    for t = 1:num_steps
                        % --- compute bell-shaped thresholds for CURRENT state ---
                        % theta = theta_par + d
                        % parallel: (gamma * S / N) * s
                        S  = state.' * W * state;
                        G  = W * state;
                        B  = 1 / (2 * (stdG^2));
                        phi = exp(-B * (G - G_opt).^2);

                        d = phi - mean(phi);                 % zero-mean
                        d = d - (d.' * state) / N * state;   % orthogonal to state
                        d = bell_amp * d;                    % heterogeneity amplitude

                        theta_par = (gamma * S / N) * state; % energy scaling
                        theta = theta_par + d;               % per-neuron threshold offset

                        % --- effective input current (subtract threshold) ---
                        h = W * state;
                        bias_strength = max(0, 1 - t*dt/T_relax);  % decaying cue
                        I_input = (h - theta) + bias_strength * 5 * noisy;

                        % --- LIF integration ---
                        dV = (-(V - V_rest) + I_input) / tau_m * dt;
                        V  = V + dV;

                        % --- periodic state update via soft decision ---
                        if mod(t, update_every) == 0
                            % Compare voltage to 0 as in your original; threshold already
                            % entered via I_input. Keep sign rule:
                            state = sign(V);
                            state(state == 0) = 1;

                            % reset voltages to rest to mimic discrete update
                            V = V_rest * ones(N,1);
                        end
                    end

                    % Final recalled pattern
                    recalled = state;

                    % Overlap-based recall quality in [0,1]
                    overlap = sum(recalled == orig) / N;
                    quality_sum = quality_sum + overlap;
                end

                recall_quality(p_idx, n_idx, g_idx, b_idx) = quality_sum / num_trials;
            end
        end
    end
end

fprintf('Simulations complete!\n');

%% -------------- Plotting --------------


% Figure 1: Recall quality vs Noise for different loading factors
% (slice at a chosen gamma and bell_amp)
gamma_idx_plot = 2;                 % choose which γ index to plot
bell_idx_plot  = 2;                 % choose which β index to plot
gamma_plot     = gamma_values(gamma_idx_plot);
beta_plot      = bell_amp_values(bell_idx_plot);

figure('Position', [100, 100, 1200, 500]);
subplot(1, 2, 1);
colors = jet(length(pattern_counts));
hold on;
for p_idx = 1:length(pattern_counts)
    mu = squeeze(recall_quality(p_idx, :, gamma_idx_plot, bell_idx_plot));
    plot(noise_levels, mu, '-o', ...
        'LineWidth', 2, 'MarkerSize', 6, 'Color', colors(p_idx,:), ...
        'DisplayName', sprintf('P=%d (\\alpha=%.2f)', pattern_counts(p_idx), pattern_counts(p_idx)/N));
end
xlabel('Noise Level', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Recall Quality', 'FontSize', 12, 'FontWeight', 'bold');
title(sprintf('Q vs Noise (\\gamma=%.2f, \\beta=%.1f)', gamma_plot, beta_plot), ...
      'FontSize', 14, 'FontWeight', 'bold');
legend('Location', 'southwest', 'FontSize', 9);
grid on; ylim([0, 1]); xlim([0, 1]); hold off;

% Figure 2: Recall quality vs Loading factor (slice at same γ, β)
subplot(1, 2, 2);
loading_factors = pattern_counts / N;
colors2 = jet(length(noise_levels));
hold on;
sample_noise_idx = 1:2:length(noise_levels);
for n_idx = sample_noise_idx
    mu = squeeze(recall_quality(:, n_idx, gamma_idx_plot, bell_idx_plot));
    plot(loading_factors, mu, '-s', ...
        'LineWidth', 2, 'MarkerSize', 6, 'Color', colors2(n_idx,:), ...
        'DisplayName', sprintf('Noise=%.1f', noise_levels(n_idx)));
end
xlabel('Loading Factor (\alpha = P/N)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Recall Quality', 'FontSize', 12, 'FontWeight', 'bold');
title(sprintf('Q vs Loading (\\gamma=%.2f, \\beta=%.1f)', gamma_plot, beta_plot), ...
      'FontSize', 14, 'FontWeight', 'bold');
legend('Location', 'northeast', 'FontSize', 9);
grid on; ylim([0, 1]); hold off;

% Figure 3: Heatmap at fixed gamma & bell_amp
figure('Position', [150, 150, 800, 600]);
mu_slice = squeeze(recall_quality(:, :, gamma_idx_plot, bell_idx_plot)); % P x noise
imagesc(noise_levels, loading_factors, mu_slice);
colorbar;
xlabel('Noise Level', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Loading Factor (\alpha = P/N)', 'FontSize', 12, 'FontWeight', 'bold');
title(sprintf('Heatmap (\\gamma=%.2f, \\beta=%.1f)', gamma_plot, beta_plot), ...
      'FontSize', 14, 'FontWeight', 'bold');
set(gca, 'YDir', 'normal'); colormap(jet); caxis([0, 1]);
hold on; contour(noise_levels, loading_factors, mu_slice, [0.5, 0.7, 0.9], ...
                 'LineColor', 'white', 'LineWidth', 2); hold off;

% Figure 4A: Quality vs noise for different gamma (fixed beta, fixed P)
figure('Position', [200, 200, 1200, 450]);
p_idx_noiseplot = 5;  % choose P index for slicing (alpha ~ 0.111)
alpha_plot = pattern_counts(p_idx_noiseplot)/N;
subplot(1,2,1);
hold on; colors = lines(length(gamma_values));
for g_idx = 1:length(gamma_values)
    mu = squeeze(recall_quality(p_idx_noiseplot, :, g_idx, bell_idx_plot));
    plot(noise_levels, mu, '-o', 'LineWidth', 2, 'Color', colors(g_idx,:), ...
        'DisplayName', sprintf('\\gamma=%.2f', gamma_values(g_idx)));
end
grid on; xlabel('Noise'); ylabel('Recall Quality');
title(sprintf('Q vs Noise (\\beta=%.1f, \\alpha=%.3f)', beta_plot, alpha_plot));
legend('Location','southwest'); ylim([0,1]);

% Figure 4B: Quality vs noise for different beta (fixed gamma, same P)
subplot(1,2,2);
hold on; colors = lines(length(bell_amp_values));
for b_idx = 1:length(bell_amp_values)
    mu = squeeze(recall_quality(p_idx_noiseplot, :, gamma_idx_plot, b_idx));
    plot(noise_levels, mu, '-o', 'LineWidth', 2, 'Color', colors(b_idx,:), ...
        'DisplayName', sprintf('\\beta=%.1f', bell_amp_values(b_idx)));
end
grid on; xlabel('Noise'); ylabel('Recall Quality');
title(sprintf('Q vs Noise (\\gamma=%.2f, \\alpha=%.3f)', gamma_plot, alpha_plot));
legend('Location','southwest'); ylim([0,1]);

%% -------- Summary (based on selected slice) --------
fprintf('\n=== Summary (slice: gamma=%.2f, beta=%.1f) ===\n', gamma_plot, beta_plot);
zero_noise_quality = mu_slice(:, 1);           % at noise = 0
critical_idx = find(zero_noise_quality < 0.9, 1);
if ~isempty(critical_idx)
    fprintf('Critical loading factor (Q<0.9 at noise=0): alpha ≈ %.3f (P=%d)\n', ...
        loading_factors(critical_idx), pattern_counts(critical_idx));
else
    fprintf('All tested loading factors achieved Q ≥ 0.9 at zero noise (this slice).\n');
end
capacity_idx = find(zero_noise_quality > 0.8, 1, 'last');
if ~isempty(capacity_idx)
    fprintf('Capacity estimate (Q>0.8 at noise=0): P ≈ %d (alpha ≈ %.3f)\n', ...
        pattern_counts(capacity_idx), loading_factors(capacity_idx));
end
fprintf('==============================================\n');
%% ===== EXTRA PLOTS (append-only) =====
% Choose the slice (fixed alpha via P index) and which gamma/beta to hold fixed
p_idx_alpha      = min(5, length(pattern_counts));   % pick P index (alpha = pattern_counts(p_idx_alpha)/N)
bell_idx_fixed   = min(3, length(bell_amp_values));  % fix beta index for Plot 1
gamma_idx_fixed  = min(2, length(gamma_values));     % fix gamma index for Plot 2

alpha_val  = pattern_counts(p_idx_alpha) / N;
beta_val   = bell_amp_values(bell_idx_fixed);
gamma_val  = gamma_values(gamma_idx_fixed);

% ---- Plot 1: Q vs Noise for different gamma (fixed alpha & beta) ----
figure('Position', [260, 260, 1100, 420]);
hold on; grid on; box on;
colors = lines(length(gamma_values));
for g_idx = 1:length(gamma_values)
    mu = squeeze(recall_quality(p_idx_alpha, :, g_idx, bell_idx_fixed));   % [noise]
    plot(noise_levels, mu, '-o', 'LineWidth', 2, 'MarkerSize', 6, ...
        'Color', colors(g_idx,:), 'DisplayName', sprintf('\\gamma = %.2f', gamma_values(g_idx)));
end
xlabel('Noise level');
ylabel('Recall Quality');
title(sprintf('Recall Quality vs Noise (fixed \\alpha = %.3f, \\beta = %.1f)', alpha_val, beta_val));
legend('Location','southwest');
ylim([0,1]); xlim([min(noise_levels) max(noise_levels)]);
hold off;

% ---- Plot 2: Q vs Noise for different beta (fixed alpha & gamma) ----
figure('Position', [300, 300, 1100, 420]);
hold on; grid on; box on;
colors = lines(length(bell_amp_values));
for b_idx = 1:length(bell_amp_values)
    mu = squeeze(recall_quality(p_idx_alpha, :, gamma_idx_fixed, b_idx)); % [noise]
    plot(noise_levels, mu, '-o', 'LineWidth', 2, 'MarkerSize', 6, ...
        'Color', colors(b_idx,:), 'DisplayName', sprintf('\\beta = %.1f', bell_amp_values(b_idx)));
end
xlabel('Noise level');
ylabel('Recall Quality');
title(sprintf('Recall Quality vs Noise (fixed \\alpha = %.3f, \\gamma = %.2f)', alpha_val, gamma_val));
legend('Location','southwest');
ylim([0,1]); xlim([min(noise_levels) max(noise_levels)]);
hold off;

%% ===== PREVIOUSLY REQUESTED PLOTS (P=32) =====

% Define fixed indices for P=32, gamma=0, and beta=0
p_idx_32    = find(pattern_counts == 32);
gamma_idx_0 = find(gamma_values == 0);
bell_idx_0  = find(bell_amp_values == 0);
alpha_32    = pattern_counts(p_idx_32) / N;

% ---- Plot 3: Q vs Noise for different gamma (fixed P=32 & beta=0) ----
figure('Position', [340, 340, 1100, 420]);
hold on; grid on; box on;
colors = lines(length(gamma_values));
for g_idx = 1:length(gamma_values)
    mu = squeeze(recall_quality(p_idx_32, :, g_idx, bell_idx_0)); % [noise]
    plot(noise_levels, mu, '-o', 'LineWidth', 2, 'MarkerSize', 6, ...
        'Color', colors(g_idx,:), 'DisplayName', sprintf('\\gamma = %.2f', gamma_values(g_idx)));
end
xlabel('Noise level');
ylabel('Recall Quality');
title(sprintf('Recall Quality vs Noise (fixed P=%d (\\alpha = %.3f), \\beta = %.1f)', ...
    pattern_counts(p_idx_32), alpha_32, bell_amp_values(bell_idx_0)));
legend('Location','southwest');
ylim([0,1]); xlim([min(noise_levels) max(noise_levels)]);
hold off;

% ---- Plot 4: Q vs Noise for different beta (fixed P=32 & gamma=0) ----
figure('Position', [380, 380, 1100, 420]);
hold on; grid on; box on;
colors = lines(length(bell_amp_values));
for b_idx = 1:length(bell_amp_values)
    mu = squeeze(recall_quality(p_idx_32, :, gamma_idx_0, b_idx)); % [noise]
    plot(noise_levels, mu, '-o', 'LineWidth', 2, 'MarkerSize', 6, ...
        'Color', colors(b_idx,:), 'DisplayName', sprintf('\\beta = %.1f', bell_amp_values(b_idx)));
end
xlabel('Noise level');
ylabel('Recall Quality');
title(sprintf('Recall Quality vs Noise (fixed P=%d (\\alpha = %.3f), \\gamma = %.2f)', ...
    pattern_counts(p_idx_32), alpha_32, gamma_values(gamma_idx_0)));
legend('Location','southwest');
ylim([0,1]); xlim([min(noise_levels) max(noise_levels)]);
hold off;

%% ... (All simulation loops and previous plots remain unchanged) ...

%% ===== NEW REQUESTED HEATMAP (Q vs gamma and beta at P=32, Noise=0) =====

%% ... (Code remains unchanged up to here) ...

%% ===== NEW REQUESTED HEATMAP (Q vs gamma and beta at P=32) =====

% --- Heatmap Parameters ---
% === CHANGE THIS VALUE for future computations (e.g., 0.1, 0.5, 1.0) ===
noise_level_heatmap = 0.9; 
p_val_heatmap = 32;        % Fixed pattern count for heatmap

% Define fixed indices
p_idx_32    = find(pattern_counts == p_val_heatmap);
noise_idx_param = find(noise_levels == noise_level_heatmap); 

% Error handling for index finding (Recommended for robustness)
if isempty(p_idx_32)
    error('P=%d not found in pattern_counts array.', p_val_heatmap);
end
if isempty(noise_idx_param)
    error('Noise level %.1f not found in noise_levels array. Please choose a value from 0 to 1 with 0.1 increments.', noise_level_heatmap);
end

alpha_32    = pattern_counts(p_idx_32) / N;

% Slice the quality data: Q(P=32, Noise=param, gamma, beta)
% The result is a matrix of size [gamma_values x bell_amp_values]
Q_gamma_beta_slice = squeeze(recall_quality(p_idx_32, noise_idx_param, :, :));

% ----------------- HEATMAP PLOTTING (gamma -2 top to 0 bottom) -----------------

% Use original gamma_values: [-2.0, -1.0, -0.5, 0.0]
gamma_values_plot = gamma_values; 
Q_slice_plot = Q_gamma_beta_slice;

figure('Position', [420, 420, 800, 600]);
% Plot using the original ordering. Default YDir='reverse' makes -2.0 appear at the top.
imagesc(bell_amp_values, gamma_values_plot, Q_slice_plot); 
colorbar;

xlabel('Heterogeneity Threshold \beta', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Threshold Scaling \gamma', 'FontSize', 12, 'FontWeight', 'bold');
title(sprintf('Recall Quality vs (\\gamma, \\beta) at P=%d (\\alpha=%.3f), Noise=%.1f (\\gamma -2 top to 0 bottom)', ...
      pattern_counts(p_idx_32), alpha_32, noise_level_heatmap), ... % Now uses the parameter
      'FontSize', 14, 'FontWeight', 'bold');
      
% This reverts to the default 'YDir', 'reverse', which makes the axis run from -2 at the top to 0 at the bottom.
colormap(jet); caxis([0, 1]);
hold on; 
contour(bell_amp_values, gamma_values_plot, Q_slice_plot, [0.5, 0.7, 0.9], ...
                 'LineColor', 'white', 'LineWidth', 2); 
hold off;



%% ===== SAVE ALL RESULTS AND PARAMETERS =====
fprintf('\nSaving all computational results to MAT-file...\n');

save(sprintf('Fig4A_SpikingHopfield_Data_%s.mat', datestr(now, 'yyyy-mm-dd_HH-MM-SS')), ...
     'recall_quality', 'N', 'pattern_counts', 'noise_levels', 'num_trials', ...
     'tau_m', 'V_rest', 'V_thresh', 'V_reset', 'dt', 'T_sim', 'T_relax', ...
     'update_every_ms', 'gamma_values', 'bell_amp_values');

fprintf('Data saved successfully to ChatGPT_SpikingHopfield_Data.mat.\n');