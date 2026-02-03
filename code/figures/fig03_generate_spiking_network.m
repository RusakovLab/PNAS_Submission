%% FIGURE 3: Spiking Hopfield Network with Dynamic Thresholds
% LIF network with bell-shaped thresholds
% 9-panel figure showing capacity analysis

clear; close all; clc;

%% ========== PARAMETERS ==========
N = 144;                                   % Number of neurons
pattern_counts = [2, 4, 8, 12, 16, 20, 24, 28, 32];
noise_levels   = 0:0.1:1.0;
num_trials     = 10;

% LIF parameters
tau_m    = 10;      % ms
V_rest   = 0;
V_thresh = 1;
V_reset  = 0;
dt       = 0.5;     % ms
T_sim    = 200;     % ms
T_relax  = 50;      % ms
update_every_ms = 5;
update_every   = max(1, round(update_every_ms/dt));

% Threshold parameters
gamma_values    = [-2.0, -1.5, -1.0, -0.5, 0.0];
bell_amp_values = [0.0, 1.0, 2.0, 3.0, 4.0];

%% Initialize storage
fprintf('=== FIGURE 3: SPIKING HOPFIELD NETWORK ===\n');
fprintf('Network size: %d neurons\n', N);
fprintf('Simulating %d conditions...\n', ...
    length(pattern_counts) * length(noise_levels) * length(gamma_values) * length(bell_amp_values));

recall_quality = zeros(length(pattern_counts), length(noise_levels), ...
                       length(gamma_values), length(bell_amp_values));

%% Run simulations
start_time = tic;
for p_idx = 1:length(pattern_counts)
    P = pattern_counts(p_idx);
    fprintf('Patterns: %d/%d (P=%d)\n', p_idx, length(pattern_counts), P);
    
    for n_idx = 1:length(noise_levels)
        noise = noise_levels(n_idx);
        
        for g_idx = 1:length(gamma_values)
            gamma = gamma_values(g_idx);
            
            for b_idx = 1:length(bell_amp_values)
                bell_amp = bell_amp_values(b_idx);
                
                quality_sum = 0;
                for trial = 1:num_trials
                    rng(1000 + 100*p_idx + 10*n_idx + g_idx + b_idx + 10000*trial);
                    
                    % Create patterns
                    patterns = 2 * randi([0, 1], N, P) - 1;
                    
                    % Hebbian weights
                    W = (patterns * patterns') / N;
                    W(1:N+1:end) = 0;  % No self-connections
                    
                    % Test pattern
                    test_idx = randi(P);
                    orig = patterns(:, test_idx);
                    
                    % Add noise
                    noisy = orig;
                    num_flips = round(noise * N);
                    if num_flips > 0
                        flip_idx = randperm(N, num_flips);
                        noisy(flip_idx) = -noisy(flip_idx);
                    end
                    
                    % Bell statistics
                    G_pat = W * patterns;
                    G_opt = mean(G_pat(:));
                    stdG  = std(G_pat(:));
                    if stdG == 0, stdG = eps; end
                    
                    % Run dynamics
                    state = noisy;
                    V = V_rest * ones(N, 1);
                    num_steps = round(T_sim / dt);
                    
                    for t = 1:num_steps
                        % Compute thresholds
                        S = state' * W * state;
                        G = W * state;
                        B = 1 / (2 * stdG^2);
                        phi = exp(-B * (G - G_opt).^2);
                        d = phi - mean(phi);
                        d = d - (d' * state) / N * state;
                        d = bell_amp * d;
                        theta_par = (gamma * S / N) * state;
                        theta = theta_par + d;
                        
                        % Input and relaxation
                        h = W * state;
                        bias_strength = max(0, 1 - t*dt/T_relax);
                        I_input = h - theta + bias_strength * 5 * noisy;
                        
                        % Integrate
                        dV = (-(V - V_rest) + I_input) / tau_m * dt;
                        V = V + dV;
                        
                        % Update state
                        if mod(t, update_every) == 0
                            state = sign(V);
                            state(state == 0) = 1;
                            V = V_rest * ones(N, 1);
                        end
                    end
                    
                    % Quality
                    overlap = sum(state == orig) / N;
                    quality_sum = quality_sum + overlap;
                end
                
                recall_quality(p_idx, n_idx, g_idx, b_idx) = quality_sum / num_trials;
            end
        end
    end
end

fprintf('Completed in %.1f minutes\n', toc(start_time)/60);

%% ========== GENERATE FIGURE 3 ==========
loading_factors = pattern_counts / N;

% Find baseline indices (γ=0, β=0)
baseline_g = find(gamma_values == 0);
baseline_b = find(bell_amp_values == 0);

% Two main configurations to compare
config1_g = baseline_g;  % γ = 0.0
config1_b = baseline_b;  % β = 0.0

config2_g = find(gamma_values == -1.0, 1);  % γ = -1.0
config2_b = find(bell_amp_values == 1.0, 1);  % β = 1.0

fig = figure('Position', [50, 50, 1400, 1800], 'Color', 'w');

%% PANEL A: Recall Quality vs Noise (γ=0, β=0)
subplot(3, 3, 1);
hold on; grid on; box on;
colors_patterns = lines(length(pattern_counts));  % Generate colors for all patterns
for p_idx = 1:length(pattern_counts)
    y = squeeze(recall_quality(p_idx, :, config1_g, config1_b));
    plot(noise_levels, y, '-o', 'LineWidth', 2, 'Color', colors_patterns(p_idx,:), ...
         'DisplayName', sprintf('P=%d (n=%d)', pattern_counts(p_idx), pattern_counts(p_idx)));
end
xlabel('\sigma'); ylabel('Recall Quality');
title(sprintf('(\\gamma=%.2f, \\beta=%.1f)', gamma_values(config1_g), bell_amp_values(config1_b)));
legend('Location', 'best', 'FontSize', 7);
ylim([0, 1.05]);
text(0.05, 0.95, 'A', 'Units', 'normalized', 'FontSize', 16, 'FontWeight', 'bold');

%% PANEL B: Recall Quality vs Loading Factor (γ=0, β=0)
subplot(3, 3, 2);
hold on; grid on; box on;
noise_indices = [1, 3, 5, 7, 9, 11];  % 0.0, 0.2, 0.4, 0.6, 0.8, 1.0
colors_noise = lines(length(noise_indices));  % Colors for noise levels
for i = 1:length(noise_indices)
    n_idx = noise_indices(i);
    y = squeeze(recall_quality(:, n_idx, config1_g, config1_b));
    plot(loading_factors, y, '-o', 'LineWidth', 2, 'Color', colors_noise(i,:), ...
         'DisplayName', sprintf('Noise=%.1f', noise_levels(n_idx)));
end
xlabel('\alpha = P/N'); ylabel('Recall Quality');
title(sprintf('(\\gamma=%.2f, \\beta=%.1f)', gamma_values(config1_g), bell_amp_values(config1_b)));
legend('Location', 'best', 'FontSize', 7);
ylim([0, 1.05]);
text(0.05, 0.95, 'B', 'Units', 'normalized', 'FontSize', 16, 'FontWeight', 'bold');

%% PANEL C: 2D Heatmap (γ=0, β=0)
subplot(3, 3, 3);
heatmap_data = squeeze(recall_quality(:, :, config1_g, config1_b))';
imagesc(loading_factors, noise_levels, heatmap_data);
set(gca, 'YDir', 'normal');
colorbar; colormap(jet); caxis([0, 1]);
xlabel('\sigma'); ylabel('\alpha = P/N');
title(sprintf('\\gamma=%.2f, \\beta=%.1f', gamma_values(config1_g), bell_amp_values(config1_b)));
text(0.05, 0.95, 'C', 'Units', 'normalized', 'FontSize', 16, 'FontWeight', 'bold', 'Color', 'w');

%% PANEL D: Recall Quality vs Noise (γ=-1.0, β=1.0)
subplot(3, 3, 4);
hold on; grid on; box on;
for p_idx = 1:length(pattern_counts)
    y = squeeze(recall_quality(p_idx, :, config2_g, config2_b));
    plot(noise_levels, y, '-o', 'LineWidth', 2, 'Color', colors_patterns(p_idx,:), ...
         'DisplayName', sprintf('P=%d', pattern_counts(p_idx)));
end
xlabel('\sigma'); ylabel('Recall Quality');
title(sprintf('(\\gamma=%.2f, \\beta=%.1f)', gamma_values(config2_g), bell_amp_values(config2_b)));
legend('Location', 'best', 'FontSize', 7);
ylim([0, 1.05]);
text(0.05, 0.95, 'D', 'Units', 'normalized', 'FontSize', 16, 'FontWeight', 'bold');

%% PANEL E: Recall Quality vs Loading Factor (γ=-1.0, β=1.0)
subplot(3, 3, 5);
hold on; grid on; box on;
for i = 1:length(noise_indices)
    n_idx = noise_indices(i);
    y = squeeze(recall_quality(:, n_idx, config2_g, config2_b));
    plot(loading_factors, y, '-o', 'LineWidth', 2, 'Color', colors_noise(i,:), ...
         'DisplayName', sprintf('Noise=%.1f', noise_levels(n_idx)));
end
xlabel('\alpha = P/N'); ylabel('Recall Quality');
title(sprintf('(\\gamma=%.2f, \\beta=%.1f)', gamma_values(config2_g), bell_amp_values(config2_b)));
legend('Location', 'best', 'FontSize', 7);
ylim([0, 1.05]);
text(0.05, 0.95, 'E', 'Units', 'normalized', 'FontSize', 16, 'FontWeight', 'bold');

%% PANEL F: 2D Heatmap (γ=-1.0, β=1.0)
subplot(3, 3, 6);
heatmap_data = squeeze(recall_quality(:, :, config2_g, config2_b))';
imagesc(loading_factors, noise_levels, heatmap_data);
set(gca, 'YDir', 'normal');
colorbar; colormap(jet); caxis([0, 1]);
xlabel('\sigma'); ylabel('\alpha = P/N');
title(sprintf('\\gamma=%.2f, \\beta=%.1f', gamma_values(config2_g), bell_amp_values(config2_b)));
text(0.05, 0.95, 'F', 'Units', 'normalized', 'FontSize', 16, 'FontWeight', 'bold', 'Color', 'w');

%% PANEL G: γ-β Heatmap at low load, no noise
subplot(3, 3, 7);
p_idx_g = 1;  % P=2
n_idx_g = 1;  % noise=0
heatmap_g = squeeze(recall_quality(p_idx_g, n_idx_g, :, :));
imagesc(bell_amp_values, gamma_values, heatmap_g);
set(gca, 'YDir', 'normal');
colorbar; colormap(jet); caxis([0, 1]);
xlabel('Heterogeneity Threshold \beta');
ylabel('Threshold Scaling \gamma');
title(sprintf('\\alpha=%.3f (P=%d), \\sigma=%.1f', ...
    loading_factors(p_idx_g), pattern_counts(p_idx_g), noise_levels(n_idx_g)));
text(0.05, 0.95, 'G', 'Units', 'normalized', 'FontSize', 16, 'FontWeight', 'bold', 'Color', 'w');

%% PANEL H: γ-β Heatmap at high load, no noise
subplot(3, 3, 8);
p_idx_h = 9;  % P=32
n_idx_h = 1;  % noise=0
heatmap_h = squeeze(recall_quality(p_idx_h, n_idx_h, :, :));
imagesc(bell_amp_values, gamma_values, heatmap_h);
set(gca, 'YDir', 'normal');
colorbar; colormap(jet); caxis([0, 1]);
xlabel('Heterogeneity Threshold \beta');
ylabel('Threshold Scaling \gamma');
title(sprintf('\\alpha=%.3f (P=%d), \\sigma=%.1f', ...
    loading_factors(p_idx_h), pattern_counts(p_idx_h), noise_levels(n_idx_h)));
text(0.05, 0.95, 'H', 'Units', 'normalized', 'FontSize', 16, 'FontWeight', 'bold', 'Color', 'w');

%% PANEL I: γ-β Heatmap at high load, high noise
subplot(3, 3, 9);
p_idx_i = 9;  % P=32
n_idx_i = 10; % noise=0.9
heatmap_i = squeeze(recall_quality(p_idx_i, n_idx_i, :, :));
imagesc(bell_amp_values, gamma_values, heatmap_i);
set(gca, 'YDir', 'normal');
colorbar; colormap(jet); caxis([0, 1]);
xlabel('Heterogeneity Threshold \beta');
ylabel('Threshold Scaling \gamma');
title(sprintf('\\alpha=%.3f (P=%d), \\sigma=%.1f', ...
    loading_factors(p_idx_i), pattern_counts(p_idx_i), noise_levels(n_idx_i)));
text(0.05, 0.95, 'I', 'Units', 'normalized', 'FontSize', 16, 'FontWeight', 'bold', 'Color', 'w');

%% Add overall title
sgtitle('Figure 3: Spiking Hopfield Network with Dynamic Thresholds', ...
        'FontSize', 14, 'FontWeight', 'bold');

%% Save figure
saveas(fig, '../../figures_out/main/fig03_spiking_hopfield_network.png');
saveas(fig, '../../figures_out/main/fig03_spiking_hopfield_network.fig');
fprintf('\n=== Figure 3 Generation Complete ===\n');
fprintf('Figures saved to: figures_out/main/\n');
