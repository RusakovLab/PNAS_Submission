%% LIF Hopfield Network with Non-Uniform (Bell-Shaped) Thresholds
% Panels A–D + A/B comparison (OLD vs MODIFIED) for γ, β effects
clear; close all; clc;
%% ---------------- Parameters ----------------
N = 144;                                   % Number of neurons
pattern_counts = [2, 4, 8, 12, 16, 20, 24, 28, 32];  % P => alpha=P/N
noise_levels   = 0:0.1:1;
num_trials     = 10;                      % Trials per condition
% LIF neuron parameters (base)
tau_m_base    = 10;      % ms
V_rest        = 0;
V_thresh_base = 1;       % nominal threshold (used in MOD as part of Vthr_eff)
V_reset       = 0;
dt            = 0.5;     % ms
% Timing (OLD vs MOD)
T_sim_OLD       = 200;   T_relax_OLD = 50;  update_every_ms_OLD = 5;  cue_gain_OLD = 5;
T_sim_MOD       = 250;   T_relax_MOD = 30;  update_every_ms_MOD = 2;  cue_gain_MOD = 2;
T_sim_MOD       = 100;   
% Adaptive threshold parameter grids (γ: parallel; β: heterogeneity amplitude)
gamma_values    = [0, 0.6, 1.0, 1.4, 2.0];         % include baseline 0 and >0 range
bell_amp_values = [0, 0.2, 0.5, 0.8, 1.2];         % include baseline 0 and >0 range
% MOD-only gains for stronger γ/β effects
k_par   = 2.0;     % gain on gamma (parallel component)
k_het   = 1.5;     % gain on beta (heterogeneity component)
k_width = 0.8;     % bell width multiplier (smaller => narrower, stronger)
% Decision nonlinearity (MOD)
beta_dec = 6;      % logistic decision sharpness around Vthr_eff (2–10 recommended)
%% ------------- Storage for results -------------
dims = [length(pattern_counts), length(noise_levels), length(gamma_values), length(bell_amp_values)];
recall_quality_OLD = zeros(dims);
recall_quality_MOD = zeros(dims);
fprintf('Running simulations on (gamma, beta) grid for BOTH variants (OLD & MOD)...\n');
%% ----------------- Helper: weight builder -----------------
buildW = @(patterns) (patterns*patterns.')/size(patterns,2);
%% ----------------- Core simulator as a function -----------------
function Q = run_variant(variant, N, P, noise, num_trials, ...
                         gamma_values, bell_amp_values, ...
                         V_rest, V_reset, V_thresh_base, tau_m_base, dt, ...
                         T_sim, T_relax, update_every_ms, cue_gain, ...
                         k_par, k_het, k_width, beta_dec, buildW)
    num_steps      = round(T_sim / dt);
    update_every   = max(1, round(update_every_ms/dt));
    Q = zeros(length(gamma_values), length(bell_amp_values)); % averaged over trials
    for g_idx = 1:length(gamma_values)
        gamma = gamma_values(g_idx);
        for b_idx = 1:length(bell_amp_values)
            bell_amp = bell_amp_values(b_idx);
            quality_sum = 0;
            for trial = 1:num_trials
                rng(1000 + 100*P + 10*round(noise*10) + g_idx + b_idx + 10000*trial);
                % -- Patterns and weights --
                patterns = 2 * randi([0, 1], N, P) - 1;  % {-1, +1}
                W = buildW(patterns);                    % Hebbian
                W(1:N+1:end) = 0;                        % no self
                % -- Choose pattern and corrupt it --
                test_idx = randi(P);
                orig = patterns(:, test_idx);
                noisy = orig;
                num_flips = round(noise * N);
                if num_flips > 0
                    flip_idx = randperm(N, num_flips);
                    noisy(flip_idx) = -noisy(flip_idx);
                end
                % -- Precompute pattern-based stats (used by OLD) --
                if strcmp(variant,'OLD')
                    G_pat  = W * patterns;                 % N x P
                    G_opt  = mean(G_pat(:));
                    stdG   = std(G_pat(:)); if stdG==0, stdG = eps; end
                end
                % -- Initialize dynamics --
                state = noisy;                          % current ±1 state
                V     = V_rest * ones(N, 1);           % membrane potential
                tau_m = tau_m_base;                    % (same both variants)
                for t = 1:num_steps
                    % ---------- Build theta ----------
                    switch variant
                        case 'OLD'
                            % OLD: theta_par from energy; bell from pattern stats
                            S  = state.' * W * state;
                            theta_par = (gamma * S / N) * state;
                            G   = W * state;
                            B   = 1 / (2 * (stdG^2));
                            phi = exp(-B * (G - G_opt).^2);
                            d   = phi - mean(phi);
                            d   = d - (d.'*state)/N * state;
                            d   = bell_amp * d;
                            theta = theta_par + d;
                            theta = theta - mean(theta);   % keep ⟨Vth⟩ fixed
                            % OLD input current; strong cue; V reset each update
                            h = W * state;
                            bias_strength = max(0, 1 - t*dt/T_relax);
                            I_input = (h - theta) + bias_strength * cue_gain * noisy;
                            % integrate
                            dV = (-(V - V_rest) + I_input) / tau_m * dt;
                            V  = V + dV;
                            if mod(t, update_every) == 0
                                state = sign(V);
                                state(state == 0) = 1;
                                V = V_rest * ones(N,1);  % HARD RESET (OLD)
                            end
                        case 'MOD'
                            % MOD: field-scaled & current-state-driven θ
                            h = W * state;                 % local field
                            sig_h = std(h); if sig_h==0, sig_h = eps; end
                            % parallel (gamma), scaled
                            S  = state.' * W * state;      % scalar
                            theta_par = (gamma * S / N) * state;
                            theta_par = k_par * theta_par;
                            % bell (beta) from CURRENT field stats
                            G    = h;
                            G_mu = mean(G);
                            G_sd = std(G); if G_sd==0, G_sd=eps; end
                            B    = 1 / (2 * (k_width * G_sd)^2);
                            phi  = exp(-B * (G - G_mu).^2);
                            d    = phi - mean(phi);                     % zero-mean
                            d    = d - (d.'*state)/N * state;           % ⟂ to state
                            d    = (d/(std(d)+eps));                     % normalize
                            d    = k_het * sig_h * d;                    % scale to field
                            d    = bell_amp * d;
                            theta = theta_par + d;
                            theta = theta - mean(theta);                 % keep ⟨Vth⟩
                            % dual channel: input subtraction + effective threshold
                            bias_strength = max(0, 1 - t*dt/T_relax);
                            I_input = (h - theta) + bias_strength * cue_gain * noisy;
                            % integrate (NO hard reset)
                            dV = (-(V - V_rest) + I_input) / tau_m * dt;
                            V  = V + dV;
                            if mod(t, update_every) == 0
                                Vthr_eff = V_thresh_base + theta;       % dynamic thr
                                p_up   = 1 ./ (1 + exp(-beta_dec*(V - Vthr_eff)));
                                state  = 2*(p_up >= 0.5) - 1;
                                % gentle leak instead of wipe (preserve θ history)
                                V = 0.3*V + 0.7*V_rest;
                            end
                    end % switch
                end % t loop
                overlap = sum(state == orig) / N;
                quality_sum = quality_sum + overlap;
            end % trial
            Q(g_idx, b_idx) = quality_sum / num_trials;
        end % beta
    end % gamma
end % function
%% ----------------- Run OLD & MOD over the full grid -----------------
for p_idx = 1:length(pattern_counts)
    P = pattern_counts(p_idx);
    alpha = P / N;
    fprintf('P=%d (alpha=%.3f)\n', P, alpha);
    for n_idx = 1:length(noise_levels)
        noise = noise_levels(n_idx);
        % OLD
        Q_old = run_variant('OLD', N, P, noise, num_trials, ...
                            gamma_values, bell_amp_values, ...
                            V_rest, V_reset, V_thresh_base, tau_m_base, dt, ...
                            T_sim_OLD, T_relax_OLD, update_every_ms_OLD, cue_gain_OLD, ...
                            k_par, k_het, k_width, beta_dec, buildW);
        recall_quality_OLD(p_idx, n_idx, :, :) = Q_old;
        % MOD
        Q_mod = run_variant('MOD', N, P, noise, num_trials, ...
                            gamma_values, bell_amp_values, ...
                            V_rest, V_reset, V_thresh_base, tau_m_base, dt, ...
                            T_sim_MOD, T_relax_MOD, update_every_ms_MOD, cue_gain_MOD, ...
                            k_par, k_het, k_width, beta_dec, buildW);
        recall_quality_MOD(p_idx, n_idx, :, :) = Q_mod;
    end
end
fprintf('Simulations complete for both variants.\n');
%% ----------------- Derived metrics (MOD shown in Panels A–D) -----------------
loading_factors = pattern_counts / N;
% Baselines (γ=0, β=0)
g0 = find(gamma_values == 0, 1);
b0 = find(bell_amp_values == 0, 1);
Q_base_OLD = squeeze(recall_quality_OLD(:, :, g0, b0));
Q_base_MOD = squeeze(recall_quality_MOD(:, :, g0, b0));
% Positive γ,β indices
pos_g_idx = find(gamma_values > 0);
pos_b_idx = find(bell_amp_values > 0);
% Best over γ>0,β>0 (MOD)
Q_best_pos_MOD  = squeeze(max(max(recall_quality_MOD(:, :, pos_g_idx, pos_b_idx), [], 4), [], 3));
DeltaQ_best_MOD = Q_best_pos_MOD - Q_base_MOD;
% Best over γ>0,β>0 (OLD) to get its boundary for overlay comparison
Q_best_pos_OLD  = squeeze(max(max(recall_quality_OLD(:, :, pos_g_idx, pos_b_idx), [], 4), [], 3));
DeltaQ_best_OLD = Q_best_pos_OLD - Q_base_OLD;
% Argmax winners (MOD)
gamma_star_idx = zeros(size(Q_base_MOD));
beta_star_idx  = zeros(size(Q_base_MOD));
for p = 1:length(pattern_counts)
    for n = 1:length(noise_levels)
        S = squeeze(recall_quality_MOD(p, n, pos_g_idx, pos_b_idx));
        [mx, linidx] = max(S(:));
        [ig, ib] = ind2sub(size(S), linidx);
        gamma_star_idx(p,n) = pos_g_idx(ig);
        beta_star_idx(p,n)  = pos_b_idx(ib);
    end
end
gamma_star = gamma_values(gamma_star_idx);
beta_star  = bell_amp_values(beta_star_idx);
% Masks (MOD)
mask_no_improve_MOD = DeltaQ_best_MOD <= 0;
gamma_star_masked   = gamma_star;  gamma_star_masked(mask_no_improve_MOD) = NaN;
beta_star_masked    = beta_star;   beta_star_masked(mask_no_improve_MOD)  = NaN;
%% ----------------- Panels A & B (MOD vs OLD Comparison) -----------------
% Choose α ≈ 0.19
[~, p_idx_alpha] = min(abs(pattern_counts/N - 0.19));
alpha_val = pattern_counts(p_idx_alpha)/N;

% --- Modified Panel A: Q vs noise for γ (β fixed) - MOD vs OLD comparison ---
gamma_comp_vals  = [0, 2.0];
g_comp_indices   = arrayfun(@(v) find(gamma_values == v, 1), gamma_comp_vals);
beta_fixed_val   = 0.5;
[~, bell_idx_fixed] = min(abs(bell_amp_values - beta_fixed_val));
beta_fixed_val = bell_amp_values(bell_idx_fixed);

figure('Name','Panel A (Q vs Noise for gamma, MOD vs OLD)','Position',[80 80 700 520]); hold on; grid on; box on;
cols = [0 0 0; 0 0 1]; % Black for OLD, Blue for MOD

% Iterate over the selected gamma values (0 and 2.0)
for i = 1:length(gamma_comp_vals)
    g_idx = g_comp_indices(i);
    gamma_val = gamma_values(g_idx);
    
    % Plot OLD: Dashed line, Black color
    mu_old = squeeze(recall_quality_OLD(p_idx_alpha, :, g_idx, bell_idx_fixed));
    plot(noise_levels, mu_old, '--', 'LineWidth', 1.5, 'MarkerSize', 5, 'Marker','o', ...
        'Color', cols(1,:), 'DisplayName', sprintf('OLD \\gamma=%.1f', gamma_val));

    % Plot MOD: Solid line, Blue color
    mu_mod = squeeze(recall_quality_MOD(p_idx_alpha, :, g_idx, bell_idx_fixed));
    plot(noise_levels, mu_mod, '-', 'LineWidth', 2.5, 'MarkerSize', 6, 'Marker','s', ...
        'Color', cols(2,:), 'DisplayName', sprintf('MOD \\gamma=%.1f', gamma_val));
end

xlabel('Noise \sigma'); ylabel('Recall quality Q');
title(sprintf('Panel A: Q vs Noise (\\alpha=%.3f, \\beta=%.1f fixed) [MOD vs OLD]', alpha_val, beta_fixed_val), 'Interpreter', 'tex');
legend('Location','southwest', 'Interpreter', 'tex'); ylim([0,1]); xlim([0,1]); hold off;

% --- Modified Panel B: Q vs noise for β (γ fixed) - MOD vs OLD comparison ---
bell_comp_vals  = [0, 1.2];
b_comp_indices  = arrayfun(@(v) find(bell_amp_values == v, 1), bell_comp_vals);
gamma_fixed_val   = 1.0;
[~, gamma_idx_fixed] = min(abs(gamma_values - gamma_fixed_val));
gamma_fixed_val = gamma_values(gamma_idx_fixed);

figure('Name','Panel B (Q vs Noise for beta, MOD vs OLD)','Position',[820 80 700 520]); hold on; grid on; box on;
cols = [0 0 0; 1 0 0]; % Black for OLD, Red for MOD

% Iterate over the selected beta values (0 and 1.2)
for i = 1:length(bell_comp_vals)
    b_idx = b_comp_indices(i);
    beta_val = bell_amp_values(b_idx);
    
    % Plot OLD: Dashed line, Black color
    mu_old = squeeze(recall_quality_OLD(p_idx_alpha, :, gamma_idx_fixed, b_idx));
    plot(noise_levels, mu_old, '--', 'LineWidth', 1.5, 'MarkerSize', 5, 'Marker','o', ...
        'Color', cols(1,:), 'DisplayName', sprintf('OLD \\beta=%.1f', beta_val));

    % Plot MOD: Solid line, Red color
    mu_mod = squeeze(recall_quality_MOD(p_idx_alpha, :, gamma_idx_fixed, b_idx));
    plot(noise_levels, mu_mod, '-', 'LineWidth', 2.5, 'MarkerSize', 6, 'Marker','s', ...
        'Color', cols(2,:), 'DisplayName', sprintf('MOD \\beta=%.1f', beta_val));
end

xlabel('Noise \sigma'); ylabel('Recall quality Q');
title(sprintf('Panel B: Q vs Noise (\\alpha=%.3f, \\gamma=%.1f fixed) [MOD vs OLD]', alpha_val, gamma_fixed_val), 'Interpreter', 'tex');
legend('Location','southwest', 'Interpreter', 'tex'); ylim([0,1]); xlim([0,1]); hold off;

%% -------- Panel C: ΔQ_best heatmap (MOD) with TeX-safe annotations --------
figure('Name','Panel C (MOD)','Position',[80 640 700 560]);
imagesc(noise_levels, loading_factors, DeltaQ_best_MOD);
set(gca,'YDir','normal'); 
colormap(parula);
hcb = colorbar;          % use handle for robust tick edits if needed
caxis([min(0,min(DeltaQ_best_MOD(:))), max(0,max(DeltaQ_best_MOD(:)))]);
xlabel('Noise \sigma','Interpreter','tex'); 
ylabel('Loading factor \alpha (=P/N)','Interpreter','tex');
tstr = '\\Delta Q_{\\rm max+}^{\\rm MOD}(\\alpha,\\sigma) = \\max_{\\gamma>0,\\beta>0} Q - Q_{\\rm baseline}';
title(tstr, 'Interpreter','tex');
hold on;
% MOD ΔQ=0 contour (solid black)
[~, hCmod] = contour(noise_levels, loading_factors, DeltaQ_best_MOD, [0 0], ...
    'LineColor','k', 'LineWidth',2);
% ---------- Peak improvement (MOD) and label ----------
[maxDeltaQ, linidx] = max(DeltaQ_best_MOD(:));
[p_star, n_star]    = ind2sub(size(DeltaQ_best_MOD), linidx);
alpha_star          = loading_factors(p_star);
sigma_star          = noise_levels(n_star);
gamma_star_peak     = gamma_star(p_star, n_star);
beta_star_peak      = beta_star(p_star, n_star);
plot(sigma_star, alpha_star, 'kp', 'MarkerSize', 12, 'MarkerFaceColor', 'k');
% TeX-safe multiline label: use \Delta Q, \newline (not \n)
peakLbl = sprintf('  peak \\Delta Q=%.2f\\newline  \\gamma^*=%.2g, \\beta^*=%.2g', ...
                  maxDeltaQ, gamma_star_peak, beta_star_peak);
text(sigma_star, alpha_star, peakLbl, ...
     'Color','k','FontSize',9,'FontWeight','bold', ...
     'VerticalAlignment','bottom','Interpreter','tex');
% ---------- Region stats ----------
improve_mask_MOD             = DeltaQ_best_MOD > 0;
improve_fraction_percent_MOD = 100 * nnz(improve_mask_MOD) / numel(DeltaQ_best_MOD);
mean_improve_in_region_MOD   = mean(DeltaQ_best_MOD(improve_mask_MOD), 'omitnan');
% Row-wise noise threshold (σ0) for MOD using linear interpolation of first sign change
sigma0_by_alpha_MOD = nan(length(loading_factors),1);
for p = 1:length(loading_factors)
    v = DeltaQ_best_MOD(p,:);
    sprod = sign(v(1:end-1)).*sign(v(2:end));
    idx = find(sprod <= 0, 1, 'first');  % first crossing or touch
    if ~isempty(idx) && idx < length(noise_levels)
        x1 = noise_levels(idx);   x2 = noise_levels(idx+1);
        y1 = v(idx);              y2 = v(idx+1);
        if y2 ~= y1
            sigma0_by_alpha_MOD(p) = x1 - y1 * (x2 - x1) / (y2 - y1);
        else
            sigma0_by_alpha_MOD(p) = x1; % exact zero at grid
        end
    end
end
q = prctile(sigma0_by_alpha_MOD(~isnan(sigma0_by_alpha_MOD)), [25 50 75]);
sigma_q1_MOD = q(1);  sigma_med_MOD = q(2);  sigma_q3_MOD = q(3);
% IQR band (light gray) and median σ0 (dashed)
yl = [min(loading_factors) max(loading_factors)];
patch([sigma_q1_MOD sigma_q3_MOD sigma_q3_MOD sigma_q1_MOD], ...
      [yl(1) yl(1) yl(2) yl(2)], ...
      [0 0 0], 'FaceAlpha', 0.06, 'EdgeColor', 'none');
plot([sigma_med_MOD sigma_med_MOD], yl, 'k--', 'LineWidth', 1.5);
text(sigma_med_MOD, yl(2), '  median \sigma_0', ...
     'FontSize',9, 'FontWeight','bold', 'VerticalAlignment','top', ...
     'Color','k', 'Interpreter','tex');
% ---------- Overlay OLD ΔQ=0 boundary (dotted dark gray) ----------
[~, hCold] = contour(noise_levels, loading_factors, DeltaQ_best_OLD, [0 0], ...
                     'LineColor',[0.25 0.25 0.25], 'LineStyle',':', 'LineWidth', 2.0);
uistack(hCold,'bottom');
% Legend (use dummy objects to show styles cleanly)
hPeak = plot(nan,nan,'kp','MarkerSize',12,'MarkerFaceColor','k');
hBand = patch(nan(1,4), nan(1,4), [0 0 0], 'FaceAlpha', 0.06, 'EdgeColor', 'none');
hMed  = plot(nan,nan,'k--','LineWidth',1.5);
legend([hCmod, hCold, hPeak, hBand, hMed], ...
       {'MOD \Delta Q = 0','OLD \Delta Q = 0','Peak','IQR band','Median \sigma_0'}, ...
       'Location','southoutside', 'Interpreter', 'tex');
% ---------- TeX-safe stats textbox ----------
txt = sprintf(['Improved region (MOD): %.1f%%%% of grid\n' ...
               'Mean ΔQ |_{ΔQ>0} (MOD) = %.2f\n' ...
               'Peak (MOD): ΔQ=%.2f at (α=%.3f, σ=%.2f)\n' ...
               '(γ^*=%.2g, β^*=%.2g)\n' ...
               'Boundary σ_0 (MOD): median=%.2f, IQR=[%.2f, %.2f]'], ...
               improve_fraction_percent_MOD, mean_improve_in_region_MOD, ...
               maxDeltaQ, alpha_star, sigma_star, ...
               gamma_star_peak, beta_star_peak, ...
               sigma_med_MOD, sigma_q1_MOD, sigma_q3_MOD);
% Convert any newline chars to TeX \newline explicitly
txt_tex = strrep(txt, newline, '\newline');
annotation('textbox',[0.15 0.78 0.35 0.18], ...
           'String', txt_tex, 'FitBoxToText','on', ...
           'Interpreter','tex','FontSize',9, ...
           'BackgroundColor',[1 1 1 0.7]);
hold off;
%% ----------------- Panel D: Winners map (MOD, masked) -----------------
% -------- Panel D: Winners map (MOD, masked) with TeX-safe titles --------
figure('Name','Panel D (MOD)','Position',[820 640 920 560]);
subplot(1,2,1);
imagesc(noise_levels, loading_factors, gamma_star_masked);
set(gca,'YDir','normal');
hcb1 = colorbar;
xlabel('Noise \sigma','Interpreter','tex'); 
ylabel('\alpha','Interpreter','tex');
title('\gamma^* maximizing Q (masked where \Delta Q \leq 0)', 'Interpreter','tex');
set(hcb1,'Ticks',gamma_values,'TickLabels',string(gamma_values));
subplot(1,2,2);
imagesc(noise_levels, loading_factors, beta_star_masked);
set(gca,'YDir','normal');
hcb2 = colorbar;
xlabel('Noise \sigma','Interpreter','tex'); 
ylabel('\alpha','Interpreter','tex');
title('\beta^* maximizing Q (masked where \Delta Q \leq 0)', 'Interpreter','tex');
set(hcb2,'Ticks',bell_amp_values,'TickLabels',string(bell_amp_values));
colormap(jet);
%% ----------------- Console summaries -----------------
fprintf('\n=== Panel C summaries ===\n');
fprintf('MOD improved region: %.1f%%%% of (alpha, sigma) grid\n', improve_fraction_percent_MOD);
fprintf('MOD mean ΔQ within improved region: %.3f\n', mean_improve_in_region_MOD);
fprintf('MOD peak ΔQ = %.3f at alpha=%.3f (P≈%d), sigma=%.2f, gamma*=%g, beta*=%g\n', ...
        maxDeltaQ, alpha_star, round(alpha_star*N), sigma_star, gamma_star_peak, beta_star_peak);
% OLD vs MOD median boundary comparison
% (compute OLD median σ0 for completeness)
sigma0_by_alpha_OLD = nan(length(loading_factors),1);
for p = 1:length(loading_factors)
    v = DeltaQ_best_OLD(p,:);
    s = sign(v); sc = s(1:end-1).*s(2:end);
    idx = find(sc <= 0, 1, 'first');
    if ~isempty(idx) && idx < length(noise_levels)
        x1 = noise_levels(idx); x2 = noise_levels(idx+1);
        y1 = v(idx);            y2 = v(idx+1);
        if y2 ~= y1
            sigma0_by_alpha_OLD(p) = x1 - y1 * (x2 - x1) / (y2 - y1);
        else
            sigma0_by_alpha_OLD(p) = x1;
        end
    end
end
qOLD = prctile(sigma0_by_alpha_OLD(~isnan(sigma0_by_alpha_OLD)), [25 50 75]);
fprintf('Boundary sigma0 (OLD): median=%.3f, IQR=[%.3f, %.3f]\n', qOLD(2), qOLD(1), qOLD(3));
fprintf('Shift of median boundary (OLD→MOD): %.3f toward lower noise (positive means leftward)\n', qOLD(2)-sigma_med_MOD);
%% ----------------- Save all results -----------------
panelC_stats_MOD = struct();
panelC_stats_MOD.improve_fraction_percent = improve_fraction_percent_MOD;
panelC_stats_MOD.mean_improve_in_region   = mean_improve_in_region_MOD;
panelC_stats_MOD.peak = struct('DeltaQ',maxDeltaQ,'alpha',alpha_star,'sigma',sigma_star, ...
                               'P',round(alpha_star*N), ...
                               'gamma_star',gamma_star_peak,'beta_star',beta_star_peak);
panelC_stats_MOD.boundary = struct('median_sigma0',sigma_med_MOD,'q1',sigma_q1_MOD,'q3',sigma_q3_MOD, ...
                                   'sigma0_by_alpha',sigma0_by_alpha_MOD);
save('ChatGPT_SpikingHopfield_Data.mat', ...
     'recall_quality_OLD','recall_quality_MOD','N','pattern_counts','noise_levels','num_trials', ...
     'tau_m_base','V_rest','V_thresh_base','V_reset','dt', ...
     'T_sim_OLD','T_relax_OLD','update_every_ms_OLD','cue_gain_OLD', ...
     'T_sim_MOD','T_relax_MOD','update_every_ms_MOD','cue_gain_MOD', ...
     'gamma_values','bell_amp_values', ...
     'DeltaQ_best_MOD','DeltaQ_best_OLD','Q_base_MOD','Q_base_OLD', ...
     'gamma_star','beta_star','panelC_stats_MOD', ...
     'k_par','k_het','k_width','beta_dec');
fprintf('Saved: ChatGPT_SpikingHopfield_Data.mat\n');