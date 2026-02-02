%% Fig. 1 (PNAS-ready): gamma/beta threshold modulation and preserved minima
% A) Threshold decomposition (gamma vs beta) for one stored pattern
% B) Input-dependent bell-shaped beta function (zero-mean)
% C) Energy landscape: baseline E1 vs modulated thresholds E2 (stored minima preserved)

clear; clc; close all; rng(1);

%% ---- User knobs ----------------------------------------------------------
gamma    = -1.00;   % global state-aligned modulation
bell_amp = 10.00;   % amplitude of bell-shaped heterogeneity (energy-neutral)
ref_pat  = 1;       % which stored pattern to illustrate in panel A
% --------------------------------------------------------------------------

%% 1) Network definition
network_size = 6;
patterns = [               % columns are patterns in {-1,+1}
    -1  1 -1  1
     1 -1 -1 -1
    -1  1  1 -1
     1  1 -1  1
    -1 -1  1  1
     1 -1 -1  1
];
[N, num_patterns] = size(patterns);
assert(N == network_size, 'patterns must have N rows');

%% 2) Hebbian weights (canonical normalization)
W = zeros(network_size);
for m = 1:num_patterns
    p = patterns(:,m);
    W = W + p*p.';
end
W = W - diag(diag(W));
W = W / network_size;

%% 3) Enumerate all states
num_states = 2^network_size;
all_states = zeros(network_size, num_states);
for i = 0:num_states-1
    bits = dec2bin(i, network_size) - '0';   % row vector 0/1
    all_states(:, i+1) = 2*bits(:) - 1;      % column in {-1,+1}
end

%% 4) Baseline energy E1
E1 = zeros(1,num_states);
for i = 1:num_states
    s = all_states(:,i);
    E1(i) = -0.5 * (s.'*W*s);
end

%% 5) Minima-preserving heterogeneous thresholds and energy E2
G_all  = W * all_states;
stdG   = std(G_all(:));
G_pat  = W * patterns;
G_opt  = mean(G_pat(:)); % bell center around typical pattern drive

get_thresholds = @(s) local_thresholds_minima_preserving(s, W, G_opt, stdG, gamma, bell_amp);

E2 = zeros(1,num_states);
for i = 1:num_states
    s = all_states(:,i);
    th = get_thresholds(s);
    E2(i) = -0.5*(s.'*W*s) + th.'*s;
end

%% 6) Indices of stored patterns
pattern_indices = zeros(1,num_patterns);
for m = 1:num_patterns
    p = patterns(:,m);
    pattern_indices(m) = find(all(all_states==p,1), 1);
end

%% 7) Panel A: threshold decomposition for a reference stored pattern
s_ref = patterns(:, ref_pat);
G_ref = W * s_ref;
S_ref = s_ref.' * W * s_ref;

% Global gamma component
C = (gamma * S_ref / N);
theta_gamma = C * s_ref;

% Local beta component: bell-shaped, zero-mean and orthogonal to s_ref
B = 1 / max(2*stdG^2, eps);
phi  = exp(-B * (G_ref - G_opt).^2);
phi0 = phi - mean(phi);

theta_beta = phi0 - (phi0.'*s_ref)/N * s_ref;  % orthogonal to s_ref
theta_beta = bell_amp * theta_beta;

theta_total = theta_gamma + theta_beta;

fprintf('Fig1 ref pattern %d: |C|=%.3f, sum(beta)=%.2e, (beta^T s)/N=%.2e\n', ...
    ref_pat, abs(C), sum(theta_beta), (theta_beta.'*s_ref)/N);

%% 8) Panel B: bell-shaped function
if stdG > 0
    gmin = G_opt - 4*stdG; gmax = G_opt + 4*stdG;
else
    gmin = min(G_ref)-1; gmax = max(G_ref)+1;
end
ggrid = linspace(gmin, gmax, 400);
phi_grid  = exp(-B * (ggrid - G_opt).^2);
phi0_grid = phi_grid - mean(phi_grid);

uniform_grid = zeros(size(ggrid));

%% 9) Plot: PNAS-style Fig. 1 (clean hierarchy, consistent semantics)
fs_axis  = 11;
fs_label = 12;
fs_title = 12;

lw_main  = 2.4;
lw_mid   = 2.0;
lw_thin  = 1.6;

col_gamma = [0 0 0];        % black
col_beta  = [0.85 0 0];     % red
col_total = [0.45 0 0.75];  % purple
col_E1    = [0 0.25 0.9];   % blue
col_E2    = [0.9 0 0];      % red

fig = figure('Color','w','Position',[80 80 1150 720]);
tiledlayout(2,2,'TileSpacing','compact','Padding','compact');

% -------- Panel A --------
nexttile(1); hold on; grid on; box on;
idx = 1:N;

plot(idx, theta_gamma, '--', 'Color', col_gamma, 'LineWidth', lw_thin, ...
    'DisplayName','Global term (\gamma)');
plot(idx, theta_beta, '-',  'Color', col_beta,  'LineWidth', lw_mid, ...
    'DisplayName','Local term (\beta)');
plot(idx, theta_total,'-',  'Color', col_total, 'LineWidth', lw_main, ...
    'DisplayName','Total threshold');

yline(0, ':', 'Color', [0 0 0], 'HandleVisibility','off');

xlabel('Neuron index','FontSize',fs_label,'FontWeight','bold');
ylabel('Threshold contribution (\theta)','FontSize',fs_label,'FontWeight','bold');
title('A  Threshold decomposition for a stored pattern','FontSize',fs_title);

set(gca,'FontSize',fs_axis,'LineWidth',1.1,'XLim',[0.5 N+0.5]);
ylim([-5.5 5.5]);  % tighter range improves readability

lgA = legend('Location','southwest','FontSize',9);
lgA.Box = 'off';

% -------- Panel B --------
nexttile(2); hold on; grid on; box on;

plot(ggrid, uniform_grid, '--', 'Color', [0 0 0], 'LineWidth', lw_thin, ...
    'DisplayName','Uniform (baseline)');
plot(ggrid, bell_amp*phi0_grid, '-', 'Color', col_beta, 'LineWidth', lw_main, ...
    'DisplayName', sprintf('Bell-shaped, zero-mean (\\times %.1f)', bell_amp));

xline(G_opt, ':', 'Color', [0 0 0], 'LineWidth', 1.2, 'DisplayName','G_{opt}');

xlabel('Synaptic input (G)','FontSize',fs_label,'FontWeight','bold');
ylabel('Local modulation f(G)','FontSize',fs_label,'FontWeight','bold');
title('B  Input-dependent local modulation (β term)','FontSize',fs_title);

set(gca,'FontSize',fs_axis,'LineWidth',1.1);
lgB = legend('Location','northwest','FontSize',9);
lgB.Box = 'off';

% -------- Panel C --------
nexttile([1 2]); hold on; grid on; box on;

plot(1:num_states, E1, '-', 'Color', col_E1, 'LineWidth', lw_mid, 'DisplayName','Baseline energy E_1');
plot(1:num_states, E2, '-', 'Color', col_E2, 'LineWidth', lw_mid, 'DisplayName','Modulated thresholds E_2');

plot(pattern_indices, E1(pattern_indices), 'o', 'Color', col_E1, 'MarkerSize', 8, ...
    'MarkerFaceColor', col_E1, 'DisplayName','Stored patterns (E_1)');
plot(pattern_indices, E2(pattern_indices), 'o', 'Color', col_E2, 'MarkerSize', 8, ...
    'MarkerFaceColor', col_E2, 'DisplayName','Stored patterns (E_2)');

xlabel('State index','FontSize',fs_label,'FontWeight','bold');
ylabel('Energy','FontSize',fs_label,'FontWeight','bold');
title('C  Energy landscape with preserved minima','FontSize',fs_title);

set(gca,'FontSize',fs_axis,'LineWidth',1.1,'XLim',[1 num_states]);

lgC = legend('Location','southeast','FontSize',9);
lgC.Box = 'off';

% Optional figure header (often omitted in PNAS main figures)
% sgtitle(sprintf('\\gamma=%.2f, bell\\_amp=%.1f', gamma, bell_amp), 'FontSize', 12, 'FontWeight','bold');

% Export
outBase = 'Fig1_PNAS_clean';
print(fig, [outBase '.pdf'], '-dpdf', '-painters');
print(fig, [outBase '.png'], '-dpng', '-r600');
fprintf('Saved: %s.pdf and %s.png\n', outBase, outBase);

%% ----------------- helper -----------------
function th = local_thresholds_minima_preserving(s, W, G_opt, stdG, gamma, bell_amp)
    N = numel(s);
    S = s.'*W*s;

    % Global term (gamma)
    theta_gamma = (gamma * S / N) * s;

    % Local term (beta); if degenerate stdG, omit
    if stdG <= 0
        th = theta_gamma;
        return;
    end

    G = W*s;
    B = 1/(2*stdG^2);

    phi = exp(-B*(G - G_opt).^2);

    d = phi - mean(phi);        % zero-mean
    d = d - (d.'*s)/N * s;      % orthogonal to s

    th = theta_gamma + bell_amp * d;
end
