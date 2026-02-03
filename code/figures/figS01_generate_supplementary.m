%% Fig. S1 — Heterogeneous literature estimates with trend + uncertainty (SI-ready)
% Data synthesized from: Desai (1999), van Welie (2004), Soriano (2006),
% Kolomiets (2009), Howard (2014).
%
% Design goals for SI:
%  - Show heterogeneity transparently, but avoid visually misleading CI explosions.
%  - Use STUDY-BLOCK bootstrap (not point-wise) to respect inter-study structure.
%  - Display median fit + 50% and 95% bands, with a conservative DISPLAY CAP
%    to prevent extrapolative peaks unsupported by data.
%
% Requires Optimization Toolbox for lsqcurvefit.

clear; clc; close all;

%% ========== USER SETTINGS ==========
showZones  = false;   % SI: optional; keep OFF unless you justify boundaries in text
showLOSO   = true;    % print LOSO peak stability (recommended for SI)
nBoot      = 3000;    % bootstrap iterations (SI can be heavier)
rng(1);               % reproducibility

alpha95 = 0.05;       % 95% band
alpha50 = 0.50;       % 50% band (IQR-like)
% ================================

%% 1) Data by study
D_Desai    = [0.20, 1.70];  % Desai et al. (1999)
D_Howard   = [0.70, 0.83];  % Howard et al. (2014)
D_vanWelie = [3.00, 0.60];  % van Welie et al. (2004)

D_Soriano = [ ...
    0.5, 2.0;
    1.0, 3.5;
    2.0, 1.4
];

D_Kolomiets = [ ...
    0.2, 1.4;
    1.0, 1.8;
    2.0, 1.7;
    6.0, 1.2;
    20.0, 0.7
];

studies = struct( ...
    'name', {'Desai1999','Howard2014','vanWelie2004','Soriano2006','Kolomiets2009'}, ...
    'D',    {D_Desai,    D_Howard,    D_vanWelie,    D_Soriano,    D_Kolomiets} ...
);

% Concatenate for the main (non-bootstrap) fit
G = []; E = []; Study = {};
for k = 1:numel(studies)
    Dk = studies(k).D;
    nk = size(Dk,1);
    G = [G; Dk(:,1)]; %#ok<AGROW>
    E = [E; Dk(:,2)]; %#ok<AGROW>
    Study = [Study; repmat({studies(k).name}, nk, 1)]; %#ok<AGROW>
end

%% 2) Log-Gaussian fit
% E = A * exp(- (ln(G)-mu)^2 / (2*sigma^2)) + C
gauss_model = @(p, x) p(1) .* exp(-((log(x) - p(2)).^2) ./ (2*p(3)^2)) + p(4);

% Initial guesses and bounds (keep modest to reduce pathological peaks)
p0 = [1.5, 0.0, 1.0, 0.7];      % [A, mu, sigma, C]
lb = [0.0, -5,  0.10, 0.0];
ub = [6.0,  5,  3.00, 3.0];

options = optimset('Display','off');

[p_fit, ~, residual] = lsqcurvefit(gauss_model, p0, G, E, lb, ub, options);

SS_tot = sum((E - mean(E)).^2);
SS_res = sum(residual.^2);
R2 = 1 - SS_res / SS_tot;
fprintf('Main fit R^2 = %.2f\n', R2);

%% 3) Smooth grid for plotting
g_smooth = logspace(log10(min(G)*0.8), log10(max(G)*1.2), 250);
e_fit = gauss_model(p_fit, g_smooth);

%% 4) Study-block bootstrap (resample studies with replacement; keep within-study points)
% Rationale:
%  - In literature compilation, between-study variability dominates.
%  - Block bootstrap reduces overconfident or pathological parameter excursions caused by point-wise resampling.
%
% Still, with very few studies, some resamples can yield extreme peaks.
% We therefore apply a conservative DISPLAY CAP for the band (SI plotting only),
% based on observed data range.

e_boot = nan(nBoot, numel(g_smooth));
mu_boot = nan(nBoot,1);

for b = 1:nBoot
    % sample studies with replacement
    idxStudies = randi(numel(studies), numel(studies), 1);

    Gb = []; Eb = [];
    for j = 1:numel(idxStudies)
        Dj = studies(idxStudies(j)).D;
        Gb = [Gb; Dj(:,1)]; %#ok<AGROW>
        Eb = [Eb; Dj(:,2)]; %#ok<AGROW>
    end

    try
        pb = lsqcurvefit(gauss_model, p_fit, Gb, Eb, lb, ub, options);
        e_boot(b,:) = gauss_model(pb, g_smooth);
        mu_boot(b)  = pb(2);
    catch
        % leave NaNs
    end
end

% Remove failed fits
ok = all(isfinite(e_boot),2);
e_boot = e_boot(ok,:);
mu_boot = mu_boot(ok);

fprintf('Bootstrap fits kept: %d / %d\n', size(e_boot,1), nBoot);

%% 5) Quantile bands (median + 50% + 95%)
q = @(p) prctile(e_boot, p, 1);

e_med = q(50);

lo95 = q(100*(alpha95/2));
hi95 = q(100*(1-alpha95/2));

lo50 = q(100*(alpha50/2));     % 25th
hi50 = q(100*(1-alpha50/2));   % 75th

%% 6) DISPLAY CAP to prevent misleading excursions
% We cap only what is plotted (does NOT change stats, printed outputs, or fit).
% Choose a conservative cap tied to observed E range.
Emax_obs = max(E);
cap_hi = Emax_obs * 1.35;   % adjust if you prefer 1.25–1.50
cap_lo = 0;                 % excitability cannot be negative here

clip = @(x) min(max(x, cap_lo), cap_hi);

lo95p = clip(lo95); hi95p = clip(hi95);
lo50p = clip(lo50); hi50p = clip(hi50);
e_medp = clip(e_med);

%% 7) Optional LOSO peak stability (in ln-space)
if showLOSO
    fprintf('\nLOSO peak stability (mu in ln(G), G_peak = exp(mu)):\n');
    for k = 1:numel(studies)
        omit = studies(k).name;
        keep = ~strcmp(Study, omit);
        Gk = G(keep); Ek = E(keep);
        try
            pk = lsqcurvefit(gauss_model, p_fit, Gk, Ek, lb, ub, options);
            fprintf('  Omit %-12s  mu = %+6.3f   G_peak = %6.3f\n', omit, pk(2), exp(pk(2)));
        catch
            fprintf('  Omit %-12s  fit failed\n', omit);
        end
    end

    if ~isempty(mu_boot)
        fprintf('\nBootstrap peak summary (study-block):\n');
        fprintf('  median mu = %+6.3f  (G_peak = %6.3f)\n', median(mu_boot), exp(median(mu_boot)));
        fprintf('  2.5–97.5%% mu = [%+6.3f, %+6.3f]\n', prctile(mu_boot,2.5), prctile(mu_boot,97.5));
    end
end

%% 8) Plot (SI framing)
fig = figure('Color','w','Position',[100 100 900 600]);
hold on;

% Optional regimes (avoid mechanistic labels; keep OFF unless justified)
if showZones
    ylim_max = cap_hi * 1.05;
    x1 = 0.30; x2 = 1.50;

    patch([min(g_smooth) x1 x1 min(g_smooth)], [0 0 ylim_max ylim_max], [0.93 0.93 0.93], ...
        'EdgeColor','none','FaceAlpha',0.25);
    patch([x1 x2 x2 x1], [0 0 ylim_max ylim_max], [0.90 0.96 0.90], ...
        'EdgeColor','none','FaceAlpha',0.22);
    patch([x2 max(g_smooth) max(g_smooth) x2], [0 0 ylim_max ylim_max], [0.98 0.90 0.90], ...
        'EdgeColor','none','FaceAlpha',0.18);

    text(min(g_smooth)*1.05, ylim_max*0.98, '\bf Low-drive', 'FontSize', 10);
    text(x1*1.02,          ylim_max*0.98, '\bf Intermediate', 'FontSize', 10);
    text(x2*1.02,          ylim_max*0.98, '\bf High-drive', 'FontSize', 10);
end

% 95% band (light)
h95 = fill([g_smooth, fliplr(g_smooth)], [lo95p, fliplr(hi95p)], [0.88 0.88 0.88], ...
    'EdgeColor','none','FaceAlpha',0.40, 'DisplayName','Bootstrap 95% band (display-capped)');

% 50% band (darker)
h50 = fill([g_smooth, fliplr(g_smooth)], [lo50p, fliplr(hi50p)], [0.75 0.75 0.75], ...
    'EdgeColor','none','FaceAlpha',0.45, 'DisplayName','Bootstrap 50% band');

% Median + original fit
hmed = semilogx(g_smooth, e_medp, 'k-', 'LineWidth', 2.2, 'DisplayName','Bootstrap median fit');
hfit = semilogx(g_smooth, e_fit,  'k--','LineWidth', 1.5, 'DisplayName','Fit to pooled points');

% Points by study (shape-coded; minimal color reliance)
ms = 9;
plotStudy = @(D, mk, name) semilogx(D(:,1), D(:,2), mk, ...
    'MarkerSize', ms, 'LineWidth', 2, 'MarkerFaceColor','w', 'DisplayName', name);

h1 = plotStudy(D_Desai,    'o', 'Desai et al. (1999)');
h2 = plotStudy(D_Howard,   's', 'Howard et al. (2014)');
h3 = plotStudy(D_vanWelie, 'd', 'van Welie et al. (2004)');
h4 = semilogx(D_Soriano(:,1),   D_Soriano(:,2),   'x', 'MarkerSize', ms+2, 'LineWidth', 2.5, 'DisplayName', 'Soriano et al. (2006)');
h5 = semilogx(D_Kolomiets(:,1), D_Kolomiets(:,2), '+', 'MarkerSize', ms+2, 'LineWidth', 2.5, 'DisplayName', 'Kolomiets et al. (2009)');

% Baseline E=1 (define in caption; keep line here)
yline(1, 'k--', 'LineWidth', 1, 'HandleVisibility','off');

% Axes and layout
set(gca,'XScale','log','FontSize',12,'LineWidth',1.2);
xlim([min(g_smooth) max(g_smooth)]);
ylim([0 cap_hi*1.05]);

xlabel('Normalized synaptic conductance (G_{syn})','FontSize',14,'FontWeight','bold');
ylabel('Relative intrinsic excitability (E)','FontSize',14,'FontWeight','bold');
title('Heterogeneous literature estimates with non-monotonic trend (SI)','FontSize',15);

grid on;

% Legend (no R^2 here)
legend([h95, h50, hmed, hfit, h1, h2, h3, h4, h5], 'Location','NorthEast', 'FontSize',9);

hold off;

%% Export to supplementary folder
saveas(fig, '../../figures_out/supplementary/figS01_literature_invertedU.png');
saveas(fig, '../../figures_out/supplementary/figS01_literature_invertedU.fig');
print(fig, '../../figures_out/supplementary/figS01_literature_invertedU.pdf', '-dpdf', '-painters');

fprintf('\n=== Supplementary Figure 1 Generation Complete ===\n');
fprintf('Figures saved to: figures_out/supplementary/\n');