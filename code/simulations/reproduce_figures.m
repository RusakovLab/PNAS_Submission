function reproduce_figures()
% REPRODUCE_FIGURES Generate all manuscript figures
%
% This script reproduces all figures for the manuscript:
% "Dynamic excitability enhances robustness and capacity in 
%  recurrent neural networks"

% Create output directories if they don't exist
if ~exist('../../figures_out/main', 'dir')
    mkdir('../../figures_out/main');
end
if ~exist('../../figures_out/supplementary', 'dir')
    mkdir('../../figures_out/supplementary');
end

fprintf('========================================\n');
fprintf('Reproducing all manuscript figures\n');
fprintf('========================================\n\n');

% Navigate to figures directory
cd ../figures

% Generate main figures
fprintf('Generating Figure 1 (Overview)...\n');
fig01_generate_main;
fprintf('✓ Figure 1 complete\n\n');

fprintf('Generating Figure 2 (Classical Hopfield)...\n');
fig02_generate_classical;
fprintf('✓ Figure 2 complete\n\n');

fprintf('Generating Figure 3 (Spiking Network)...\n');
fig03_generate_spiking_network;
fprintf('✓ Figure 3 complete\n\n');

fprintf('Generating Figure 4A (Spiking Dynamics)...\n');
fig04_generate_spiking_A;
fprintf('✓ Figure 4A complete\n\n');

fprintf('Generating Figure 4B (Spiking Performance)...\n');
fig04_generate_spiking_B;
fprintf('✓ Figure 4B complete\n\n');

% Generate supplementary figures
fprintf('Generating Supplementary Figure 1...\n');
figS01_generate_supplementary;
fprintf('✓ Supplementary Figure 1 complete\n\n');

% Return to simulations directory
cd ../simulations

fprintf('========================================\n');
fprintf('All figures generated successfully!\n');
fprintf('========================================\n');
fprintf('Main figures saved to: figures_out/main/\n');
fprintf('Supplementary figures saved to: figures_out/supplementary/\n');

end