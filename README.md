# Dynamic Excitability Enhances Robustness and Capacity in Recurrent Neural Networks

This repository contains the **MATLAB** code (and optional cached outputs) used to reproduce the results reported in the manuscript:

**Dynamic excitability enhances robustness and capacity in recurrent neural networks**  
(submitted to *PNAS Nexus*)

The codebase includes:
- **Classical Hopfield networks** with fixed thresholds
- Networks with **dynamic, activity-dependent thresholds** parameterized by γ and β
- Rate-based and spiking (**leaky integrate-and-fire, LIF**) formulations
- Noise-free and noisy regimes
- Standard performance metrics (e.g., recall accuracy/overlap, robustness, convergence dynamics)

---

## Repository Structure

```
.
├── code/
│   ├── figures/
│   │   ├── fig01_generate_main.m
│   │   ├── fig02_generate_classical.m
│   │   ├── fig03_generate_spiking_network.m
│   │   ├── fig04_generate_spiking.m
│   │   └── figS01_generate_supplementary.m
│   │
│   └── simulations/
│       └── reproduce_figures.m
│
├── figures_out/
│   ├── main/
│   └── supplementary/
│
└── README.md
```

**Note:** Each figure generation script is self-contained and requires no external parameter or utility files.

---

## Requirements

- MATLAB R2021a or later (tested with MATLAB R2022b)
- No proprietary toolboxes required beyond core MATLAB

Runtime scales with network size and parameter sweeps; large runs may take several hours on a standard workstation.

---

## Quick Start

### Reproduce All Manuscript Figures

From the repository root:

```matlab
cd code/simulations
reproduce_figures
```

This will generate all figures and save them to `figures_out/main/` and `figures_out/supplementary/`.

### Generate a Single Figure

Navigate to the figures directory and run the corresponding script:

```matlab
cd code/figures

% Generate Figure 1 (Overview - threshold decomposition)
fig01_generate_main

% Generate Figure 2 (Classical Hopfield network)
fig02_generate_classical

% Generate Figure 3 (Spiking network)
fig03_generate_spiking_network

% Generate Figure 4 (Spiking dynamics comparison)
fig04_generate_spiking

% Generate Supplementary Figure 1
figS01_generate_supplementary
```

**Output locations:**
- Main figures: `figures_out/main/`
- Supplementary figures: `figures_out/supplementary/`

---

## Model Summary

Binary patterns are stored using Hebbian connectivity:

$$
W = \frac{1}{N}\sum_\mu \xi^\mu (\xi^\mu)^\top
$$

Dynamic thresholds are defined (manuscript notation) by:

$$
\theta_i = \gamma\, \frac{\mathbf{s}^\top W\mathbf{s}}{N}\, s_i + \beta\, d_i
$$

where the first term is a global, parallel component and the second term introduces controlled heterogeneity.

Full definitions, numerical schemes, and parameter values are documented in the manuscript and Supplementary Information.

---

## Data and Code Availability

- **Code:** All MATLAB source code required to reproduce the figures is contained in this repository.
- **Data:** All figure outputs are generated directly from code. Each figure script is self-contained and does not require external data files.
- **Reproducibility:** Scripts use explicit random seeds for deterministic results.

Upon acceptance, a permanent archival copy of this repository will be deposited in a public repository (e.g., Zenodo or Code Ocean) and assigned a DOI. The DOI will be added here and cited in the manuscript.

---

## Use of AI Tools

Generative AI tools (Claude 3.5 Sonnet, Anthropic) were used to assist with code documentation, repository organization, and manuscript editing. All scientific content, code logic, analyses, and conclusions were produced and verified by the authors.

---

## License

MIT License (recommended for broad reuse). If you intend a different license, replace this section and add a `LICENSE` file.

---

## Citation

If you use this code, please cite:

Savchenko L. *et al.* Dynamic excitability enhances robustness and capacity in recurrent neural networks. *PNAS Nexus* (under review).

(Citation details will be updated upon publication.)

---

## Contact

Leonid Savchenko, University College London  
Email: savtchenko@yahoo.com
