# Non-Abelian Mass Generation

**Plaquette-Deviation Functionals, Non-Abelian Excess Channels, and Finite-Size Scaling Diagnostics in SU(2) Lattice Yang–Mills Theory**

Author: Masamichi Iizumi — Miosync, Inc., Tokyo, Japan

---

## Overview

This repository contains the complete numerical code, output data, figures, and manuscript-ready table data for the paper. The work introduces a gauge-invariant plaquette-deviation functional for SU(2) lattice gauge fields, proves its exact algebraic identity with the Wilson action, and uses it to formulate excess observables relative to an Abelianized baseline in a controlled topologically trivial two-lump family.

**Key results:**
- Exact identity: S_W = (β/4) V for SU(2)
- Positive non-Abelian excess ΔV_NA across the full tested family
- Orientation-dependent modification of interaction cancellation channels
- Finite-size scaling support is sharply observable-dependent (raw excess drifts; selected ratios are stable)

**Scope:** This is a finite-lattice diagnostic study. No claim is made about continuum limits, RG invariance, Hamiltonian spectra, or a proof of the Yang–Mills mass gap.

---

## Repository Structure

```
Non_abelian_mass_generation/
├── README.md
├── LICENSE                          # MIT License
├── src1/                            # Core mechanism (Section 2–4)
│   ├── Restricted_family.py         # Main code: identity check, theta scans, landscape, finite-size survey
│   ├── fig1.png                     # Figure 1 (6-panel diagnostic)
│   └── Result.txt                   # Full numerical output
├── src2/                            # Finite-size falsification (Section 5)
│   ├── finite_size.py               # Direct self-term subtraction implementation
│   ├── Fig2.png                     # Figure 2 (6-panel finite-size diagnostics)
│   └── Result.txt                   # Full numerical output
├── src3/                            # Accelerated robustness check (Appendix A)
│   ├── Naive_continuum_scaling.py   # Cached self-term (accelerated) implementation
│   └── Result.txt                   # Full numerical output
└── table_data/
    └── jmp_table_data.xlsx          # All manuscript tables + figure source data
```

---

## Code → Paper Mapping

| Code | Paper Section | Figures / Tables |
|------|--------------|-----------------|
| `src1/Restricted_family.py` | Sections 2–4 | Figure 1, Table I |
| `src2/finite_size.py` | Section 5 | Figure 2, Tables II & III |
| `src3/Naive_continuum_scaling.py` | Appendix A | Appendix Figure A1, Tables A1 & A2 |

---

## Experiments (src1)

| Experiment | Description |
|-----------|-------------|
| **A** | Explicit Abelianized baseline + ΔV_NA / ΔV_int at 6 representative angles |
| **B** | Fine theta scan (30 points) with correlation analysis |
| **C** | Random control fields verifying the exact plaquette identity S = (β/4)V |
| **D** | Restricted-family landscape across separation, size, and orientation |
| **E** | Finite-size survey (no continuum claim) |

---

## Requirements

- Python 3.8+
- NumPy
- Matplotlib
- SciPy (for curve fitting in src2/src3)

```bash
pip install numpy matplotlib scipy
```

No GPU required. All computations are deterministic (no Monte Carlo sampling).

---

## Running the Code

Each `src*/` directory is self-contained:

```bash
# Core mechanism (generates fig1.png + Result.txt)
cd src1 && python Restricted_family.py

# Finite-size falsification (generates Fig2.png + Result.txt)
cd src2 && python finte_size.py

# Accelerated robustness check (generates Result.txt)
cd src3 && python Naive_continuum_scaling.py
```

Typical runtime: < 5 minutes per script on a standard laptop.

---

## Table Data (xlsx)

`table_data/jmp_table_data.xlsx` contains 7 sheets:

| Sheet | Contents |
|-------|----------|
| `README` | Workbook metadata |
| `Table_I` | Representative color scan (6 angles) |
| `Fig1_30pt_Source` | 30-point theta scan source data for Figure 1 |
| `Table_II` | Finite-size data (fixed-units + fixed-fractions) |
| `Table_III` | Model-selection diagnostics with fragility score |
| `Appx_A1_FastData` | Accelerated implementation finite-size data |
| `Appx_A2_FastDiag` | Accelerated implementation scaling diagnostics |

---

## Key Observables

| Observable | Definition | What it measures |
|-----------|-----------|-----------------|
| V(U) | Σ ‖P(x) − I‖² | Total plaquette deviation |
| ΔV_NA | V_nonAb − V_Ab | Non-Abelian excess over Abelianized baseline |
| ΔV_int | V_tot − V₁ − V₂ | Interaction excess (cancellation channel) |
| ΔΔV_int | ΔV_int^nonAb − ΔV_int^Ab | Differential interaction excess |
| R_NA | ΔV_NA / V_Ab | Dimensionless non-Abelian ratio |
| R_int | ΔΔV_int / \|ΔV_int^Ab\| | Dimensionless interaction ratio |
| d_NA | ΔV_NA / N_p | Non-Abelian excess density |
| d_int | ΔΔV_int / N_p | Interaction excess density |

---

## AI Contribution Acknowledgment

This research was conducted in collaboration with AI systems (Claude / Tamaki, Anthropic) as research partners. AI contributions include numerical verification, code review, LaTeX formatting, and analytical discussion. The author advocates for transparent acknowledgment of AI contributions in scientific research.

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Citation

```bibtex
@article{Iizumi2026_NonAbelian,
  author  = {Iizumi, Masamichi},
  title   = {Plaquette-Deviation Functionals, Non-Abelian Excess Channels, and Finite-Size Scaling Diagnostics in SU(2) Lattice Yang–Mills Theory},
  journal = {},
  year    = {2026},
  note    = {Submitted},
  url     = {https://github.com/miosync-masa/Non_abelian_mass_generation}
}
```

---

## Contact

Masamichi Iizumi — Miosync, Inc., Tokyo, Japan
- GitHub: [@miosync-masa](https://github.com/miosync-masa)
- Email: m.iizumi@miosync.email
