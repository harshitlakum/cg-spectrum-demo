# CG Spectrum Demo

This repository demonstrates the behavior of the Conjugate Gradient (CG) method for symmetric positive-definite (SPD) systems, highlighting:
- The impact of eigenvalue distribution (clusters and outliers) on CG convergence.
- Comparison of Steepest Descent (SD) and CG for the 2D Laplace equation.

## Features

- Python and MATLAB code for all experiments.
- Convergence plots showing relative A-norm error vs iteration.
- Theoretical bounds (from CG convergence theory) for reference.
- All code is modular and easy to extend.

## Structure

- `python/`: All Python scripts and helper modules.
- `matlab/`: All MATLAB scripts and helper functions.
- `figures/`: Output plots for presentations or reports.

## How to Run

### Python
1. Install requirements:
    ```
    pip install -r requirements.txt
    ```
2. Run the eigenvalue clustering CG demo:
    ```
    python cg_spectrum_example.py
    ```
3. Run the 2D Laplace SD vs CG demo:
    ```
    python cg_sd_2d_laplace.py
    ```

### MATLAB
- Run the scripts `cg_spectrum_example.m` or `cg_sd_2d_laplace.m` in MATLAB.
- Helper functions should be in the same folder or MATLAB path.

## Theoretical Background

The CG convergence rate depends on the eigenvalue distribution of the matrix. When most eigenvalues are tightly clustered and a few are large outliers, CG rapidly "eliminates" the directions of the outliers, after which convergence is much faster (see results/plots).


