# An Energy Minimization-Based Deep Learning Approach with Enhanced Stability for the Allen-Cahn Equation

This repository contains the 1D Python (PyTorch) implementation for the paper:

**"An Energy Minimization-Based Deep Learning Approach with Enhanced Stability for the Allen-Cahn Equation"**

This code demonstrates solving the 1D Allen-Cahn equation using a  Neural Network that minimizes the Ginzburg-Landau energy functional, incorporating specific techniques for enhanced stability, notably a variance constraint loss term.

## Overview

The Allen-Cahn equation is a reaction-diffusion equation that models phase separation processes. Traditional PINN approaches often minimize the PDE residual directly. This work focuses on minimizing the associated energy functional:

E(u) = ∫ [ (β²/2) * (∇u)² + (1/4) * (u² - 1)² ] dx

The network `u(x)` is trained to minimize this energy, subject to boundary conditions and an initial state (via pre-training). To improve stability and prevent the solution from collapsing to trivial constants (+1 or -1), we introduce additional loss terms, including:

1.  **Boundary Loss:** Enforces Neumann boundary conditions (u_x = 0 at x = -1, 1).
2.  **Variance Constraint Loss:** Penalizes solutions where the variance of the predicted output `U_pred` falls below a specified minimum threshold (`σ²_min`), calculated as `L_V = (max(0, σ²_min - Var(U_pred)))²`. This encourages non-trivial phase profiles.

## Features

*   Implementation of a PINN using PyTorch for the 1D Allen-Cahn equation.
*   Energy functional minimization as the primary loss component.
*   Neumann boundary condition enforcement.
*   Variance constraint loss for enhanced stability.
*   Pre-training step to initialize the network close to the initial condition `u(x, t=0)`.
*   Command-line arguments for configuring hyperparameters (network depth/width, epochs, β).
*   Saving of results: model weights, loss history (total, residual, boundary, variance), final predictions, and parameters.
*   Plotting utilities for visualizing loss curves and the final predicted solution.


## Usage

Run the main script from the command line:

```bash
python main.py
