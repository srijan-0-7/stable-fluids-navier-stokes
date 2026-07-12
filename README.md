# Stable Numerical Simulation of Two-Dimensional Incompressible Fluid Flow

A real-time implementation of Jos Stam's **Stable Fluids** algorithm for simulating two-dimensional incompressible fluid dynamics using the Navier–Stokes equations.

The project demonstrates the numerical techniques commonly used in computational fluid dynamics (CFD) and computer graphics, including semi-Lagrangian advection, implicit diffusion, pressure projection, and interactive momentum injection.

---

## Overview

This simulator solves the incompressible Navier–Stokes equations on a staggered grid using stable numerical integration methods that remain robust under large time steps.

The implementation is entirely written in Python using NumPy for numerical computation and Pygame for visualization.

Unlike particle-based simulations, this solver models velocity and density fields on an Eulerian grid while maintaining incompressibility through pressure projection.

---

## Demonstration

<p align="center">

### Fluid Simulation

<img src="assets/fluid.gif" width="750"/>

### Velocity Field Visualization

<img src="assets/velocity.gif" width="750"/>

### Vortex Injection

[https://github.com/user-attachments/assets/5bcf05e4-8fbe-4540-a429-9256af5ef70e](https://github.com/user-attachments/assets/8489374f-a5ff-490f-97df-1c03f34af809)

</p>

---

## Numerical Methods

The simulation pipeline follows Jos Stam's Stable Fluids formulation:

1. Velocity diffusion
2. Pressure projection
3. Semi-Lagrangian velocity advection
4. Pressure projection
5. Density diffusion
6. Density advection
7. Rendering

Core numerical techniques include:

- Semi-Lagrangian Advection
- Implicit Diffusion
- Jacobi Linear Solver
- Pressure Projection
- Divergence-Free Velocity Enforcement
- Finite Difference Spatial Discretization
- Gaussian Vortex Injection

---

## Mathematical Model

The simulation solves the incompressible Navier–Stokes equations
<img width="2816" height="1536" alt="Image" src="https://github.com/user-attachments/assets/778cbc88-4d5c-4147-8e7b-21b97bac5566" />

## Features

- Real-time 2D fluid simulation
- Stable Navier–Stokes solver
- Interactive dye injection
- Interactive momentum injection
- Vortex generation
- Velocity field visualization
- Density transport
- Pressure projection
- Implicit diffusion
- Configurable simulation parameters
- Interactive rendering at 60 FPS

---

## Controls

| Key | Action |
|------|--------|
| Left Mouse | Inject dye and momentum |
| Right Mouse | Spawn vortex |
| V | Toggle velocity vectors |
| Space | Pause simulation |
| C | Clear simulation |

---

## Installation

```bash
git clone https://github.com/username/stable-fluids-navier-stokes.git

cd stable-fluids-navier-stokes

pip install -r requirements.txt
```

---

## Run

```bash
python fluid.py
```

---

## Dependencies

- Python 3.x
- NumPy
- Pygame
```bash
pip install numpy pygame
```
---

## References

Jos Stam.

**Stable Fluids.**

Proceedings of SIGGRAPH 1999.

Bridson, Robert.

**Fluid Simulation for Computer Graphics.**

CRC Press.

---

## Future Work

- GPU acceleration using CUDA
- Adaptive grids
- Pressure multigrid solver
- Vorticity confinement
- Higher-order advection schemes
- Free surface simulation
- 3D fluid solver
- Smoke and fire rendering
- OpenGL visualization

---

## License

MIT License
