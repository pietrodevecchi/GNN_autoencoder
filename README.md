
# GNN Autoencoder for Reduced Order Modeling of Dynamical Systems

This repository implements the methodologies presented in the MSc Thesis titled **"Graph Neural Networks based AutoEncoder in Reduced Order Modeling of Dynamical Systems"**, authored by Pietro Devecchi at Politecnico di Milano. The thesis proposes a Graph Neural Network (GNN)-based Autoencoder framework for compressing and reconstructing solutions of Partial Differential Equations (PDEs), particularly targeting the advection-diffusion problem and Navier-Stokes equations.

For further references: [Graph Neural Networks based AutoEncoder in Reduced Order Modeling of Dynamical Systems](https://www.politesi.polimi.it/handle/10589/227635).

## Table of Contents

- [Project Overview](#project-overview)
- [Key Contributions](#key-contributions)
- [Model Architecture](#model-architecture)
- [Acknowledgments](#acknowledgments)

---

## Project Overview

The repository provides a framework for **Reduced Order Modeling (ROM)** using Graph Neural Networks, targeting computationally expensive simulations of dynamical systems. It focuses on addressing two benchmark problems:
1. **Advection-Diffusion Problem**: Capturing the behavior of a scalar field in a domain with a circular obstacle whose position varies.
2. **Navier-Stokes Flow Past a Cylinder**: Simulating unsteady fluid dynamics in complex geometries.

The framework leverages GNN-based architectures to:
- Encode high-dimensional simulation data into a compact latent space.
- Directly map input parameters to this latent space for rapid inference.
- Decode latent representations back to high-dimensional solutions with minimal loss.

## Key Contributions

The thesis introduces:
- **A novel GNN-based autoencoder** capable of handling arbitrary geometric configurations.
- **Parameter-to-latent mapping** for bypassing full-order simulations during inference.
- Rigorous evaluation of the framework's ability to generalize to unseen parameter configurations and geometries.
- Application to benchmark problems with highly satisfactory reconstruction accuracy and computational efficiency.

---

## Model Architecture

The GNN Autoencoder consists of three key components:

### Encoder
- Processes graph-based PDE solutions.
- Uses graph convolutional layers to aggregate local neighborhood information.
- Encodes solutions into a reduced latent representation.

### Decoder
- Reconstructs the full-order solution from the latent space.
- Employs upsampling techniques to match the original resolution.

### Parameter-to-Latent Mapping
- Maps problem-specific input parameters (\( C_x, C_y \)) to the latent space, enabling inference without requiring full-order simulations.

Loss functions include:
- **Mean Squared Error (MSE)** between ground truth and reconstructed solutions.
- **Relative \( L^2 \) Error** for assessing solution fidelity.

---

## Acknowledgments

This work is based on the MSc Thesis submitted to Politecnico di Milano as part of the Mathematical Engineering program. The author acknowledges the guidance of supervisors and the computational resources provided by the university.

For more details, refer to the full thesis: [Graph Neural Networks based AutoEncoder in Reduced Order Modeling of Dynamical Systems](https://www.politesi.polimi.it/handle/10589/227635).

---
