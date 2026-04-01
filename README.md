# Deep-neural-network-from-scratch
Implementation of a deep feedforward neural network from scratch using NumPy, with training visualizations and decision boundaries.

# Deep Feedforward Neural Network from Scratch

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/numpy-1.21+-blue.svg)](https://numpy.org/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sania-abid/deep-neural-network-from-scratch/blob/main/dfnn_from_scratch.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project implements a **deep feedforward neural network (DFNN)** entirely from scratch using **Python and NumPy**. No high‑level frameworks (TensorFlow, PyTorch) are used. The network is trained on the synthetic `make_moons` dataset to perform binary classification, and the training process is visualized with loss curves and decision boundaries.

---

## Features

- **Activation functions** (Sigmoid, ReLU, Tanh) and their derivatives.
- **Numerically stable binary cross‑entropy loss** with clipping.
- **DeepNeuralNetwork class** with:
  - Forward and backward propagation (vectorized).
  - Gradient descent optimization (full‑batch and mini‑batch).
  - Support for multiple hidden layers and different activations.
  - L2 regularization (weight decay) option.
- **Visualizations**:
  - Dataset scatter plot.
  - Training loss over epochs.
  - Decision boundary with confidence scores.
- **Evaluation** – accuracy and confusion matrix.
- **Gradient check** for debugging.
- **Model saving** (download from Colab).

---

## Dataset

The `make_moons` dataset (200 samples, noise=0.2) is used. It is a classic 2D binary classification problem with a non‑linear decision boundary, ideal for testing neural networks.

---

## Architecture

A default architecture is used for demonstration:
- Input layer: 2 neurons
- Hidden layer 1: 20 neurons, ReLU
- Hidden layer 2: 10 neurons, ReLU
- Output layer: 1 neuron, Sigmoid

Weights are initialized using **He initialization** (recommended for ReLU). Training uses **full‑batch gradient descent** with learning rate 0.05 for 1000 epochs.

---

## Results

The trained network achieves **~71% training accuracy** (as seen in the notebook) and produces a smooth decision boundary that separates the two moon shapes.

*For visualizations of the loss curve and decision boundary, refer to the notebook.*

---

## Getting Started

### Prerequisites
- Python 3.8+
- Install dependencies:
  ```bash
  pip install -r requirements.txt
