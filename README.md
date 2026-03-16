# Japanese Characters Classifier

A neural network built with PyTorch to classify handwritten Japanese Hiragana characters from the Kuzushiji-MNIST (KMNIST) dataset.

## Overview

This project trains a neural network to recognise 10 classes of handwritten Hiragana characters. The dataset contains 70,000 grayscale images (28×28 pixels), split into training and testing sets. The goal is to achieve at least **80% accuracy** with minimal overfitting across a series of experiments.

## Dataset

The [Kuzushiji-MNIST](https://www.kaggle.com/datasets/anokas/kuzushiji/data) dataset consists of four files:

| File | Description |
|---|---|
| `kmnist-train-imgs.npz` | Training images (60,000) |
| `kmnist-train-labels.npz` | Training labels |
| `kmnist-test-imgs.npz` | Test images (10,000) |
| `kmnist-test-labels.npz` | Test labels |

Download these from Kaggle and place them in the project root (or upload them directly into the Colab notebook).

## Project Structure

```
├── Assignment1-PartB-Instructions.ipynb   # Main notebook with instructions and code
└── README.md
```

## Workflow

1. **Import & Load Data** — Load the `.npz` files with NumPy and visualise sample images.
2. **Prepare Data** — Reshape images to include a channel dimension, cast to `float32`, standardise pixel values to [0, 1], and one-hot encode the labels.
3. **Define Model** — Design a neural network architecture in PyTorch.
4. **Train** — Train the model using a chosen optimiser and loss function over multiple epochs.
5. **Evaluate** — Measure accuracy on training and test sets, plot learning curves, and generate a confusion matrix.

## Requirements

- Python 3.x
- NumPy
- Matplotlib
- PyTorch

A GPU runtime (e.g. Google Colab with GPU accelerator) is recommended for faster training.

## Getting Started

1. Clone this repository.
2. Download the KMNIST dataset files from [Kaggle](https://www.kaggle.com/datasets/anokas/kuzushiji/data).
3. Open `Assignment1-PartB-Instructions.ipynb` in Google Colab or Jupyter Notebook.
4. Follow the instructions and fill in the **TODO** sections.
5. Run at least 3 experiments, tuning architecture and hyperparameters to reach ≥80% test accuracy.
