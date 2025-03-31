# Adversarial-Attacks-MNIST-CIFAR10

- This project implements **adversarial attacks** on deep learning models using the **FGSM** and **PGD** methods, on both **MNIST** and **CIFAR-10** datasets.
- The implementation includes both **targeted** and **untargeted** variants of the attacks.

## 📚 Table of Contents
- [Overview](#-overview)
- [Project Structure](#-project-structure)
- [Setup](#-setup)
- [Usage](#-usage)
- [Results](#-results)
- [References](#-references)

## 🔍 Overview

Adversarial examples are inputs to a model that are intentionally designed to cause the model to make a mistake. This project explores:

- **FGSM (Fast Gradient Sign Method)**
- **PGD (Projected Gradient Descent)**
- Both in **Targeted** and **Untargeted** forms

Datasets used:
- [✔] MNIST (Grayscale, 10-class digits)
- [✔] CIFAR-10 (RGB, 10-class objects)

## 📁 Project Structure

```
.
├── attacks/
│   └── fgsm.py                  # FGSM attack (targeted & untargeted)
│   └── pgd.py                   # PGD attack (targeted & untargeted)
├── models/
│   ├── mnist_model.py          # MLP for MNIST
│   └── cifar10_model.py        # ResNet18 for CIFAR-10
├── utils/
│   └── data_loader.py          # MNIST / CIFAR-10 loaders
├── test_mnist.py               # Training + Attack on MNIST
├── test_cifar.py               # Training + Attack on CIFAR-10
├── results/
│   └── *.png                   # Visualizations of adversarial examples
├── requirements.txt
└── README.md
```

## ⚙️ Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/adversarial-attacks.git
cd adversarial-attacks
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

## 🚀 Usage

### Train and attack on **MNIST**:
```bash
python test_mnist.py
```

### Train and attack on **CIFAR-10**:
```bash
python test_cifar.py
```

This will:
- Train a model from scratch
- Evaluate clean/test accuracy
- Perform FGSM & PGD attacks (both targeted and untargeted)
- Save visualizations to `results/`

## 📊 Results (example)

| Dataset   | Attack Type       | Accuracy |
|-----------|-------------------|----------|
| MNIST     | Clean             | 98.5%    |
| MNIST     | FGSM Untargeted   | 2.1%     |
| MNIST     | PGD Targeted      | 10.3%    |
| CIFAR-10  | Clean             | 85.0%    |
| CIFAR-10  | FGSM Untargeted   | 15.4%    |
| CIFAR-10  | PGD Targeted      | 33.2%    |

Visualization examples will be saved to:
```
results/mnist_attacks.png
results/cifar10_attacks.png
```

## 📖 References

- [Explaining and Harnessing Adversarial Examples (Goodfellow et al., 2014)](https://arxiv.org/abs/1412.6572)
- [Towards Deep Learning Models Resistant to Adversarial Attacks (Madry et al., 2017)](https://arxiv.org/abs/1706.06083)
- PyTorch official tutorials
