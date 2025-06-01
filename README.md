# CNN Image Classification (MNIST & CIFAR-10)

This project uses Convolutional Neural Networks (CNNs) with TensorFlow/Keras to classify images from two popular datasets: **MNIST** (handwritten digits) and **CIFAR-10** (natural images like dogs, planes, etc).

The entire pipeline — from data loading to visualization — is contained in a single script: `CNN.py`.

---

## 🧠 Features

- ✅ Loads & preprocesses MNIST and CIFAR-10 datasets
- 🧱 Implements two CNN architectures:
  - Simple CNN (basic model)
  - Deep CNN (stacked Conv layers)
- 📉 Trains and validates models with EarlyStopping
- 📈 Generates:
  - Sample image grids
  - Accuracy/loss plots
  - Confusion matrix
  - Prediction previews (color-coded)
- 🧪 Evaluation using accuracy & classification reports

---

## 📦 Datasets

- **MNIST**: 28x28 grayscale digits (0–9)
- **CIFAR-10**: 32x32 RGB images in 10 classes (airplane, dog, truck, etc.)

---

## 🚀 Getting Started

1. **Clone the repository**
   ```bash
   git clone https://github.com/RayanBatada/cnn-image-classification.git
   cd cnn-image-classification
