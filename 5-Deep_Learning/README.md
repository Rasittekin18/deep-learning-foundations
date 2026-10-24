# Deep Learning: From Scratch to Advanced Frameworks

A practical and structured deep learning repository bridging fundamental mathematical concepts with modern deep learning frameworks.

## 📌 Features

- **From-Scratch Implementations:** Mathematical foundations of Forward/Backward Propagation and Gradient Descent using pure `NumPy`.
- **Framework Implementations:** Model pipelines built with **Keras / TensorFlow** and **PyTorch**.
- **Real-World Datasets:**
  - *Sign Language Digit Dataset* ($64 \times 64$ gesture images)
  - *MNIST / Digit Recognizer* ($28 \times 28$ handwritten digits)
  - *Stock Price Time-Series Data* (Sequential trends)

---

## 📚 Modules Overview

### 1. Data Preprocessing & Exploration
- Pixel intensity normalization ($[0, 255] \rightarrow [0, 1]$).
- Categorical One-Hot Encoding for multi-class classification.
- Tensor reshaping and channel formatting $(N, H, W, C)$.

### 2. Logistic Regression (Single-Layer Neural Network)
- **Math:** $Z = W^T X + b$, $A = \sigma(Z) = \frac{1}{1 + e^{-Z}}$
- Binary Cross-Entropy Loss computation.
- Explicit gradient updates via Backpropagation.
- Pure `NumPy` implementation vs `scikit-learn` benchmark.

### 3. Artificial Neural Networks (ANN)
- Addition of non-linear Hidden Layers using $\tanh$ activation.
- Random Parameter Initialization  to break symmetry.
- Custom Multi-Layer Perceptron (MLP) vs `Keras Sequential` & `PyTorch Module`.

### 4. Convolutional Neural Networks (CNN)
- Spatial feature extraction using $3 \times 3$ / $5 \times 5$ Kernels.
- **Same Padding** and **Max Pooling** for dimension control and spatial invariance.
- **Dropout** regularization to prevent overfitting.
- Data Augmentation (`ImageDataGenerator`) and classification via **Softmax**.

### 5. Recurrent Neural Networks & LSTM (RNN / LSTM)
- Sequential analysis and temporal dependency modeling.
- Mitigation of Vanishing Gradients using **LSTM Gates** (Forget, Input, Cell State, Output).
- Time-series windowing and forecasting.

---

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/Rasittekin18/5-Deep_Learning.git
cd 5-Deep_Learning

# Install dependencies
pip install numpy pandas matplotlib scikit-learn tensorflow torch jupyterlab

# Launch Notebooks
jupyter lab
```
