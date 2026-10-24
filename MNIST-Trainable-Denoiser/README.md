MNIST Trainable Denoiser 🧹
- This project implements a trainable denoising model for the MNIST dataset.
It removes Gaussian noise from handwritten digit images using a learnable kernel optimized via gradient descent.

🚀 Features
- Loads and preprocesses MNIST dataset (normalized 0–1).
- Adds Gaussian noise with user-defined mean (μ) and standard deviation (σ).
- Trains a single trainable nxn kernel to denoise images.
- Uses Euclidean distance between clean and denoised images as the loss function.
- Plots loss over all iterations.
- Visualizes results: noisy input, denoised output, and original image.
- Shows kernel distribution and 3D surface plot of the learned filter.

📂 Project Structure 
mnist-trainable-denoiser/
│── mnist_denoiser.py     # Main training and evaluation script
│── README.md             # Project description
│── requirements.txt      # Dependencies
│── results/              # (Optional) save images or figures

🔧 Installation
- Clone the repository and install dependencies: 

- git clone https://github.com/Rasittekin18/mnist-trainable-denoiser.git
- cd mnist-trainable-denoiser
- pip install -r requirements.txt

▶️ Usage
- Run the script to train the kernel and visualize results:
- python mnist_denoiser.py

- Training uses the first 1000 MNIST images by default.
- Loss graph, kernel visualization, histogram, 3D surface, and sample image comparisons are generated automatically.


📊 Example Output

- Loss over iterations (Euclidean distance between clean and denoised images)
-Learned kernel visualization
- Histogram of kernel values
- 3D surface of kernel
- Comparison of noisy, denoised, and original images
