MNIST Digit Difference with Logistic Regression
- This project trains a Logistic Regression model on the MNIST dataset to recognize handwritten digits (0–9).
- It then demonstrates the absolute difference calculation between pairs of predicted digits, such as:
- |5-4| = 1
- |4-5| = 1 
- The model not only predicts the digits but also shows confidence scores (maximum probability and geometric mean probability).

🚀 Features
- Loads and preprocesses the MNIST dataset (normalization + flattening).
- Trains a Logistic Regression classifier on digit recognition.
- Evaluates model performance with Accuracy and F1-score.
- Visualizes random digit pairs with predicted labels and absolute differences.
- Displays both max confidence and geometric confidence values.

🛠️ Installation
Clone this repository and install the required dependencies:
- git clone https://github.com/Rasittekin18/mnist-digit-diff.git
- cd mnist-digit-difference
- pip install -r requirements.txt


📊 Usage
Run the script:
python mnist_diff.py 

Expected output:

- Model performance (Accuracy and F1 Score).
- 4 random digit pairs, their predicted values, absolute differences, and confidence scores.
- Visualizations of the selected images with predictions.

📈 Example Output

- Model Başarımı:
- Accuracy: 0.9205
- F1 Score: 0.9203

- Rastgele 4 Görüntü Çifti Analizi:

Görüntü Çifti 1:
  - Tahminler: 5 ve 4
  - Mutlak Fark: |5 - 4| = 1
  - Max Güven: 0.9987562356754321, 0.9990123456123456
  - Geo Güven: 0.0967345634563456, 0.1034546123456123
