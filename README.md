# Sign_language_recognition
This project is a deep learning-based Sign Language Recognition System that detects American Sign Language (ASL) gestures using a Convolutional Neural Network (CNN). It is trained on the publicly available Sign Language MNIST dataset, which contains grayscale images of hand signs representing 24 English alphabets (A–Y, excluding J and Z). 

# 🖐️ Sign Language Recognition System (CNN-based)

This project builds a real-time **Sign Language Recognition System** using a **Convolutional Neural Network (CNN)** trained on the [Sign Language MNIST](https://www.kaggle.com/datasets/datamunge/sign-language-mnist) dataset. It recognizes hand gestures corresponding to alphabets A–Y (excluding J and Z) and displays predictions live through your webcam.

---

## 🚀 Features

- Trained CNN with 95%+ accuracy
- Real-time gesture prediction via webcam
- Preprocessing: grayscale conversion, normalization, reshaping
- Predicts 25 static gestures (A–Y, excluding J and Z)
- Uses OpenCV to display live video with predictions

---

## 🧰 Tech Stack

- TensorFlow / Keras
- OpenCV
- Python: NumPy, Pandas, Matplotlib, Seaborn
- Scikit-learn

---

## 📦 Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Abhishek-soni-02 /sign-language-recognition.git
cd sign-language-recognition

📁 Dataset Information
Source: Kaggle - Sign Language MNIST

Format: CSV file where each row represents a 28x28 grayscale image, flattened into 784 columns, with a label (0–24) representing the sign.

Classes: 25 static signs (A–Y), excluding J and Z (which require motion)

📊 Model Performance
Architecture: 3 Convolutional layers with Batch Normalization, MaxPooling, Dropout

Final Accuracy: ~95% on test set

Loss Function: Categorical Crossentropy

Optimizer: Adam

📝 Future Improvements
 Add Flask or Streamlit web deployment

 Improve hand segmentation using skin detection or MediaPipe

 Add sound feedback for accessibility

 Save predictions or create live dashboard

👨‍💻 Author
Abhishek Soni
B.Tech CSE (AI/ML) — Manipal University Jaipur
📍 India

LinkedIn | GitHub

📄 License
This project is licensed under the MIT License — feel free to use and modify it.

🙏 Acknowledgements
Kaggle: Sign Language MNIST Dataset

TensorFlow & OpenCV communities

