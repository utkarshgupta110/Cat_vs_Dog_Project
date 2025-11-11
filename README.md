# 🐱🐶 Cat vs Dog Image Classifier using Deep Learning (MobileNetV2)

This project classifies images as either **Cat 🐱** or **Dog 🐶** using a **Convolutional Neural Network (CNN)** based on **MobileNetV2 Transfer Learning**. It is designed as a simple and complete academic submission project.

---

## 📘 Project Overview
This project builds a deep learning model using TensorFlow and Keras to distinguish between cat and dog images. It applies **Transfer Learning** for efficient training and high accuracy even on a small, balanced dataset (≈1012 cat images and ≈1013 dog images).

---

## 🧠 Objective
- Detect whether an image contains a **Cat** or a **Dog**
- Use **MobileNetV2** as a pretrained base model
- Apply fine-tuning to improve accuracy
- Evaluate performance and make predictions on custom images

---

## 🧩 Technologies Used
- Python 🐍  
- TensorFlow / Keras  
- NumPy, Matplotlib, Seaborn  
- OpenCV, Pillow  
- Scikit-learn  
- MobileNetV2 (Transfer Learning)

---

## 📂 Project Structure
Cat_vs_Dog_Project/  
│  
├── data/  
│   ├── cat/  
│   └── dog/  
│  
├── notebooks/  
│   └── Cat_vs_Dog_Transfer.ipynb  
│  
├── outputs/  
│   └── catdog_model.keras  
│  
├── requirements.txt  
├── README.md 
└── .gitignore

---

## ⚙️ Installation Guide
### Step 1 — Clone the Repository
git clone https://github.com/YOUR-USERNAME/Cat_vs_Dog_Project.git  
cd Cat_vs_Dog_Project  

### Step 2 — Create Virtual Environment
python -m venv venv  
venv\Scripts\activate      # for Windows  
# or  
source venv/bin/activate   # for Mac/Linux  

### Step 3 — Install Dependencies
pip install -r requirements.txt  

---

## 🧾 requirements.txt
tensorflow>=2.10  
numpy>=1.24  
matplotlib>=3.7  
pandas>=2.0  
opencv-python>=4.7  
scikit-learn>=1.2  
seaborn>=0.12  
Pillow>=9.5  
requests>=2.31  

---

## 🚀 How to Run
1. Open Jupyter Notebook or Google Colab  
2. Open `notebooks/Cat_vs_Dog_Transfer.ipynb`  
3. Run all cells in order  
4. The model will train, evaluate, and save automatically  

---

## 🧠 Model Details
- **Base Model:** MobileNetV2 pretrained on ImageNet  
- **Top Layers:** Dense(128, ReLU), Dropout(0.3), Dense(1, Sigmoid)  
- **Loss Function:** Binary Crossentropy  
- **Optimizer:** Adam  
- **Validation Accuracy:** ~90–96%  

---

## 🧾 Predict Your Own Image
predict_image(r"C:\Users\Utkarsh Gupta\OneDrive\Desktop\Cat_vs_Dog_Project\data\dog\dog.4014.jpg", model)  

Expected Output:  
🧠 Prediction: Dog 🐶  
🎯 Confidence: 97.25%  

---

## 💾 Save and Load Model
model.save("outputs/catdog_model.keras")  
model = tf.keras.models.load_model("outputs/catdog_model.keras")  

---

## 📊 Example Predictions
| Input Image | Prediction |
|--------------|-------------|
| ![cat](https://i.imgur.com/4AiXzf8.jpeg) | 🐱 Cat |
| ![dog](https://i.imgur.com/Xq2dJcv.jpeg) | 🐶 Dog |

---

## 🧩 Troubleshooting
**Issue:** `NameError: name 'tf' is not defined`  
➡️ Solution: Run `import tensorflow as tf` before using it  

**Issue:** Model predicts only one class  
➡️ Solution: Check folder names (`cat`, `dog`) and ensure correct preprocessing  

**Issue:** `Unknown layer: 'TrueDivide'`  
➡️ Solution: Use `.keras` format instead of `.h5`  

**Issue:** `No images found`  
➡️ Solution: Ensure images are directly inside `data/cat` and `data/dog` folders  

---

## 📜 .gitignore
data/  
outputs/*.h5  
outputs/*.keras  
__pycache__/  
*.pyc  
.ipynb_checkpoints/  
venv/  
.DS_Store  

---

## 🧑‍💻 Author
**Utkarsh Gupta (202210101150001)**
**Shivendra Gupta (202210101150021)**
🎓 B.Tech in Computer Science (Data Science & AI)  
🏫 Shri Ramswaroop Memorial University  
📍 Lucknow, India  

---
