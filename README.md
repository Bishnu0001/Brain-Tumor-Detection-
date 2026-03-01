# 🧠 Brain Tumor Detection using Deep Learning

This project is a **Deep Learning based Brain Tumor Detection System** that classifies MRI brain images into four categories:

- Glioma Tumor
- Meningioma Tumor
- Pituitary Tumor
- No Tumor

The system uses a **Convolutional Neural Network (CNN)** model built with **TensorFlow/Keras** and provides predictions through a **Streamlit Web Application**.

This project demonstrates how **Artificial Intelligence can assist in medical image analysis** by automatically detecting brain tumors from MRI images.

---

## 🚀 Features

✔ Brain Tumor Classification  
✔ CNN Deep Learning Model  
✔ MRI Image Processing  
✔ Model Training and Testing  
✔ Real-time Prediction  
✔ Streamlit Web Application  
✔ Confidence Score Display

---

## 🛠 Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- Streamlit
- PIL (Python Imaging Library)

---

## 🧠 Model Architecture

The CNN model consists of:

- Convolution Layer (32 Filters)
- MaxPooling Layer
- Convolution Layer (64 Filters)
- MaxPooling Layer
- Flatten Layer
- Dense Layer (128 Neurons)
- Output Layer (Softmax)

Number of Classes = **4**

---

## 📂 Dataset Structure
Dataset/
├── Training/
│ ├── glioma/
│ ├── meningioma/
│ ├── pituitary/
│ └── notumor/
│
└── Testing/
├── glioma/
├── meningioma/
├── pituitary/
└── notumor/
