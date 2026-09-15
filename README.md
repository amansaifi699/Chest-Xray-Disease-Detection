# 🩺 Chest X-ray Disease Detection using ResNet50 & Grad-CAM

<div align="center">

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge\&logo=python\&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge\&logo=tensorflow\&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge\&logo=keras\&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge\&logo=opencv\&logoColor=white)
![Research](https://img.shields.io/badge/Research-IEEE%20CE2CT%202026-blueviolet?style=for-the-badge)

### Explainable AI for Multi-Class Chest X-ray Disease Classification

*Deep Learning • Computer Vision • Medical Imaging • Explainable AI*

</div>

---

## 📖 Overview

This project presents an **AI-powered medical image classification system** for detecting chest diseases from X-ray images using **Transfer Learning with ResNet50** and **Grad-CAM Explainable AI**.

The system classifies chest X-rays into **four disease categories** and generates visual explanations showing which regions of the X-ray influenced the prediction.

This work was developed as my final-year research project and was presented at the **2026 Second International Conference on Computer Science, Electrical, Electronics and Communication Technologies (IEEE CE2CT 2026)**.

---

## 📄 Research Publication

**Title:** *Explainable Medical Image Classification using ResNet50 and Grad-CAM for Chest X-ray Analysis*

* 👨‍💻 **First Author**
* 🎤 **Conference Presenter**
* 📍 **IEEE CE2CT 2026 — Bhimtal, Uttarakhand, India**
* 📚 Proceedings submitted for **IEEE Xplore** publication.

> IEEE Xplore DOI/link will be added after publication.

---

## 🎯 Problem Statement

Early detection of lung diseases from chest X-ray images is important for supporting medical diagnosis.

This project uses **Deep Learning** and **Explainable AI** to build a model that can classify multiple chest diseases while also highlighting the important regions responsible for the prediction.

---

## 🩻 Disease Classes

The model predicts one of the following classes:

| Disease            | Description               |
| ------------------ | ------------------------- |
| 🦠 COVID           | COVID-19 infected lungs   |
| 🌫 Lung Opacity    | Opacity / lung infection  |
| 🫁 Viral Pneumonia | Viral pneumonia infection |
| ✅ Normal           | Healthy chest X-ray       |

---

## 🧠 Model Architecture

### Transfer Learning using ResNet50

The model uses a pretrained **ResNet50** network trained on ImageNet.

Pipeline:

1. Chest X-ray image preprocessing.
2. Data augmentation.
3. Transfer learning using ResNet50.
4. Fine-tuning the classification layers.
5. Prediction using Softmax.
6. Grad-CAM visualization.

---

## ⚙️ Technologies Used

### Languages

* Python
* C++

### Deep Learning

* TensorFlow
* Keras
* ResNet50
* Grad-CAM

### Data Processing

* NumPy
* Pandas
* Matplotlib
* OpenCV

### Development Environment

* Google Colab
* Jupyter Notebook
* Git
* GitHub

---

## 📂 Dataset

**COVID-19 Radiography Dataset**

The dataset contains chest X-ray images for four classes:

* COVID
* Lung Opacity
* Viral Pneumonia
* Normal

### Dataset Split

* **Training:** 70%
* **Validation:** 15%
* **Testing:** 15%

---

## 🔄 Data Preprocessing

The preprocessing pipeline includes:

* Image resizing to **224 × 224**.
* Pixel normalization.
* Random rotation.
* Zoom augmentation.
* Horizontal flipping.
* Train / Validation / Test split.

---

## 🚀 Model Training

### ResNet50 Transfer Learning

Training strategy:

* Pretrained ImageNet weights.
* Frozen backbone during initial training.
* Fine-tuning of upper ResNet layers.
* Adam optimizer.
* Categorical Cross Entropy Loss.
* Softmax classifier.

### Training Configuration

| Parameter     | Value                    |
| ------------- | ------------------------ |
| Image Size    | 224×224                  |
| Batch Size    | 32                       |
| Optimizer     | Adam                     |
| Learning Rate | 0.0001                   |
| Loss Function | Categorical Crossentropy |

---

## 📊 Results

### Model Performance

* ✅ Multi-class disease classification.
* ✅ Grad-CAM explainability.
* ✅ Transfer Learning using ResNet50.
* ✅ Evaluation using Confusion Matrix and Classification Report.

### Evaluation Metrics

* Classification Report
* Confusion Matrix
* Test Accuracy
* Validation Accuracy

---

## 🔥 Explainable AI using Grad-CAM

Grad-CAM highlights the regions of the chest X-ray that contribute most to the prediction.

Benefits:

* Improves interpretability.
* Helps visualize model attention.
* Makes predictions easier to understand.

---

## 📁 Repository Structure

```text
Chest-Xray-Disease-Detection/
│
├── README.md
├── notebook/
│   └── chest_xray_diagnosis.ipynb
├── model/
│   └── resnet50_covid_model.h5
├── src/
│   ├── train.py
│   ├── predict.py
│   ├── gradcam.py
│   ├── evaluate.py
│   └── utils.py
├── images/
├── requirements.txt
├── LICENSE
└── .gitignore
```

---

## 💻 Installation

Clone the repository.

```bash
git clone https://github.com/amansaifi699/Chest-Xray-Disease-Detection.git
```

Move into the project.

```bash
cd Chest-Xray-Disease-Detection
```

Install dependencies.

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Project

### Train Model

```bash
python src/train.py
```

### Evaluate Model

```bash
python src/evaluate.py
```

### Predict a Chest X-ray

```bash
python src/predict.py --image sample_xray.png
```

---

## 📈 Features

* Multi-class Chest X-ray Classification.
* Transfer Learning with ResNet50.
* Explainable AI using Grad-CAM.
* TensorFlow/Keras implementation.
* Data preprocessing and augmentation.
* Confusion Matrix visualization.
* Classification Report generation.

---

## 🎓 Academic Context

This project was completed as part of my **Bachelor of Technology in Computer Science & Engineering** at **IMS Engineering College (AKTU), India**.

It combines concepts from:

* Artificial Intelligence
* Deep Learning
* Computer Vision
* Medical Imaging
* Explainable AI

---

## 👨‍💻 Author

**Aman**

* 🎓 B.Tech Computer Science & Engineering (AKTU)
* 📄 First Author — IEEE CE2CT 2026
* 🤖 AI Research Enthusiast
* 🔬 Computer Vision & Explainable AI

### Connect with me

* GitHub: **amansaifi699**
* LinkedIn: **aman-b8808924a**
* Email: **[amanshooter4@gmail.com](mailto:amanshooter4@gmail.com)**

---

## ⭐ Acknowledgements

* TensorFlow & Keras
* ResNet50 pretrained ImageNet model
* COVID-19 Radiography Dataset
* IEEE CE2CT 2026 Conference

---

## 📜 License

This repository is released under the **MIT License** for academic and research purposes.

> **Disclaimer:** This project is intended for educational and research purposes only. It is **not** a clinical diagnostic tool or a substitute for professional medical advice.
