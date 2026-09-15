# 🩺 Chest X-ray Disease Detection using ResNet50 & Grad-CAM

<div align="center">

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![Research](https://img.shields.io/badge/Research-IEEE_CE2CT_2026-blueviolet?style=for-the-badge)
![Computer Vision](https://img.shields.io/badge/Computer_Vision-Medical_Imaging-success?style=for-the-badge)

### Explainable AI for Multi-Class Chest X-ray Disease Classification

*Deep Learning • Transfer Learning • Medical Imaging • Explainable AI*

</div>

---

## 📖 Project Overview

This repository contains my **Bachelor of Technology Final-Year Research Project**, which focuses on detecting multiple chest diseases from chest X-ray images using **Deep Learning** and **Explainable Artificial Intelligence (XAI)**.

A fine-tuned **ResNet50** model is used to classify chest radiographs into four disease categories, while **Grad-CAM** provides visual explanations by highlighting the lung regions responsible for predictions.

The project combines:

- 🧠 Deep Learning
- 🩻 Computer Vision
- 🔬 Medical Imaging
- 🤖 Explainable AI (Grad-CAM)

This research work was accepted and presented at an **IEEE International Conference (CE2CT 2026)**.

---

# 🎓 Research Publication

## Explainable Medical Image Classification using ResNet50 and Grad-CAM for Chest X-ray Analysis

**Conference:** 2026 Second International Conference on Computer Science, Electrical, Electronics and Communication Technologies (**IEEE CE2CT 2026**)

| Publication Information | Details |
|-------------------------|---------|
| 👨‍💻 First Author | **Aman** |
| 🎤 Role | Conference Presenter |
| 📍 Conference Venue | Graphic Era Hill University, Bhimtal, Uttarakhand, India |
| 📅 Conference Year | 2026 |
| 📚 Proceedings | Submitted to IEEE Xplore |

> **Status:** Accepted, Presented, and submitted for IEEE Xplore publication.

---

# 🎯 Problem Statement

Chest X-ray imaging is one of the most commonly used diagnostic tools for identifying respiratory diseases.

Manual interpretation requires expert radiologists and may be time-consuming in large-scale healthcare settings.

This project proposes an AI system capable of:

- Detecting multiple chest diseases.
- Improving diagnostic support.
- Providing visual explanations using Grad-CAM.
- Increasing transparency of AI predictions.

---

# 🩻 Disease Categories

The model classifies chest X-ray images into **4 categories**.

| Class | Description |
|-------|-------------|
| 🦠 COVID-19 | Chest X-rays of COVID-19 infected patients |
| 🌫 Lung Opacity | Lung opacity / infection cases |
| 🫁 Viral Pneumonia | Viral Pneumonia infected lungs |
| ✅ Normal | Healthy chest X-rays |

---

# 🧠 Model Architecture

## Transfer Learning using ResNet50

The proposed model uses a pretrained **ResNet50** convolutional neural network initialized with **ImageNet weights**.

### Model Pipeline

1. Chest X-ray image preprocessing.
2. Image resizing to **224×224**.
3. Data augmentation.
4. Feature extraction using pretrained ResNet50.
5. Fine-tuning upper convolution layers.
6. Fully connected Softmax classifier.
7. Grad-CAM visualization for explainability.

### Why ResNet50?

- Deep residual architecture.
- Better feature extraction.
- Faster convergence.
- High performance on medical image datasets.

---

# ⚙️ Technologies Used

## Programming Languages

- Python
- C++

## Deep Learning Libraries

- TensorFlow
- Keras
- NumPy
- Pandas

## Computer Vision

- OpenCV
- Matplotlib
- Grad-CAM

## Development Environment

- Google Colab
- Jupyter Notebook
- Git
- GitHub

---

# 📂 Dataset

## COVID-19 Radiography Database

Dataset Source:

> COVID-19 Radiography Database (Kaggle)

The dataset contains chest X-ray images collected from publicly available medical repositories.

### Dataset Information

| Dataset Property | Value |
|-----------------|------:|
| Total Images | **21,165** |
| Disease Classes | **4** |
| Image Type | Chest X-ray |
| Image Size | 224 × 224 |

---

## 📊 Dataset Distribution

The COVID-19 Radiography Database contains **21,165 chest X-ray images** divided into four disease categories.

<p align="center">
  <img src="./images/dataset_distribution.png" width="750"/>
</p>

**Figure 1.** Class-wise distribution of the COVID-19 Radiography Dataset used for training, validation, and testing.

| Class | Training | Validation | Testing | Total |
|------|---------:|-----------:|--------:|------:|
| COVID-19 | 2,531 | 542 | 543 | 3,616 |
| Lung Opacity | 4,208 | 902 | 902 | 6,012 |
| Normal | 7,134 | 1,529 | 1,529 | 10,192 |
| Viral Pneumonia | 941 | 202 | 202 | 1,345 |
| **Total** | **14,814** | **3,175** | **3,176** | **21,165** |

---

## 🩻 Representative Chest X-ray Images

The figure below shows representative chest radiographs used during model training.

<p align="center">
  <img src="./images/sample_xrays.png" width="700"/>
</p>

**Figure 2.** Representative chest X-ray images belonging to COVID-19, Lung Opacity, Normal, and Viral Pneumonia classes.

---

# 🔄 Data Preprocessing

The preprocessing pipeline prepares chest X-ray images before model training.

### Preprocessing Steps

- Resize images to **224 × 224** pixels.
- Normalize pixel values.
- Data augmentation.
- Shuffle dataset.
- Train/Validation/Test split.

### Data Augmentation

- Random Rotation
- Horizontal Flip
- Zoom
- Width Shift
- Height Shift

This helps improve generalization and reduce overfitting.

---

# 🚀 Model Training

## Transfer Learning Strategy

The ResNet50 backbone was initialized using pretrained ImageNet weights.

### Training Procedure

- Freeze pretrained convolution layers.
- Train custom classification head.
- Fine-tune upper ResNet layers.
- Evaluate on unseen testing dataset.

### Hyperparameters

| Parameter | Value |
|-----------|------:|
| Model | ResNet50 |
| Input Size | 224 × 224 |
| Batch Size | 32 |
| Epochs | 20 |
| Optimizer | Adam |
| Learning Rate | 0.0001 |
| Loss Function | Categorical Crossentropy |
| Output Activation | Softmax |

---

# 📈 Model Performance

The fine-tuned ResNet50 model significantly outperformed the baseline CNN model.

<p align="center">
  <img src="./images/performance_comparison.png" width="700"/>
</p>

**Figure 3.** Performance comparison between the baseline CNN model and the proposed fine-tuned ResNet50 model.

## Performance Comparison

| Metric | Baseline CNN | Fine-Tuned ResNet50 |
|-------|-------------:|--------------------:|
| Accuracy | 77.17% | **90.00%** |
| Precision | 0.78 | **0.90** |
| Recall | 0.77 | **0.90** |
| F1-Score | 0.76 | **0.90** |

---

# 📊 Confusion Matrix & Classification Report

The confusion matrix summarizes class-wise prediction performance on the testing dataset.

<p align="center">
  <img src="./images/confusion_matrix.png" width="700"/>
</p>

**Figure 4.** Confusion Matrix and Classification Report generated by the fine-tuned ResNet50 model.

## Class-wise Results

| Class | Precision | Recall | F1-Score |
|-------|----------:|-------:|---------:|
| COVID-19 | **0.94** | 0.85 | 0.89 |
| Lung Opacity | 0.89 | 0.84 | 0.87 |
| Normal | 0.88 | **0.94** | 0.91 |
| Viral Pneumonia | **0.95** | **0.94** | **0.95** |
| **Macro Average** | **0.92** | **0.89** | **0.90** |
| **Weighted Average** | **0.90** | **0.90** | **0.90** |

### Overall Test Accuracy

# **90.00%**

---

# 🔥 Explainable AI using Grad-CAM

Deep learning models are often considered black boxes.

Grad-CAM helps visualize which regions of the chest X-ray influenced the prediction.

<p align="center">
  <img src="./images/gradcam_visualization.png" width="750"/>
</p>

**Figure 5.** Grad-CAM heatmaps generated for representative chest X-ray images.

### Benefits of Grad-CAM

- Improves interpretability.
- Highlights disease-related lung regions.
- Supports explainable medical AI.
- Makes predictions easier to understand.

---

# 📂 Repository Structure

```text
Chest-Xray-Disease-Detection/
│
├── README.md
├── notebook/
│   └── Chest_Xray_Disease_Detection.ipynb
│
├── images/
│   ├── dataset_distribution.png
│   ├── sample_xrays.png
│   ├── performance_comparison.png
│   ├── confusion_matrix.png
│   └── gradcam_visualization.png
│
├── model/
│   └── resnet50_model.h5
│
├── src/
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── gradcam.py
│   └── utils.py
│
├── requirements.txt
├── LICENSE
└── .gitignore
```

---

# 💻 Installation

Clone this repository.

```bash
git clone https://github.com/amansaifi699/Chest-Xray-Disease-Detection.git
```

Move into the project directory.

```bash
cd Chest-Xray-Disease-Detection
```

Install dependencies.

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Project

## Train Model

```bash
python src/train.py
```

## Evaluate Model

```bash
python src/evaluate.py
```

## Predict a Chest X-ray

```bash
python src/predict.py --image sample_xray.png
```

## Generate Grad-CAM Heatmap

```bash
python src/gradcam.py --image sample_xray.png
```

---

# 📋 Project Features

- Multi-Class Chest X-ray Classification.
- Transfer Learning using ResNet50.
- Explainable AI using Grad-CAM.
- Medical Image Classification.
- Confusion Matrix Evaluation.
- Classification Report.
- Image Preprocessing Pipeline.
- TensorFlow / Keras Implementation.

---

# 📚 Research Contributions

This project demonstrates:

- Multi-class medical image classification.
- Transfer learning for healthcare applications.
- Explainable AI for medical diagnosis.
- Grad-CAM visualization for model transparency.
- Performance improvement over a baseline CNN.

---

# 🎓 Academic Context

This research project was completed as part of my **Bachelor of Technology in Computer Science & Engineering**.

**Institution:** IMS Engineering College, Ghaziabad

**University:** Dr. A.P.J. Abdul Kalam Technical University (AKTU)

### Research Domains

- Artificial Intelligence
- Machine Learning
- Deep Learning
- Computer Vision
- Medical Imaging
- Explainable AI

---

# 🚀 Future Improvements

Planned future work includes:

- Vision Transformer (ViT) comparison.
- EfficientNet implementation.
- Multi-label disease classification.
- Web deployment using Streamlit.
- Docker deployment.
- Clinical evaluation on additional datasets.

---

# 👨‍💻 Author

## Aman

**B.Tech Computer Science & Engineering (AKTU)**

**First Author — IEEE CE2CT 2026**

AI Research Enthusiast | Computer Vision | Medical AI | Explainable AI

### Connect with Me

- GitHub: **amansaifi699**
- LinkedIn: **aman-b8808924a**
- Email: **amanshooter4@gmail.com**

---

# 🙏 Acknowledgements

Special thanks to:

- TensorFlow
- Keras
- OpenCV
- Google Colab
- COVID-19 Radiography Database
- IEEE CE2CT 2026 Conference
- IMS Engineering College

---

# 📜 License

This project is released under the **MIT License** for academic and research purposes.

---

## ⚠️ Disclaimer

This project is intended **only for educational and research purposes**.

It is **not** a clinical diagnostic tool and should **not** be used as a substitute for professional medical advice or healthcare diagnosis.
