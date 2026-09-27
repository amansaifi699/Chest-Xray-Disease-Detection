<p align="center">
  <img src="assets/github_banner.png" width="100%" alt="Chest X-ray Disease Detection Banner">
</p>
# 🫁 Chest X-ray Disease Detection using Fine-Tuned ResNet50 with Explainable AI
<p align="center">

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-Deep_Learning-D00000?style=for-the-badge&logo=keras)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer_Vision-5C3EE8?style=for-the-badge&logo=opencv)
![Grad-CAM](https://img.shields.io/badge/Grad--CAM-Explainable_AI-purple?style=for-the-badge)

</p>

### Multi-Class Chest X-ray Classification using Transfer Learning and Grad-CAM

*COVID-19 • Lung Opacity • Viral Pneumonia • Normal*

**First Author Research Project | IEEE CE2CT 2026 Proceedings**

</div>

---

## Project Highlights

* Fine-tuned **ResNet50** for multi-class chest X-ray disease classification.
* **Transfer Learning** using ImageNet pretrained weights.
* **Grad-CAM** visualization for explainable medical AI.
* Evaluated on the **COVID-19 Radiography Database** containing **21,165** chest X-ray images.
* Accepted in the **2026 Second International Conference on Advances in Computer Science, Electrical, Electronics and Communication Technologies (CE2CT 2026)**, with proceedings accepted for publication in **IEEE Xplore**.

## Overview

Chest X-ray interpretation plays a crucial role in the early diagnosis of pulmonary diseases such as COVID-19 and pneumonia. This repository presents the implementation of my research project on **multi-class chest X-ray disease classification** using a **fine-tuned ResNet50** deep learning model integrated with **Gradient-weighted Class Activation Mapping (Grad-CAM)** for visual interpretability.

The model classifies chest radiographs into four clinically relevant categories:

* COVID-19
* Lung Opacity
* Viral Pneumonia
* Normal

Unlike conventional convolutional neural networks, the proposed transfer learning approach leverages pretrained ResNet50 features and provides interpretable heatmaps highlighting image regions that influence predictions. The project combines classification accuracy with explainable AI to improve transparency in medical image analysis.

