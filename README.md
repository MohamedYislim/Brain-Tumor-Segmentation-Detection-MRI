# 🧠 Brain Tumor Segmentation and Detection using BraTS MRI Dataset

## 📌 Project Overview

This project focuses on **brain tumor segmentation and detection** from medical MRI images using deep learning techniques.  
It is based on the **BraTS MRI dataset**, which is widely used for brain tumor analysis and medical image segmentation research.

The project includes a complete workflow starting from MRI preprocessing, tumor segmentation using a **U-Net architecture**, detection using **transfer learning**, and final model evaluation with visualization of results.

---

## 🎯 Objectives

The main objectives of this project are:

- Preprocess MRI brain images from the BraTS dataset.
- Extract useful image slices for tumor analysis.
- Build and train a U-Net model for brain tumor segmentation.
- Apply transfer learning for tumor detection.
- Evaluate model performance using visual and quantitative results.
- Analyze computational impact using emissions tracking.

---

## 🧬 Dataset

The project uses the **BraTS Brain Tumor Segmentation Dataset**, which contains multimodal MRI scans used for brain tumor segmentation tasks.

Typical MRI modalities include:

- **T1**
- **T1ce**
- **T2**
- **FLAIR**
- **Segmentation masks**

The dataset is organized and processed to extract relevant MRI slices containing tumor regions.

---

## 🏗️ Project Structure

```text
Brain-Tumor-Segmentation-Detection-MRI/
│
├── data/
│   └── Processed MRI images and masks
│
├── dataset/
│   └── BraTS dataset files
│
├── results/
│   └── Evaluation results and visualizations
│
├── 01_Pre-traitement.ipynb
│   └── MRI preprocessing and slice extraction
│
├── 02_Segmentation_UNet.ipynb
│   └── U-Net model implementation for tumor segmentation
│
├── 03_Detection_TransferLearning.ipynb
│   └── Tumor detection using transfer learning
│
├── 04_Evaluation_et_Visualisation.ipynb
│   └── Model evaluation and visualization of predictions
│
├── train.py
│   └── Training script
│
├── utils.py
│   └── Utility functions for preprocessing and training
│
│
├── model.pth
│   └── Trained model checkpoint
│
├── best_model.pth
│   └── Best trained model weights
│
└── README.md
