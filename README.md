![BraTS - Brain Tumor Segmentation and Detection](brats-banner.png)

# 🧠 Brain Tumor Segmentation and Detection using BraTS MRI Dataset

## 📌 Project Overview

This project focuses on **brain tumor segmentation and detection** from medical MRI images using deep learning techniques. It is based on the **BraTS MRI dataset**, widely used for brain tumor analysis and medical image segmentation research.

The project implements a complete workflow, including MRI preprocessing, tumor segmentation using a **U-Net architecture**, tumor detection using **transfer learning**, and model evaluation with visualizations.

## 🎯 Objectives

- Preprocess MRI brain images from the BraTS dataset.
- Extract relevant MRI slices for tumor analysis.
- Build and train a U-Net model for brain tumor segmentation.
- Apply transfer learning for tumor detection.
- Evaluate model performance using quantitative metrics and visual results.
- Analyze computational impact using emissions tracking.

## 🧬 Dataset

The project uses the **BraTS Brain Tumor Segmentation Dataset**, which contains multimodal MRI scans for brain tumor segmentation.

Typical MRI modalities include:

- **T1:** Structural brain imaging.
- **T1ce:** T1-weighted imaging with contrast enhancement.
- **T2:** Imaging that highlights tissue characteristics and fluid.
- **FLAIR:** Imaging useful for identifying abnormalities around tumors.
- **Segmentation masks:** Ground-truth annotations identifying tumor regions.

The dataset is processed to extract relevant MRI slices and their corresponding segmentation masks.

## 🏗️ Project Structure

```text
Brain-Tumor-Segmentation-Detection-MRI/
│
├── brats-banner.png
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
├── model.pth
│   └── Trained model checkpoint
│
├── best_model.pth
│   └── Best trained model weights
│
└── README.md
```

## 🔬 Methodology

### 1. MRI Preprocessing

- Load MRI volumes and segmentation masks.
- Select relevant two-dimensional slices.
- Normalize image intensities.
- Prepare images and masks for model training.

### 2. Brain Tumor Segmentation — U-Net

U-Net is a convolutional neural network designed for biomedical image segmentation. It predicts a segmentation mask that identifies the tumor region at the pixel level.

### 3. Tumor Detection — Transfer Learning

Transfer learning reuses knowledge from a pretrained deep learning model and adapts it to the tumor detection task.

### 4. Evaluation and Visualization

Evaluate the models using appropriate metrics and visualize predictions alongside the original MRI images and ground-truth masks.

Possible segmentation metrics include:

- Dice Similarity Coefficient (Dice).
- Intersection over Union (IoU).
- Precision and recall.

### 5. Computational Impact

Analyze computational performance and, where configured, estimate the environmental impact of model execution using emissions-tracking tools.

## 🛠️ Technologies Used

- **Python**
- **PyTorch**
- **Jupyter Notebook**
- **NumPy**
- **Matplotlib**
- **Medical MRI image processing**
- **U-Net**
- **Transfer Learning**

## 🚀 Getting Started

### Prerequisites

Install Python and the required project dependencies.

### Installation

```bash
git clone https://github.com/MohamedYislim/Brain-Tumor-Segmentation-Detection-MRI.git

cd Brain-Tumor-Segmentation-Detection-MRI

pip install torch torchvision numpy matplotlib jupyter
```

Install any additional dependencies required by the notebooks.

### Run the Project

Execute the notebooks in the following order:

1. `01_Pre-traitement.ipynb`
2. `02_Segmentation_UNet.ipynb`
3. `03_Detection_TransferLearning.ipynb`
4. `04_Evaluation_et_Visualisation.ipynb`

**Note:** The repository structure, filenames, dependencies, and execution commands should match the actual contents of the repository.

## 📊 Expected Results

The project aims to produce:

- Preprocessed MRI slices.
- Predicted brain tumor segmentation masks.
- Tumor detection predictions.
- Quantitative model evaluation metrics.
- Visual comparisons between MRI images, ground-truth masks, and model predictions.

## ⚠️ Disclaimer

This project is intended for educational and research purposes. Its predictions are not a substitute for professional medical diagnosis.

## 👨‍💻 Author

**Mohamed Yislim**

- GitHub: [MohamedYislim](https://github.com/MohamedYislim)
- Portfolio: [Personal Portfolio](https://portfolio-ltuu.vercel.app/)

