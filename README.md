# 🧠 Alzheimer’s Disease Detection using Deep Learning (ResNet50)

A deep learning project for multi-class classification of Alzheimer’s disease stages using brain MRI images from the OASIS dataset. This project uses transfer learning with ResNet50 implemented in PyTorch.

---

## 🚀 Overview

This project focuses on classifying MRI brain scans into multiple Alzheimer’s stages:

* NonDemented
* Very Mild Demented
* Mild Demented
* Moderate Demented

The model leverages pretrained CNN features and adapts them for medical image classification.

---

## 🧠 Model Architecture

We use a pretrained **ResNet50** as a feature extractor and replace the final classification layer.

* Base Model: ResNet50 (pretrained on ImageNet)
* Custom Head:

  * Linear → ReLU → Dropout → Linear (4 classes)
* Loss Function: CrossEntropyLoss
* Optimizer: Adam

---

## 📂 Dataset

Dataset used: **OASIS Alzheimer’s Detection Dataset**

* MRI brain scan images (2D slices)
* Multi-class labels
* Dataset split carefully to avoid **data leakage (patient-wise split)**

---

## ⚙️ Project Pipeline

1. **Data Preprocessing**

   * Resize images to 224×224
   * Normalize pixel values
   * Convert grayscale → 3-channel

2. **Data Splitting**

   * Train / Validation / Test
   * Split by **patient ID** (not image)

3. **Data Augmentation**

   * Rotation
   * Horizontal Flip
   * Random transformations

4. **Model Training**

   * Freeze base layers initially
   * Train custom classifier
   * Fine-tune deeper layers

5. **Evaluation**

   * Accuracy
   * Confusion Matrix
   * Precision / Recall / F1-score

---

## 🛠️ Tech Stack

* Python
* PyTorch
* Torchvision
* NumPy
* Matplotlib

---

## 🖥️ Installation

```bash
git clone https://github.com/your-username/alzheimer-detection.git
cd alzheimer-detection

pip install -r requirements.txt
```

---

## ▶️ Usage

### Train the model

```bash
python train.py
```

### Evaluate the model

```bash
python evaluate.py
```

---

## 💾 Model Saving

Model weights are saved using:

```python
torch.save(model.state_dict(), "best_model.pth")
```

---

## 📊 Results

| Metric    | Value |
| --------- | ----- |
| Accuracy  | XX%   |
| Precision | XX    |
| Recall    | XX    |
| F1-score  | XX    |

> Replace with actual results after training.

---

## ⚠️ Limitations

* Dataset size is limited
* MRI slices treated as independent images
* Not clinically validated
* Potential class imbalance

---

## 🔍 Future Improvements

* Use 3D CNNs for volumetric data
* Apply Grad-CAM for explainability
* Improve dataset size and quality
* Deploy as a web application

---

## 📌 Important Note

This project is intended for **academic and research purposes only**.
It is **not a medical diagnostic tool**.

---

## 👨‍💻 Author

Saurabh Sharma

---

## ⭐ Acknowledgements

* OASIS Dataset
* PyTorch Community
