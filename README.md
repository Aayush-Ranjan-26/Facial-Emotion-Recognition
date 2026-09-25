<div align="center">

  <h1>🎭 Real-Time Facial Emotion Recognition</h1>

  <p>
    <strong>A high-performance, CPU-optimized facial expression analysis engine built with classical Machine Learning</strong>
  </p>

  <p>
    <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"></a>
    <a href="https://opencv.org/"><img src="https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV"></a>
    <a href="https://scikit-learn.org/"><img src="https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-Learn"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-2ea44f?style=for-the-badge" alt="License: MIT"></a>
  </p>

  <p>
    <b>⚡ High-FPS Live Inference</b> &bull;
    <b>🎯 Haar + HOG + SVM Pipeline</b> &bull;
    <b>💻 Zero GPU Required</b>
  </p>

</div>

A lightweight, real-time facial expression and emotion recognition system built with **classical Machine Learning**. By leveraging **Haar Cascade classifiers**, **Histogram of Oriented Gradients (HOG)**, and a **Support Vector Machine (SVM)** with an RBF kernel, this project achieves responsive emotion detection on standard CPU hardware without requiring heavy deep learning frameworks or dedicated GPUs.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [System Architecture](#-system-architecture)
- [Emotion Classes](#-emotion-classes)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Execution & Usage](#-execution--usage)
  - [1. Real-Time Application](#1-real-time-application)
  - [2. Diagnostic Self-Check](#2-diagnostic-self-check)
- [Pipeline Details](#-pipeline-details)
- [Performance & Configuration](#-performance--configuration)
- [Acknowledgments](#-acknowledgments)

---

## 📖 Overview

While modern facial analysis frequently relies on deep Convolutional Neural Networks (CNNs), they typically require high computational overhead and GPU acceleration. This system demonstrates that a carefully tuned classical ML pipeline can deliver robust, low-latency live inferences directly on edge CPUs:

- **⚡ Low Latency & High Frame Rate**: Runs smoothly on standard CPU architectures with minimal memory footprint.
- **🎯 Robust Feature Extraction**: Employs Histogram of Oriented Gradients (HOG) to capture structural facial cues (eyebrow curvature, mouth shape) while discarding lighting and skin-tone variations.
- **⚖️ Balanced Multi-Class Classification**: Utilizes an RBF-kernel SVM with class-weight compensation to counteract dataset imbalances (such as underrepresented *disgust* samples).
- **💾 Automated Persistence**: Trains once, evaluates comprehensive classification metrics, and caches the serialized model artifact for instant subsequent startups.

---

## 🔄 System Architecture

The end-to-end processing pipeline operates in five cohesive stages:

```
┌─────────────────┐       ┌───────────────────────┐       ┌────────────────────────┐
│  Webcam Stream  │ ───►  │ Face Detection (Haar) │ ───►  │ Preprocessing (48x48) │
└─────────────────┘       └───────────────────────┘       └────────────────────────┘
                                                                       │
                                                                       ▼
┌─────────────────┐       ┌───────────────────────┐       ┌────────────────────────┐
│ Real-Time HUD   │ ◄───  │    SVM Classifier     │ ◄───  │ HOG Feature Extraction │
│ (Bounding Box)  │       │     (RBF Kernel)      │       │     (9 Orientations)   │
└─────────────────┘       └───────────────────────┘       └────────────────────────┘
```

1. **Frame Acquisition**: Captures live video stream at native camera frame rate.
2. **Face Localization**: Rapid multi-scale face detection using OpenCV's built-in Haar Cascade.
3. **Region Normalization**: Converts cropped face ROIs to single-channel grayscale, resizes to a standardized $48 \times 48$ matrix, and scales pixel intensities to $[0.0, 1.0]$.
4. **Descriptor Computation**: Generates compact gradient vectors via HOG ($9$ orientations, $6 \times 6$ pixels/cell, $2 \times 2$ cells/block).
5. **Inference & Visualization**: Evaluates feature vectors against the trained SVM and renders real-time bounding boxes with predicted emotion labels.

---

## 🏷️ Emotion Classes

The model is trained on the benchmark **FER-2013** dataset and recognizes **7 universal emotional states**:

| Class | Expression | Key Visual Features Identified by HOG |
| :--- | :---: | :--- |
| **Angry** | 😠 | Furrowed brow, narrowed eyelids, pressed lips |
| **Disgust** | 🤢 | Wrinkled nose, raised upper lip, narrowed eyes |
| **Fear** | 😨 | Raised eyebrows drawn together, widened eyes |
| **Happy** | 😊 | Elevated cheekbones, upturned lip corners, crow's feet |
| **Neutral** | 😐 | Relaxed facial musculature, horizontal lip line |
| **Sad** | 😢 | Drooping eyelid corners, downturned mouth corners |
| **Surprise** | 😲 | High-arched eyebrows, widened eyes, dropped jaw |

---

## 📁 Project Structure

```text
Facial-Emotion-Recognition/
├── data/
│   ├── train/                      # Training split organized by emotion category
│   │   ├── angry/
│   │   ├── disgust/
│   │   └── ...
│   └── test/                       # Evaluation split organized by emotion category
├── .gitignore                      # Git exclusion rules
├── emotion_model.pkl               # Serialized trained SVM classifier (generated)
├── emotion_recognition.py          # Unified application entry point & ML pipeline
├── Real-Time-...-Deep-Learning.pdf # Reference research documentation
├── README.md                       # Repository documentation
└── requirements.txt                # Python dependencies
```

---

## 🚀 Getting Started

### Prerequisites

- **Python**: Version `3.8` or newer installed.
- **Hardware**: Any standard integrated or external USB webcam.

### Installation

1. **Clone the repository** (or extract to your local workspace):
   ```bash
   git clone https://github.com/Aayush-Ranjan-26/Facial-Emotion-Recognition.git
   cd Facial-Emotion-Recognition
   ```

2. **Create and activate a virtual environment**:
   - **Windows (PowerShell)**:
     ```powershell
     python -m venv .venv
     .venv\Scripts\Activate.ps1
     ```
   - **Linux / macOS**:
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 💻 Execution & Usage

### 1. Real-Time Application

Launch the primary application using the virtual environment:

```bash
# Windows
.venv\Scripts\python emotion_recognition.py

# Linux / macOS
python emotion_recognition.py
```

- **First Run Behavior**:
  - Automatically indexes `data/train` and `data/test`.
  - Extracts HOG descriptors across training samples.
  - Trains the Support Vector Classifier (`SVC`).
  - Outputs a complete evaluation report (accuracy, precision, recall, F1-score, and confusion matrix).
  - Serializes and saves the weights to `emotion_model.pkl`.
  - Immediately launches the live webcam window.

- **Subsequent Runs**:
  - Detects existing `emotion_model.pkl`.
  - Skips dataset loading and model training.
  - Instantly opens the camera stream for real-time classification.

- **Controls**:
  - Focus on the display window and press **`Q`** at any time to release the camera and exit cleanly.

### 2. Diagnostic Self-Check

Run a headless verification check to validate that image transformation, resizing, and HOG extraction routines are functioning as expected without needing dataset access or a webcam:

```bash
# Windows
.venv\Scripts\python emotion_recognition.py --test

# Linux / macOS
python emotion_recognition.py --test
```

---

## ⚙️ Performance & Configuration

Key configuration parameters can be tuned directly within [emotion_recognition.py](emotion_recognition.py):

- **`MAX_PER_CLASS` (Default: `1200`)**:
  - *Context*: SVM training complexity scales quadratically ($O(N^2)$) with the number of samples when using non-linear kernels.
  - *Recommendation*: The default limit balances high classification fidelity with a modest training duration (~2–4 minutes on modern quad-core CPUs). If compute time permits, this value can be increased to utilize the full ~28,700 training image corpus.
- **`IMG_SIZE` (Default: `48`)**:
  - Matches the standard FER-2013 resolution ($48 \times 48$).
- **Haar Cascade Source**:
  - Utilizes OpenCV's pre-packaged frontal face model (`cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'`), removing any need to manually download external weights.

---

## 📊 Evaluation & Diagnostics

During initial training, the script performs evaluation against the dedicated holdout test set (`data/test/`) and outputs:
- **Overall Test Accuracy**
- **Per-Class Precision, Recall, and F1-Scores**
- **Confusion Matrix** (visualizing cross-class misclassification patterns)

---

## 🤝 Acknowledgments

- **FER-2013 Dataset**: Challenges in Representation Learning (ICML 2013).
- **OpenCV & Scikit-Learn**: Open-source machine learning and computer vision ecosystems enabling high-efficiency edge analytics.
