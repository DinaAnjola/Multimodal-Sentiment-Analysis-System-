# Multimodal-Sentiment-Analysis-System-
Multimodal Sentiment Analysis System that combines textual audio and visual cues. Implements BERT, LIBROSA and ResNet50, confidence thresholding and ambivalence flagging. 


# Multimodal Sentiment Analysis System with Emotional Intelligence Enhancements

This repository contains the implementation code, resources, and configurations for the multimodal sentiment analysis (MSA) system developed for academic research purposes.

The system combines textual, audio, and visual modalities using deep learning techniques. It also incorporates confidence thresholding and ambivalence flagging mechanisms to improve emotional interpretability and robustness.


## 📋 Project Overview

- Dataset: Multimodal EmotionLines Dataset (MELD)
- Modalities: Text, Audio, Video
- Main Techniques:
  - Text embeddings from BERT
  - Audio features from **MFCCs** via **Librosa**
  - Visual features from **ResNet50**
  - Early fusion of features
  - Custom Multilayer Perceptron (MLP) for classification
  - Emotional Intelligence modules:
    - Confidence Thresholding
    - Ambivalence Flagging

---

## ⚙️ Environment Setup

The model was implemented and tested in **Google Colab** with **GPU** enabled.

### Requirements:
- Python 3.11+
- PyTorch
- Hugging Face Transformers
- Librosa
- MoviePy
- OpenCV
- Scikit-learn
- Matplotlib
- Pandas, NumPy

To install all necessary dependencies:
```bash
pip install -r requirements.txt




📂 Dataset Preparation

1. Download the MELD dataset from its official repository:
https://github.com/declare-lab/MELD


2. Extract the files and organize them as follows:



dataset/
├── train/
│   ├── audio/ (.wav files)
│   ├── video/ (.mp4 files)
│   └── text/ (transcripts)
├── dev/
├── test/

Ensure the correct paths are set in the code (config.py or inside Colab notebooks).




🚀 Running the Model (Step-by-Step)

1. Preprocessing

Extract text embeddings (BERT)

Extract MFCC audio features

Extract visual features from video frames


2. Feature Fusion

Features from the three modalities are concatenated into a unified vector.


3. Model Training

Run the training notebook:

python train_multimodal_model.py

4. Evaluation

Run the evaluation script to generate performance metrics:

python evaluate_model.py




🧠 Emotional Intelligence Features

✅ Confidence Thresholding:

Flags predictions with low confidence (below softmax threshold of 0.6).

✅ Ambivalence Flagging:

Flags cases with overlapping emotions by analyzing the difference between the top-2 predicted class probabilities.




📊 Expected Results

Metric	Value (%)

Accuracy	60
Macro F1-Score	39
Weighted F1	59


> Note: Results may slightly vary due to random initialization or environment changes.






📌 Notes on Reproducibility

Ensure random seeds are fixed in your environment for best reproducibility.

Training and inference were performed on Google Colab (with GPU).

All hyperparameters and key settings are documented inside the scripts.

Full source code, logs, and outputs are provided for full traceability.





📄 Citation

If you use this repository or its ideas for academic purposes, kindly cite:

> Building a Multimodal Sentiment Analysis System that combines textual, audio and visual cues.
Researcher: Dina Anjolaoluwa Afolashade 
Supervisor: Prof. M.K. Aregbesola
Institution: Caleb University, Lagos.





 Contact

For questions or clarifications:

dina.anjolaoluwa@calebuniversity.edu.ng

GitHub Issues on this repo





