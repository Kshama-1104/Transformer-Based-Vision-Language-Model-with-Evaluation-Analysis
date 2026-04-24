# 🧠 Multimodal Image Captioning System

### Transformer-Based Vision-Language Model with Evaluation & Analysis

---

## 🚀 Project Demo

<p align="center">
  <img src="./assets/ui-home.png" width="700"/>
</p>

<p align="center">
  <img src="./assets/ui-result.png" width="700"/>
</p>

---

## 🚀 Overview

This project implements a **state-of-the-art multimodal AI system** that generates natural language descriptions from images.

It compares:

* Traditional **CNN + LSTM**
* Modern **Transformer-based Vision-Language Model (BLIP)**

---

## 🎯 Key Features

* Transformer-based caption generation
* Multimodal AI (Image + Text)
* Beam Search vs Greedy Decoding
* Evaluation using BLEU, CIDEr, METEOR
* Model comparison & error analysis

---

## 🧠 Skills Demonstrated

* Transformer Architecture
* Vision-Language Models
* Transfer Learning
* Sequence Modeling
* Attention Mechanism
* Model Evaluation
* Multimodal Learning

---

## 🏗️ System Architecture

<p align="center">
  <img src="./assets/architecture.png" width="700"/>
</p>

---

## 🔬 Model Architectures

### 🔹 CNN + LSTM (Baseline)

<p align="center">
  <img src="./assets/cnn-lstm.png" width="700"/>
</p>

* InceptionV3 + LSTM
* Encoder-decoder architecture

---

### 🔹 Transformer (BLIP)

<p align="center">
  <img src="./assets/transformer.png" width="700"/>
</p>

* Vision-Language Transformer
* Attention-based caption generation

---

## ⚙️ Caption Generation (Beam Search)

<p align="center">
  <img src="./assets/beam-search.png" width="700"/>
</p>

* Generates better captions than greedy decoding
* Explores multiple sequences

---

## 📊 Performance Evaluation

<p align="center">
  <img src="./assets/metrics.png" width="800"/>
</p>

| Metric | CNN + LSTM | Transformer |
| ------ | ---------- | ----------- |
| BLEU-1 | 0.70       | 0.83        |
| BLEU-4 | 0.32       | 0.42        |
| CIDEr  | 0.88       | 1.15        |
| METEOR | 0.26       | 0.31        |

---

## 📸 Qualitative Results

<p align="center">
  <img src="./assets/results.png" width="800"/>
</p>

---

## ❌ Error Analysis

<p align="center">
  <img src="./assets/error-analysis.png" width="800"/>
</p>

---

## 🔍 Attention Visualization

<p align="center">
  <img src="./assets/attention.png" width="800"/>
</p>

---

## 🧪 Ablation Study

<p align="center">
  <img src="./assets/ablation.png" width="800"/>
</p>

---

## 🏆 Comparison with State-of-the-Art

<p align="center">
  <img src="./assets/comparison.png" width="800"/>
</p>

---

## 📁 Dataset

* Flickr8K Dataset
* 8000 images
* 5 captions per image

---

## ⚙️ Tech Stack

* Python
* PyTorch / TensorFlow
* Hugging Face Transformers
* NumPy, Pandas
* Matplotlib, Seaborn

---

## 📌 Conclusion

This project demonstrates how modern transformer-based models outperform traditional architectures in multimodal AI tasks like image captioning.
