# Multimodal Emotion-Aware Mental Health Support System

This project presents a **dual-system AI framework** designed to provide emotionally intelligent mental health support by combining **text-based emotion recognition** and **facial emotion recognition (FER)**.

The system analyzes user input in real-time and generates **empathetic, context-aware responses** while maintaining **user safety and privacy**.

---

## Features

- **Text Emotion Recognition**
  - Transformer-based model (DistilRoBERTa)
  - Trained on GoEmotions dataset (7 emotion classes)

- **Facial Emotion Recognition (FER)**
  - CNN-based model (VGG19)
  - Trained on FER-2013 and evaluated on CK+

- **Multimodal Fusion**
  - Confidence-based late fusion of text and facial predictions

- **Empathetic Response Generation**
  - Template-based + CBT-inspired responses

-  **Safety Mechanism**
  - Basic crisis detection (e.g., distress/suicidal signals)

-  **Full Stack System**
  - Frontend: React + Tailwind CSS  
  - Backend: Node.js / Python API  
  - Real-time interaction with webcam input

---

## System Architecture

1. User input (text + optional camera image)
2. Text emotion prediction
3. Facial emotion prediction
4. Fusion of both modalities
5. Response generation
6. Safety check before output

---

## Important Note

 **Trained models are NOT included in this repository**  
Due to large file sizes, model files have been excluded.

You can:
- Train models using the provided scripts  
- Or load your own pretrained models

---

## Datasets Used

- **GoEmotions** – Text emotion classification  
- **FER-2013** – Facial emotion training  
- **CK+** – Facial emotion evaluation  
- **EmpatheticDialogues** – Response generation 

---

## Tech Stack

- Python (PyTorch, Transformers, OpenCV)
- React + Tailwind CSS
- Node.js / Express

---

## Disclaimer

This system is designed for **educational and research purposes only**.  
It is **NOT a replacement for professional mental health services**.

---

## Author

**Seth Rajarathne**  
BSc (Hons) Artificial Intelligence & Data Science  

---

## Supervisor

**Ms. Kalhari Walawage**  
---
