# Emotion Analysis Using Transformer Models

This project focuses on emotion classification using transformer-based models like DistilBERT. The primary objective is to build, train, and evaluate a machine learning model that predicts emotional sentiments from textual data. The implementation leverages the Hugging Face Transformers library for model handling and fine-tuning, along with robust preprocessing and evaluation techniques.

---

The model can be downloaded by [going to this Google Drive link](https://drive.google.com/drive/folders/1AQH4KZm4DSP5LCKb5f9D22S-DP7ROBbR) which can further be used for such tasks.

---

## Key Features
- **Model Used**: DistilBERT (pre-trained transformer model).
- **Task**: Sequence classification for multi-class emotion analysis.
- **Dataset**: Emotion-annotated dataset (preprocessed and tokenized using Hugging Face Datasets).
- **Metrics**: Accuracy, F1-score, and confusion matrix
- **App Integration**: A user-friendly web application built using Streamlit for live emotion predictions.

---

## Technologies and Tools
- **Python Libraries**:
  - Hugging Face Transformers: Model and tokenizer management.
  - Hugging Face Datasets: Efficient dataset handling.
  - Scikit-learn: Train-test split and additional evaluations.
  - PyTorch: Backend framework for model training.
  - Streamlit: Interactive web app development.
- **Development Environment**: Google Colab for seamless access to GPU resources.
- **Model**:
  - Pre-trained checkpoint: `distilbert-base-uncased`.
  - Fine-tuned for sequence classification using the Trainer API.

---

## Workflow Overview
1. **Dataset Preparation**:
   - Data split into train, validation, and test sets using stratification for balanced class distributions.
   - Tokenization performed using `AutoTokenizer` from the Hugging Face library.

2. **Model Setup**:
   - Pre-trained DistilBERT model loaded and fine-tuned using the `Trainer` API.
   - Custom configurations such as batch size, learning rate, and evaluation strategy were defined in `TrainingArguments`.

3. **Training and Evaluation**:
   - Fine-tuned for 2 epochs with a learning rate of `2e-5`.
   - Evaluation performed at the end of each epoch using metrics like accuracy and F1-score.

4. **Prediction**:
   - Predictions generated on the test set using `trainer.predict()`.
   - Results include logits, true labels, and calculated evaluation metrics.

5. **Streamlit App**:
   - A web application built to provide real-time emotion predictions from user-inputted text.
   - Simple, interactive interface for non-technical users to explore model capabilities.

---

## Video Demonstration
[![Emotion Analysis Demo](https://img.youtube.com/vi/6uNZ8gOPIRw/0.jpg)](https://youtu.be/6uNZ8gOPIRw)

Click the thumbnail above to watch a demo of the project in action!

---

## Outputs
- Trained model checkpoints saved in a specified directory.
- Log files containing training and evaluation details.
- Predictions on test data with associated metrics.
- A Streamlit app enabling real-time text-based emotion classification.

---

## Applications
- Sentiment analysis for social media, customer feedback, and user reviews.
- Understanding emotional trends and patterns in textual data.
- Deployment-ready app for practical demonstrations or end-user interaction.

---




