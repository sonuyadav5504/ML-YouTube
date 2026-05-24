# Naive Bayes, GDA, LDA and QDA From Scratch

This project implements probabilistic classification algorithms from scratch and deploys them using Streamlit.

## Dataset

The project uses the Wine dataset from sklearn.

The dataset contains chemical properties of wine samples and the goal is to classify each wine into one of three classes.

## Algorithms Implemented

### 1. Gaussian Naive Bayes

Naive Bayes assumes that features are conditionally independent given the class.

### 2. GDA / LDA

GDA with a shared covariance matrix is equivalent to Linear Discriminant Analysis.

Each class has a different mean vector, but all classes share one covariance matrix.

### 3. QDA

Quadratic Discriminant Analysis uses a separate covariance matrix for each class.

This makes QDA more flexible than LDA.

## Project Structure

```text
naive-bayes-gda-lda-qda-streamlit/
│
├── models.py
├── train_model.py
├── app.py
├── requirements.txt
└── README.md