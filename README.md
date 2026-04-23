# SMS Spam Detection using Machine Learning

A simple machine learning project that classifies SMS messages as **Spam** or **Ham (Not Spam)** using **Naive Bayes** and **TF-IDF Vectorization**. Because apparently humans needed a model to tell them that "Congratulations! You won 10 crores" is suspicious.

## 📌 Project Overview

This notebook builds a text classification model using the popular SMS Spam Collection dataset. The model is trained to predict whether a message is spam or legitimate.

## 🚀 Features

- Load and preprocess SMS dataset
- Convert text into numerical features using TF-IDF
- Train model using Multinomial Naive Bayes
- Evaluate model performance
- Display confusion matrix
- Predict custom user-entered messages

## 🛠️ Technologies Used

- Python
- Pandas
- Scikit-learn
- Matplotlib
- Jupyter Notebook

## 📂 Dataset

Dataset used: **SMS Spam Collection Dataset**

Source:  
https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv

## ⚙️ Workflow

1. Import required libraries  
2. Load dataset  
3. Encode labels (`ham = 0`, `spam = 1`)  
4. Split data into training and testing sets  
5. Apply TF-IDF Vectorizer  
6. Train Naive Bayes model  
7. Test accuracy and generate reports  
8. Predict new messages  

## 📊 Model Evaluation

The notebook includes:

- Accuracy Score  
- Classification Report  
- Confusion Matrix Visualization  

## ▶️ How to Run

1. Clone this repository:

```bash
git clone https://github.com/katheeja930/Email_Spam_Detection.git
