# Spam Email Detection using Machine Learning

## Overview
This project is a machine learning application that classifies email messages as **Spam** or **Ham (Not Spam)** using Logistic Regression and TF-IDF Vectorization.

## Features
- Load and preprocess email data
- Convert text into numerical features using TF-IDF
- Train a Logistic Regression model
- Predict whether an email is Spam or Ham
- Save the trained model using Joblib

## Technologies Used
- Python
- Pandas
- Scikit-learn
- Joblib

## Project Structure

```
Spam-Email-Detection/
├── data/
│   └── spam.csv
├── models/
│   ├── spam_model.pkl
│   └── vectorizer.pkl
├── src/
│   └── spam_detector.py
├── README.md
├── requirements.txt
└── .gitignore
```

## Installation

```bash
pip install -r requirements.txt
```

## Run the Project

```bash
python src/spam_detector.py
```

## Model Accuracy

**96.77%**

## Author

**Yeshi Chophel**