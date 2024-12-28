# Fake Currency Prediction

This project focuses on building a Machine Learning classification model to predict whether a given currency note is fake or real. The models used in this project are **Logistic Regression** and **Support Vector Machines (SVM)**. The dataset consists of various features extracted from images of currency notes.

## Overview

Fake currency is a significant issue that can harm the economy. This project aims to classify currency notes as real or fake using machine learning algorithms. The project uses supervised learning models such as Logistic Regression and Support Vector Machines (SVM) for classification.

---

## Features

1. **Feature Extraction**: The dataset contains numerical features extracted from images of currency notes.
2. **Model Training**: Logistic Regression and SVM are implemented and trained on the dataset.
3. **Performance Metrics**: Evaluate models using metrics such as accuracy, precision, recall, and F1-score.
4. **Visualization**: Visualize the performance using confusion matrices and other plots.

---

## Installation

Follow the steps below to set up the project locally:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/fake-currency-prediction.git
   cd fake-currency-prediction
   ```

2. **Install Dependencies**:
   Use `pip` to install the required Python libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. **Dataset**:
   Place the dataset file (e.g., `bank_notes.csv`) in the project directory.

---

## Usage

1. **Preprocess the Data**:
   Run the preprocessing script to clean and prepare the data:
   ```bash
   python preprocess.py
   ```

2. **Train the Models**:
   Train the Logistic Regression and SVM models using the training script:
   ```bash
   python train_models.py
   ```

3. **Evaluate the Models**:
   Evaluate model performance using the evaluation script:
   ```bash
   python evaluate.py
   ```

4. **Prediction**:
   Use the trained models to make predictions:
   ```bash
   python predict.py --input test_data.csv
   ```

---

## Technologies Used

- **Programming Language**: Python
- **Libraries**:
  - NumPy
  - Pandas
  - Scikit-learn
  - Matplotlib
  - Seaborn
