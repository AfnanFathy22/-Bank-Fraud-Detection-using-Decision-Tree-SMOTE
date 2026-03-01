# 💳 Bank Fraud Detection using Decision Tree + SMOTE

This mini project focuses on detecting fraudulent bank transactions using a Decision Tree classifier. The dataset used is the well-known credit card fraud detection dataset (creditcard.csv). The main objective is to classify transactions into two categories: 0 → Normal Transaction and 1 → Fraudulent Transaction.

## 📌 Project Objective
The goal is to build a machine learning model capable of identifying fraudulent transactions in a highly imbalanced dataset. The project demonstrates data exploration, handling class imbalance, model building, evaluation, and hyperparameter tuning.

## 📊 Exploratory Data Analysis (EDA)

Task EDA-1: The dataset shape and first 5 rows were printed to understand the number of samples and features.

Task EDA-2: Missing values were checked to ensure data integrity.

Task EDA-3: Class imbalance was visualized using a count plot to compare Fraud vs Normal transactions. The dataset was found to be extremely imbalanced, with fraudulent transactions representing a very small percentage of total transactions.

Task EDA-4: The distribution of the transaction Amount feature was explored using histograms to understand transaction value patterns.

EDA Summary:
The dataset contains a very high imbalance between normal and fraudulent transactions. Fraud cases represent only a tiny fraction of total observations. No significant missing values were found. The Amount distribution is highly skewed, with most transactions being low-value amounts. This imbalance makes fraud detection a challenging classification problem.

Task EDA-5: Accuracy is not reliable for this dataset because a model that predicts all transactions as Normal could still achieve very high accuracy due to the severe class imbalance. Therefore, metrics such as Precision, Recall, F1-Score, and AUC are more appropriate.

## 🧹 Data Preparation
- Handled any rows containing NaN values in the target column 'Class'.
- Split the dataset into training and testing sets using stratified sampling to maintain class distribution.
- Features and target variable were separated properly.

## 🌳 Baseline Model: Decision Tree (Without SMOTE)
A Decision Tree classifier was trained on the original imbalanced dataset. Evaluation metrics such as Precision, Recall, F1-score, and Confusion Matrix were calculated. The model showed high accuracy but relatively lower recall for fraudulent transactions.

## 🔄 Applying SMOTE + Decision Tree
SMOTE (Synthetic Minority Oversampling Technique) was applied to balance the dataset by generating synthetic minority class samples. A new Decision Tree model was trained on the resampled dataset.

### Why SMOTE is Not Always a Good Practice with Decision Trees
Decision Trees are non-parametric models that are already capable of handling imbalanced datasets to some extent by adjusting split criteria. Applying SMOTE may introduce synthetic data points that can cause overfitting, especially since Decision Trees can perfectly fit duplicated or synthetic patterns. As a result, performance may improve on training data but not generalize well to unseen data.

## 📈 Model Evaluation
- Confusion Matrix
- Precision, Recall, F1-Score
- Area Under the ROC Curve (AUC)
The ROC curve was plotted and AUC was calculated to evaluate the model’s ability to distinguish between fraudulent and normal transactions.

## ⚙️ Hyperparameter Tuning
Hyperparameter tuning was performed to improve model performance. Parameters such as:
- max_depth
- min_samples_split
- min_samples_leaf
- criterion
were tested using cross-validation techniques to find the optimal configuration.

## 🛠️ Technologies Used
Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Imbalanced-learn (SMOTE).

## 🎯 Skills Demonstrated
Exploratory Data Analysis, Handling Imbalanced Data, Decision Tree Modeling, SMOTE Application, Model Evaluation using Advanced Metrics, ROC-AUC Analysis, Hyperparameter Tuning, and Critical Evaluation of Resampling Techniques.
