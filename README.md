# Student Depression Prediction Project

This project aims to predict student depression based on various academic, lifestyle, and demographic factors using machine learning techniques. The workflow includes data exploration, preprocessing, and the implementation of multiple ML models.

## Table of Contents
- [Data Exploration](#data-exploration)
- [Data Preprocessing](#data-preprocessing)
- [Machine Learning Models](#machine-learning-models)

## Data Exploration

### Dataset Overview
- **Source**: `student_depression_prediction.csv`
- **Features**: 
    `id`
    `Gender`
    `Age`
    `City`
    `Profession`
    `Academic Pressure`
    `Work Pressure` 
    `CGPA` 
    `Study Satisfaction` 
    `Job Satisfaction` 
    `Sleep Duration` 
    `Dietary Habits` 
    `Degree`
    `Have you ever had suicidal thoughts ?` 
    `Work/Study Hours` 
    `Financial Stress` 
    `Family History of Mental Illness`

- **Target Variable**: `Depression`

### Data Exploration
- **Visualizations**:
  - Correlation matrix to identify feature relationships.
  - Box plots, histograms, and violin plots to explore distributions and relationships with depression.

## Data Preprocessing

### Steps
1. **Handling Missing Data**: Dropped rows with null values.
2. **Removing Duplicates**: Dropped duplicate entries.
3. **Outlier Removal**: Applied IQR to filter outliers in numerical features.
4. **Feature Engineering**:
   - Dropped `Work Pressure` due to multicollinearity.
   - Encoded categorical variables using LabelEncoder.
5. **Scaling**: Standardized features using StandardScaler.
6. **Class Imbalance**: Addressed using SMOTE on the training set.

### Dataset Splitting
- **Train-Test Split**: 80% training, 20% testing.

## Machine Learning Models

### Models Implemented
1. **Logistic Regression**
   - Configuration: ElasticNet penalty (`l1_ratio=0.5`), `saga` solver.
   - Metrics: Accuracy, Precision, Recall, F1.

2. **K-Nearest Neighbors**
   - Hyperparameter tuning using `GridSearchCV` for `n_neighbors`, `weights`, and `p`.
   - Best parameters selected based on F1 score.

3. **Decision Tree**
   - Configuration: `entropy` criterion, `max_depth=10`, `class_weight='balanced'`.

4. **Random Forest**
   - Configuration: `n_estimators=100`, `random_state=1217`.

5. **Support Vector Machine**
   - Kernel: Linear.

6. **Neural Networks**
   - **Architectures**:
     - **Baseline**: 1 hidden layer (32 units).
     - **Deep**: 3 hidden layers (64, 32, 16 units) with dropout.
     - **Wide**: 1 hidden layer (128 units) with dropout.
   - **Training**: Early stopping, Adam optimizer, binary cross-entropy loss.
   - **Metrics**: Accuracy, Precision, Recall, AUC-ROC.

### Evaluation Metrics
- **Common Metrics**: Accuracy, Precision, Recall, F1, Confusion Matrix.
- **Neural Networks**: Additional AUC-ROC curves and training history plots.