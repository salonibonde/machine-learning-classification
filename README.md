
# Churn Prediction Analysis

## Overview
This repository contains the analysis and predictive modeling techniques applied to a classification problem. The project focuses on building and optimizing machine learning models to improve predictive capabilities while balancing interpretability and accuracy.

We start with interpretable models like **Logistic Regression** and progressively explore more complex models like **K-Nearest Neighbors (KNN)** and **Random Forest** to enhance prediction power. The goal is to provide stakeholders with insights into model performance, while also ensuring that results can be explained and understood by non-experts.

## Why Does This Issue Matter?
Predicting customer churn allows businesses to retain valuable customers, reducing revenue loss and costs associated with acquiring new clients. In industries like hospitality, it helps optimize resource allocation and improves customer satisfaction by mitigating cancellations proactively.

## Methods
Our approach follows a systematic exploration of different machine learning models:

- **Logistic Regression**: We use logistic regression as a baseline model. Subset selection is performed using stepwise selection and Lasso regularization.
- **K-Nearest Neighbors (KNN)**: KNN is applied with k-fold cross-validation to determine the optimal number of neighbors (K) for best accuracy.
- **Random Forest**: This ensemble learning method is applied to capture complex interactions between variables. Hyperparameter tuning is conducted to optimize performance.
- **SMOTE**: The Synthetic Minority Over-sampling Technique (SMOTE) is used to address class imbalance in the dataset, improving model performance on minority classes.
  
## Key Files

- **data**: Contains the dataset used for model training and evaluation.
- **eda.Rmd**: Exploratory Data Analysis performed to understand data patterns and relationships.
- **subsampling.Rmd**: Subsampling techniques used for balancing classes in the dataset.
- **modeling_glm_subset_selection_lasso.Rmd**: Logistic regression model using Lasso regularization for feature selection.
- **modeling_glm_subset_selection_stepwise.Rmd**: Logistic regression model using stepwise subset selection.
- **modeling_knn.Rmd**: K-Nearest Neighbors model.
- **modeling_random_forest.Rmd**: Random Forest model with hyperparameter tuning.

## Tools and Libraries
- **R**: Main programming language used for data analysis and model building.
- **Tidyverse**: For data manipulation and visualization.
- **Cluster**, **Factoextra**: Used for feature extraction and clustering.
- **SMOTE**: For handling class imbalance.
- **RandomForest**, **glmnet**: For building models such as Random Forest and Lasso regression.
  
## Results
- **Logistic Regression**: Variables like `LeadTime`, `BookingChanges`, and `DepositTypeNonRefund` were identified as significant predictors.
- **KNN**: With cross-validation, the optimal K was identified to maximize accuracy.
- **Random Forest**: Achieved the highest performance with accuracy improvements through hyperparameter tuning.
  
## How to Run the Code

1. **Clone the repository**:
   ```bash
   git clone https://github.com/salonibonde/machine-learning-classification.git
   cd machine-learning-classification
   ```

2. **Run the analysis**:
   Open the R markdown files to execute the analysis for each model. Start with the exploratory data analysis and subsampling before proceeding with modeling:
   - `eda.Rmd` for EDA
   - `modeling_glm_subset_selection_lasso.Rmd` for logistic regression with Lasso
   - `modeling_knn.Rmd` for KNN modeling

