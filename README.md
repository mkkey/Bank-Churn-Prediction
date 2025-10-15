# Predicting Customer Churn: A Machine Learning Project for a bank

## Project Overview

This project builds a machine learning model to help a bank identify customers who are likely to leave. Since retaining customers is more cost-effective than acquiring new ones, predicting churn is a key priority. The goal is to develop a classification model with an **F1 score of at least 0.59**, while also monitoring **AUC-ROC**, **precision**, and **recall** due to the class imbalance.

## Project Objectives

1. Explore and clean the customer dataset.
2. Handle missing values and check for duplicates.
3. Analyze class imbalance and prepare data for modeling.
4. Train and compare multiple models:
   - Logistic Regression (baseline)
   - Logistic Regression with class weights
   - Logistic Regression with upsampling
   - Random Forest (with tuning)
5. Evaluate models on validation and test sets.
6. Choose the best model for deployment and explain next steps.

## Features

### 1. Data Preparation
- Load and inspect dataset (`Churn.csv`)
- Handle missing values in `Tenure` via:
  - Median imputation
  - Row dropping
- Drop non-predictive identifiers (`RowNumber`, `CustomerId`, `Surname`)
- Encode categorical features:
  - `Gender`: binary
  - `Geography`: one-hot
- Confirm class balance and feature scaling

### 2. Modeling and Evaluation
- Baseline: Logistic Regression (no imbalance fix)
- Class Balancing:
  - `class_weight='balanced'` (Logistic Regression)
  - Upsampling (minority class)
- Tree-Based Model:
  - Random Forest with `n_estimators` and `max_depth` tuning
- Metrics used:
  - **F1 Score**
  - **AUC-ROC**
  - **Precision**
  - **Recall**

### 3. Final Model Performance

| Model                       | F1 Score | AUC-ROC | Precision | Recall |
|----------------------------|----------|---------|-----------|--------|
| Logistic Regression (raw)  | 0.321    | 0.787   | 0.672     | 0.211  |
| + Class Weight             | 0.511    | 0.792   | 0.396     | 0.722  |
| + Upsampling               | 0.513    | 0.792   | 0.396     | 0.727  |
| Random Forest (validation) | 0.623    | 0.863   | 0.661     | 0.590  |
| 🎯 Final Model (test)       | 0.602    | 0.864   | 0.673     | 0.545  |

## Technologies Used

- **Python**: Main programming language  
- **pandas / numpy**: Data manipulation  
- **scikit-learn**: ML models and metrics  
- **plotly**: Interactive visualizations  
- **Jupyter Notebook**: Environment  

## Dataset

- `Churn.csv`: Includes 10,000 customer records with the following fields:
  - `CreditScore`, `Geography`, `Gender`, `Age`, `Tenure`, `Balance`, `NumOfProducts`, `HasCrCard`, `IsActiveMember`, `EstimatedSalary`
  - `Exited` (target): 1 = churned, 0 = stayed

## Key Insights

- **Age**, **Balance**, and **Activeness** were key churn drivers.
- Customers from **Germany** had the highest churn rate.
- **Random Forest** outperformed logistic models, even with imbalance handling.
- The final model meets the F1 requirement and generalizes well to unseen data.

## Recommendations

1. **Prioritize retention outreach** to:
   - German customers
   - Customers aged 40+
   - Customers with high balances and inactive usage patterns

2. **Launch early warning systems**:
   - Track balance drops and activity dips
   - Use predicted churn scores to trigger proactive engagement

3. **Review onboarding and support for German customers**:
   - Consider regional support enhancements or satisfaction surveys

4. **Explore loyalty or reward programs** to retain at-risk segments

5. **Monitor feature performance**:
   - Build dashboards to continuously observe churn-related patterns (especially geography, age, and activeness)

## Project Structure
- `notebook/`
    - `bank_churn_model.ipynb`: Main notebook with full analysis and modeling.
- `README.md`: Project summary and insights.



