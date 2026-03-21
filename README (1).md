# Banking Loan Default Prediction

## Problem Statement

This project focuses on predicting **non-defaulters** from a banking dataset using supervised machine learning. The objective is to help financial institutions in identifying low-risk customers, improving lending decisions, and mitigating default risk.

## Dataset

The dataset used in this project is titled `HACKATHON_TRAINING_DATA.CSV`. It contains customer-level information including:

* Credit limits and loan details
* Monthly outstanding balances and debits
* Risk indicators such as CRIFF scores and repayment grades
* Account behavior trends over 12 months
* KYC status, digital banking indicators, and more

The key target variable is `TARGET`, where:

* `0` indicates a non-defaulter
* `1` indicates a defaulter

A few key columns include:
`ACCT_AGE`, `LIMIT`, `OUTS`, `LOAN_TENURE`, `INSTALAMT`, `KYC_SCR`, `CRIFF_33`, `INCOME_BAND1`, `CREDIT_HISTORY_LENGTH1`, `PRODUCT_TYPE`, `ALL_LON_LIMIT`, `LATEST_NPA_TENURE`, `NO_YRS_RG3`, and others. Monthly transactional fields are also present, such as `ONEMNTHSDR`, `TWOMNTHOUTSTANGBAL`, `THREEMNTHAVGMTD`, etc.

## Methodology

### 1. Data Cleaning and Feature Engineering

* Converted binary flags (`Y/N`) to numeric (`1/0`)
* Parsed text-based durations like `2 yrs 3 mon` into total months
* Mapped categorical features (`INCOME_BAND1`, `PRODUCT_TYPE`, etc.) to numeric codes
* Converted `TIME_PERIOD` from strings like `JAN23` to numeric `YYYYMM` format
* Engineered meaningful features:

  * `overspend_ratio`: proportion of spending above monthly credit limit
  * `max_consec_overspend`: longest streak of overspending months
  * `outbal_slope`: trend direction of outstanding balances
  * `slope_MTD`: trend in term debit activity

Unnecessary high-cardinality columns (e.g., SDR, MTD, OUTSTANGBAL series) were dropped to reduce dimensionality and noise.

### 2. Preprocessing and Transformation

* Median imputation for missing values
* Yeo-Johnson transformation for normalization
* Min-Max scaling to standardize feature ranges

### 3. Handling Class Imbalance

* Used SMOTE-Tomek (combination of oversampling and under-sampling) to balance defaulters and non-defaulters

### 4. Feature Selection Using SHAP

* Trained a preliminary XGBoost model
* Computed SHAP values to identify important features
* Selected the top 30 most impactful features for final modeling

### 5. Model Training and Evaluation

**Model 1: XGBoost**

* Hyperparameter tuning using grid search
* Evaluated on test data using F1-score, precision, recall, and confusion matrix

**Model 2: LightGBM**

* Trained on SHAP-selected features
* Applied internal class balancing using `class_weight='balanced'`
* Optimized decision threshold based on precision-recall curve to improve F1-score

## Evaluation Metrics

* Accuracy
* F1-Score
* Precision and Recall
* Confusion Matrix
* Precision-Recall AUC and threshold tuning

## Prototype Interface

To demonstrate the practical application of our solution, we developed a **prototype user interface**. The interface was built to simulate how bank employees can use the model in a real-world scenario. Features of the prototype include:

* Easy input of customer financial and behavioral attributes
* Instant prediction of defaulter or non-defaulter status
* Support for credit officers to quickly evaluate loan eligibility
* Helpful summary of key risk indicators contributing to the prediction
* Designed to make the ML model accessible and interpretable for non-technical users

## Tools and Libraries

* Python (Pandas, NumPy)
* Scikit-learn
* XGBoost
* LightGBM
* SHAP (model explainability)
* imbalanced-learn (for SMOTE-Tomek resampling)
* Streamlit (for building the prototype interface)
