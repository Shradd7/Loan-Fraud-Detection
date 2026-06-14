# FICOFORCE Model Card

## Intended Use
FICOFORCE estimates ML-based fraud/default risk and supports manual fraud review. It is a decision-support system, not an automatic loan rejection engine.

## Model
The 2026 pipeline trains a LightGBM binary classifier with deterministic feature engineering, median imputation, class weighting, and validation-set threshold tuning for F1. A slower deep experiment can add Yeo-Johnson transformation and SMOTE-Tomek resampling.

## Primary Metrics
- F1 score
- Precision
- Recall
- PR-AUC
- ROC-AUC
- Confusion matrix

## Human Review Policy
Applications with high default probability, low model confidence, or conflicting location evidence should be reviewed by a credit officer before adverse action.

## Known Limitations
- Training depends on the quality and representativeness of the hackathon dataset.
- Dummy Task 2 data is synthetic and should not be treated as production identity data.
- Fairness checks should be performed before any real lending use.
