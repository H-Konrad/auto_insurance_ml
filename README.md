# Insurance Fraud Detection & Claims Prediction

## Overview

This project investigates insurance claim fraud detection and claim cost prediction using machine learning. It is split up into two sections: 

1. Insurance Fraud Classification - predicting whether a claim is fraudulent.
2. Insurance Claim Cost Prediction - estimating the total claim amount.

The project includes exploratory data analysis, feature engineering, model development, and model evaluation for both tasks. 

### Repository Structure

The project contains notebooks, utility functions, and generated figures. The following files are not included:
- Original and processed datasets
- Trained models
- Model outputs and metrics for evaluation

```text
data/               # Dataset files (not included)
figures/            # Generated plots
notebooks/          # EDA, feature engineering, modelling and analysis notebooks
notes/              # Model metrics during training
utils/              # Helper, modelling, and plotting functions
.gitignore
README.md
environment.yml     # Environment dependencies
```

### Dataset

The dataset consists of 1,000 insurance claims records with 40 features spanning numeric, categorical, and datatime data types. The data is not included in this repository, and can be downloaded from [Kaggle](https://www.kaggle.com/datasets/buntyshah/auto-insurance-claims-data).

The features describe the insurance policy, incidents, vehicles, and claim information. The classification target, `fraud_reported`, is a binary variable indicating whether a claim was fraudulent. The regression target, `total_claim_amount`, represents the total value of a claim. 