# Credit Risk ML

End-to-end Machine Learning pipeline for credit default prediction.

Dataset:
Home Credit Default Risk
https://www.kaggle.com/competitions/home-credit-default-risk/

---

## Project Goal

The goal of this project is to build a structured and extensible ML pipeline
for predicting loan default probability using financial and demographic data.

The focus is not only on model accuracy, but also on:

- Clean project architecture
- Feature engineering
- Reproducible training
- Business-oriented evaluation (ROC-AUC)

---

## Current Implementation

✔ Modular project structure  
✔ Data loading module  
✔ Feature engineering pipeline  
✔ Missing value indicators  
✔ Financial ratio features  
✔ Logistic Regression baseline  
✔ 5-Fold Cross-Validation  
✔ ROC-AUC evaluation  

Baseline Performance:
5-Fold CV ROC-AUC ≈ 0.748

---

## Project Structure
```
src/
│
├── data_loader.py
├── preprocessing.py
├── models/
│    └── train_baseline.py
│
├──notebooks/
│    (EDA and experiments – in progress)
└──requirements.txt
```
---

## Feature Engineering

Current feature transformations include:

- AGE calculation from DAYS_BIRTH
- CREDIT_INCOME_RATIO
- ANNUITY_CREDIT_RATIO
- EMPLOYED_AGE_RATIO
- Missing value flags for high-missing columns

Additional features will be added during model iteration.

---

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-Learn
- Logistic Regression
- K-Fold Cross Validation
- ROC-AUC metric

Planned:
- XGBoost
- Hyperparameter tuning
- Feature importance analysis
- Model comparison
- Visual evaluation (ROC curves, feature importance plots)
- Model serialization
- Simple FastAPI inference endpoint
- Docker containerization

---

## How to Run

Train baseline model:

    python -m src.models.train_baseline

Make sure dataset files are placed in the appropriate data directory
(see data_loader.py configuration).

---

## Roadmap

Short-term:
- Add XGBoost baseline
- Add hyperparameter tuning
- Improve missing value handling
- Add visual evaluation

Mid-term:
- Model comparison pipeline
- Feature importance analysis
- Model persistence
- Simple API for inference

Long-term:
- Dockerized deployment
- Improved reproducibility
- Experiment tracking

---

## Motivation

This project is part of my transition into applied Machine Learning.

It reflects my approach to:
- Structuring ML projects beyond notebooks
- Building reusable pipelines
- Understanding real-world financial risk modeling

Status: In active development.
