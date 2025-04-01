# Case Study for Andrew Leem – Loan Default Prediction
This project develops a production-ready machine learning pipeline for predicting loan default. It spans end-to-end processes including exploratory data analysis (EDA), preprocessing, feature engineering, model tuning, and prediction generation. The focus is on creating a scalable and interpretable pipeline for real-world application, especially where ground truth may be unavailable in test datasets.


## Features

- Robust data cleaning and preprocessing pipeline with custom transformers
- One-hot and ordinal encoding strategies for categorical features
- Feature selection based on:
  - Visual EDA insights
  - Random Forest importance metrics
  - Optuna-tuned model results
- Model training with hyperparameter optimization using Optuna
- Cross-validation with Precision-Recall metrics for imbalanced data
- Final predictions with feature ranking and interpretability

## Project Structure

```
andrew-assessment/
├── src/
│   └── andy/                    # Core package: transformers, feature, modeling logic
├── notebooks/                   # EDA and modeling notebooks
├── docs/                        # Documentation (optional)
├── data/                        # Cleaned datasets
├── .github/workflows/           # GitHub Actions setup
├── pyproject.toml               # Poetry project metadata
├── README.md                    # Project documentation
```

## Exploratory Data Analysis (EDA)

### 1D EDA Highlights

- **Younger borrowers** and **more recent loans** show **higher default rates**
- **Higher interest rates** are strongly correlated with default likelihood
- **Wedding-related loans** appear safer, while **small business loans** carry greater risk
- **Drivers** and **temporary workers** default more frequently compared to **engineers** and **teachers**

### 2D EDA Highlights

- Key risk indicators identified include:
  - `loan_to_income`
  - `dti`
  - `purpose`
  - `verification_status`

- **Loan purpose impacts default rate** even when income levels are similar
- Even **verified borrowers** show elevated default rates for certain loan purposes, such as:
  - **Medical**
  - **Small business**
  - **Renewable energy**
---

## Modeling & Feature Selection

1. **Random Forest Feature Importances**  
   - Selected **89 features** using `SelectFromModel` based on the median importance threshold.

2. **Feature Selection via EDA**  
   - Manual inspection from 1D and 2D plots revealed additional relevant features:
     - `loan_to_income`
     - `dti`
     - `purpose`
     - `verification_status`

3. **Final Feature Set**  
   - Combined features from RF and EDA after mapping to their one-hot encoded column names.
   - Initial modeling started with 89 features, and feature count was reduced to **35** based on performance trends from the elbow method.

4. **Hyperparameter Tuning with Optuna**  
   - Tuned a `RandomForestClassifier` using cross-validated ROC AUC.
   - Best parameters:
     - `n_estimators`: 243  
     - `max_depth`: 14  
     - `min_samples_split`: 8  
   - Evaluation Metric: `roc_auc`  
   - Cross-validation: 3 folds

5. **Elbow Method for Feature Count**  
   - Evaluated AUC using increasing top-K features ranked by importance.
   - Performance plateaued around **35 features**.
   - Chose 35 as the optimal number for interpretability and performance balance.

6. **Final Model**  
   - Trained a `RandomForestClassifier` on the selected top 35 features using the optimized hyperparameters from Optuna.

---

## Modeling Workflow

- **Model**: `RandomForestClassifier` (Optuna-tuned)

- **Pipeline Components**:
  - Preprocessing using `ColumnTransformer`
  - Top-K feature selection based on feature importances
  - Elbow-method-driven feature count selection
  - Hyperparameter-optimized model fitting

- **Design Principles**:
  - Fully modular and reusable pipeline
  - Clean separation between data preprocessing, modeling, and evaluation
  - Scalable structure that can accommodate further model experimentation

## Predictions & Inference

Due to the lack of ground truth in the test dataset, we focused on analyzing predicted probabilities and class distributions rather than traditional evaluation metrics.

## Evaluation 

Because the test set lacks true labels (is_default), we did not compute metrics like F1 or Precision-Recall curves on it. Instead:
	•	All model selection was done via cross-validation on the training set using ROC AUC.
	•	Threshold tuning and distribution inspection were used to understand class balance implications.
	•	Feature importance was visualized to interpret model behavior.

## Probability Distribution

The histogram of predicted default probabilities shows a bell-shaped distribution, slightly left-skewed and centered around 0.45. This suggests the model is neither overconfident nor collapsed to the mean.

## Predicted Class Distribution

~18.3% of the test data was predicted as defaults. This aligns with the distribution of defaults observed in the training dataset, suggesting stable generalization.

## Top Risk Samples

- The top **10 loans** predicted with the **highest probability of default** were flagged using the trained model.
- All top samples had **predicted probabilities above 0.83**, making them strong candidates for **manual review**, **intervention**, or **targeted risk mitigation strategies**.
- These high-risk predictions can inform decisions on credit approval, interest rate adjustments, or additional document verification.

---

## Future Work

- **Implement LightGBM** to explore potential performance improvements over Random Forest  
  *(Note: LightGBM was not included in the current build due to dependency issues on M1 chip)*

- **Model Interpretability**:
  - Integrate SHAP or TreeInterpreter to explain predictions and audit for fairness or bias.

- **Dashboarding & Usability**:
  - Develop a real-time **Streamlit dashboard** to allow loan officers to score and interpret individual applications.

- **Enhanced Feature Engineering**:
  - Explore interaction terms, domain-specific transformations, or polynomial feature expansion.

- **Production-Ready Scoring**:
  - Package the final model into a **deployable scoring pipeline** for use in APIs or batch scoring systems.
