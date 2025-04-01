# Case Study for Andrew Leem – Loan Default Prediction

This project develops a production-ready machine learning pipeline for predicting loan default. It spans end-to-end processes including exploratory data analysis (EDA), preprocessing, feature engineering, model tuning, and prediction generation. The focus is on creating a scalable and interpretable pipeline for real-world application, especially where ground truth may be unavailable in test datasets.
---

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
---

## Project Structure

andrew-assessment/
├── src/
│   └── andy/                     # Core package: transformers, feature, modeling logic
├── notebooks/                   # EDA and modeling notebooks
├── docs/                        # Documentation (optional)
├── data/                        # Cleaned datasets
├── .github/workflows/          # GitHub Actions setup
├── pyproject.toml              # Poetry project metadata
├── README.md                   # Project documentation
---

## Exploratory Data Analysis (EDA)

1D EDA Highlights
	•	Younger borrowers and newer loans show higher default rates
	•	Higher interest rates strongly correlate with default
	•	Loans for weddings tend to be safer, while loans for small businesses have greater risk
	•	Drivers and temporary workers tend to default more often than engineers or teachers

2D EDA Highlights
	•	Key risk signals: loan_to_income, dti, purpose, and verification_status
	•	Default rates differ widely across purpose even when income levels are similar
	•	Verified borrowers still default for high-risk loan purposes (e.g. medical, small biz)
---

## Modeling & Feature Selection
	1.	Random Forest Feature Importances
      Selected 89 features using SelectFromModel based on median importance threshold.
	2.	Feature Selection with EDA
      Manual inspection from 1D and 2D plots identified additional useful features:
	•	loan_to_income
	•	dti
	•	purpose, verification_status
	3.	Final Feature Set
   Combined the above with:
	•	Proper mapping of original features to one-hot encoded columns.
	•	Final modeling used ~89 features → further reduced to 35 based on elbow curve.
	4.	Optuna Hyperparameter Tuning
   Tuned RandomForest with:
	•	n_estimators: 243
	•	max_depth: 14
	•	min_samples_split: 8
	•	Scored using ROC AUC (CV=3)
	5.	Elbow Method for Feature Count
   Evaluated AUC using top-K features:
	•	Performance plateaued after ~35 features.
	•	Chose 35 as the final number for a balance of performance and interpretability.
	6.	Final Model
   A RandomForestClassifier was trained with tuned parameters on the selected 35 features.
---

## Modeling Workflow

	•	Model: RandomForestClassifier (Optuna-tuned)
	•	Final modeling pipeline combines:
      •	Preprocessing with ColumnTransformer
      •	Top k features from feature importance ranking
      •	Hyperparameter-optimized Random Forest
	•	All transformations are modular and reusable
---

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

Top 10 loans predicted with the highest default probabilities were flagged. All showed risk scores above 0.83, useful for identifying potentially high-risk applicants for manual review or intervention.
---

## Future Work

	•	Implement LightGBM to explore performance gains over Random Forest
         (Note: not currently included due to dependency issues)
	•	Add SHAP or tree interpreter for model explanation and fairness audit
	•	Build real-time Streamlit dashboard for use by loan officers
	•	Improve feature engineering using interactions or polynomial expansion
	•	Package into deployable scoring pipeline for API or batch scoring
---