# Bank Marketing ML Project

## Project Overview
This project delivers an end-to-end machine learning system to predict whether a customer will subscribe to a term deposit. It is university submission ready, GitHub portfolio ready, and discussion/viva ready.

## Business Problem
A bank conducts direct marketing campaigns to encourage customers to subscribe to term deposits. The goal is to predict subscription outcomes (yes/no) using demographic, financial, and campaign history features.

## Dataset Description
- Source file: `bank/bank-full.csv`
- Rows: 45,211
- Columns: 17
- Target: `y` (yes/no)

## Data Quality Issues
Missing values are encoded as "unknown" and are treated as missing during preprocessing. Known occurrences:
- job: 288
- education: 1857
- contact: 13020
- poutcome: 36959

## Cleaning Decisions
- Replaced "unknown" with missing values
- Removed duplicates
- Capped numeric outliers using IQR

## EDA Summary
The EDA notebook includes:
- Dataset shape, types, missing values, duplicates
- Detailed statistical summaries (numeric and categorical)
- Univariate, bivariate, and multivariate analysis
- Business insights and recommendations after each chart

## Feature Engineering
Five features were created:
- Customer_Engagement_Index = campaign + previous
- Financial_Stability_Index = balance / age
- Marketing_Effectiveness = duration / campaign
- Previous_Campaign_Success derived from poutcome
- Loan_Risk_Index from housing + loan

## Feature Selection
Three techniques are compared:
- Filter: Mutual Information
- Wrapper: RFE
- Embedded: XGBoost Feature Importance
Final features are selected based on consensus across methods.

## Models
Six models are trained:
- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost
- KNN
- SVM

**Note on SVM:** `SVC(probability=True)` has roughly O(n^2)-O(n^3) training complexity,
which is impractical on the ~36K-row training set. SVM is therefore trained on a
stratified 6,000-row subsample (a standard practice for kernel SVMs on large
datasets), while it is always evaluated on the full held-out test set.

## Validation
- Stratified train/test split (80/20)
- Stratified 5-fold cross-validation
- Train/test/CV comparison for overfitting risk

## Hyperparameter Tuning
GridSearchCV is applied to:
- Random Forest
- XGBoost
- SVM

## Threshold Optimization
Thresholds from 0.10 to 0.90 are tested to achieve:
- Precision >= 0.30
- Recall >= 0.30
The chosen threshold maximizes F1 while meeting business constraints.

## Duration Leakage Note
`duration` is known only after a call ends. Two scenarios are trained:
- With duration: post-call analysis
- Without duration: pre-campaign prediction (deployment)
The deployment model excludes duration.

## Final Results

Model comparison (without duration, pre-campaign scenario, test set):

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| XGBoost | 0.904 | 0.625 | 0.447 | 0.521 | 0.920 |
| Random Forest | 0.899 | 0.643 | 0.305 | 0.414 | 0.911 |
| Logistic Regression | 0.896 | 0.609 | 0.318 | 0.417 | 0.869 |
| SVM | 0.894 | 0.679 | 0.176 | 0.279 | 0.869 |
| KNN | 0.896 | 0.621 | 0.285 | 0.391 | 0.780 |
| Decision Tree | 0.869 | 0.440 | 0.452 | 0.446 | 0.688 |

**Best model: XGBoost (no-duration scenario)**, selected by ROC-AUC and balanced
precision/recall.

After GridSearchCV tuning (best params: `n_estimators=300`, `max_depth=4`,
`learning_rate=0.1`), tuned XGBoost reaches **ROC-AUC 0.924**.

At the selected decision threshold of **0.25**, the deployed XGBoost model achieves:
- Accuracy: 0.890
- Precision: 0.522 (>= 0.30 requirement)
- Recall: 0.742 (>= 0.30 requirement)
- F1: 0.613
- ROC-AUC: 0.924

The deployed model is the tuned, no-duration XGBoost model.

## Deployment
Project deployment platform: Streamlit Community Cloud.

App links are listed in `deployment_links.txt`.

## How to Run Locally
```bash
pip install -r requirements.txt
```

Run notebooks:
- `notebooks/EDA_Analysis.ipynb`
- `notebooks/Machine_Learning.ipynb`

Run Streamlit apps:
```bash
streamlit run streamlit_apps/streamlit_eda_app.py
streamlit run streamlit_apps/streamlit_ml_app.py
```

## Repository Structure
- notebooks/EDA_Analysis.ipynb
- notebooks/Machine_Learning.ipynb
- streamlit_apps/streamlit_eda_app.py
- streamlit_apps/streamlit_ml_app.py
- trained_models/
- saved_transformers/
- bank/bank-full.csv (raw dataset)
- cleaned_bank_dataset.csv (cleaned dataset)
- requirements.txt

## Future Improvements
- Cost-sensitive learning
- Calibration analysis
- Model monitoring and drift detection

## Acknowledgment
This project was completed as part of the Epsilon AI Machine Learning Program.

Special thanks to Epsilon AI for providing the learning materials, guidance, and project framework.

Official Epsilon AI GitHub: https://github.com/epsilon-ai

## Author
Elhossain Emad Abdelsalam
