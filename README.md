

# Diabetes Prediction with Scikit-learn

This project demonstrates data preprocessing and machine learning model building for diabetes prediction using the Pima Indians Diabetes dataset. It includes:
- Data cleaning and handling of biologically impossible values
- Exploratory data analysis and feature importance
- Training and evaluation of Logistic Regression and Random Forest models

## Usage
1. Place the dataset file (`kaggle_pimadiabetes.csv`) in the project directory.
2. Install dependencies:
	```bash
	pip install -r requirements.txt
	```
3. Run the main script:
	```bash
	python model_trained.py
	```
4. Review the output in the terminal for model performance and feature importance. \
OR \
Directly check the existing result output stored in file `model_output.txt`.


## Model Summary
- **Logistic Regression**: Used after dropping columns with many missing values and normalizing features. Provides interpretable coefficients for feature importance.
- **Random Forest**: Trained on the original data (with missing values). Handles missing data natively and provides feature importance based on tree splits.

## Based on the model training and output results, here are 10 key insights:

### Model Performance Comparison

1. **Random Forest is slightly better overall.** Random Forest achieved 78.5% accuracy compared to Logistic Regression's 77.2%, making it the stronger performer for this dataset.
2. **Both models have excellent discrimination ability.** The ROC-AUC scores (0.83 for Logistic Regression, 0.82 for Random Forest) indicate both models are very good at distinguishing between diabetic and non-diabetic patients, regardless of the classification threshold chosen.
3. **Conservative on positive predictions.** Both models have higher precision (~70%) than recall (~59-63%), meaning when they predict diabetes, they're usually correct, but they miss about 37-40% of actual diabetes cases.

### Feature Importance Insights

4. **Glucose is the dominant predictor.** Both models rank Glucose as the most important feature, with Logistic Regression giving it a coefficient of 1.12 and Random Forest assigning it 26% importance—confirming our earlier discussion about glucose being the primary diabetes indicator.
5. **Insulin matters more in ensemble models.** Random Forest ranks Insulin as the second most important feature (15.7%), while Logistic Regression places it last (0.13). This suggests complex interactions with Insulin that tree-based models capture better.
6. **BMI and Age are consistently important.** Both models rank BMI and Age in the top 4-5 features, aligning with medical knowledge that obesity and aging are major diabetes risk factors.

### Classification Behavior

7. **The model is cautious about false alarms.** With only 7 false positives (incorrectly predicting diabetes), the Random Forest avoids unnecessarily alarming healthy patients.
8. **Missing real cases is the bigger problem.** The 10 false negatives mean the model fails to identify 37% of diabetic patients (10 out of 27 total). This is reflected in the recall score of 63%.
9. **Class imbalance is evident.** The confusion matrix shows 52 non-diabetic vs. 27 diabetic cases in your test set, suggesting the model sees far more negative examples, which may bias it toward predicting "no diabetes".

### Research Findings from dataset

Research on this dataset identifies **Glucose** and **BMI** as the primary drivers for classification.

## Key Risk Factors Summary

| Factor | Priority | Key Observation / Threshold |
| :--- | :--- | :--- |
| **Glucose** | Critical | **> 140-150 mg/dL** is the strongest indicator of a positive outcome. |
| **BMI** | High | **> 30** (Obesity) strongly correlates with Type 2 diabetes. |
| **Age** | Moderate | Older patients are more likely to receive a positive diagnosis. |
| **Pregnancies**| Moderate | Higher counts correlate with long-term metabolic stress. |

---

## Detailed Factor Analysis

### 1. High Glucose Levels (Most Critical)
*Plasma glucose concentration* is the single most important feature in this dataset.
- **Threshold:** 2-hour oral glucose tolerance test values above **140-150 mg/dL**.
- **Trend:** Rows with `Glucose > 140` are far more frequently labeled as **1** (positive) compared to those with `Glucose < 100`.

### 2. High Body Mass Index (BMI)
There is a distinct correlation between elevated BMI and diagnosis.
- **Risk Profile:** A BMI above 30 is a major risk factor.
- **Combined Risk:** An individual presenting with **both** high BMI and high glucose falls into the highest risk category.

### 3. Demographic Factors
- **Age:** The likelihood of diabetes increases progressively with age in this dataset.
- **Pregnancies:** A history of multiple pregnancies is linked to a higher likelihood of diabetes, likely due to cumulative metabolic stress.
