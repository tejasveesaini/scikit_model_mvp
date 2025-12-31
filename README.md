

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
	python data_preprocess.py
	```
4. Review the output in the terminal for model performance and feature importance.

**Generate requirements.txt** \
`pip freeze > requirements.txt`


# Domain knowlege on Diabetes
Research on this dataset highlights that Glucose and BMI are the strongest predictors for this classification.


## Model Summary
- **Logistic Regression**: Used after dropping columns with many missing values and normalizing features. Provides interpretable coefficients for feature importance.
- **Random Forest**: Trained on the original data (with missing values). Handles missing data natively and provides feature importance based on tree splits.

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
