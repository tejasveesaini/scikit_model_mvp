
**Generate requirements.txt** \
`pip freeze > requirements.txt`

# Domain knowlege on Diabetes
 Research on this dataset highlights that Glucose and BMI are the strongest predictors for this classification.
​

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
