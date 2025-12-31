
# Import necessary libraries for data manipulation, machine learning, and evaluation
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split  # For splitting data into train/test sets
from sklearn.preprocessing import StandardScaler      # For feature normalization
from sklearn.linear_model import LogisticRegression   # For logistic regression model
from sklearn.ensemble import RandomForestClassifier   # For random forest model
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix  # For model evaluation

# Load the dataset from CSV into a pandas DataFrame
# This dataset contains medical data for diabetes prediction
df = pd.read_csv('kaggle_pimadiabetes.csv')


# While there are no empty cells or duplicate rows, some columns contain zeros where a value of 0 is not biologically possible for a living person.
# These zeros represent missing or unmeasured values and need to be handled.
# List of columns where 0 is invalid (cannot be 0 for a living person)
cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']


# Replace 0 with NaN in these specific columns so that pandas treats them as missing values
# This allows us to use pandas' built-in functions for handling missing data
df[cols] = df[cols].replace(0, np.nan)


# Display basic information about the dataset
print("Dataset shape:", df.shape)  # Shows number of rows and columns
print("\nFirst few rows:")
print(df.head())  # Preview of the data
print("\nDataset info:")
print(df.info())  # Data types and non-null counts
print("\nMissing values:")
print(df.isnull().sum())  # Count of missing values per column
print("\nBasic statistics:")
print(df.describe())  # Summary statistics for each column

# Check class distribution to see if the dataset is balanced or imbalanced
# This is important for model evaluation and interpretation
print("\nDiabetes distribution:")
print(df['Outcome'].value_counts())


# Prepare data for Logistic Regression (which does not accept missing values)
# For this, we drop columns with many missing values and drop rows with remaining missing values
# 1. Drop the 'Insulin' column (many missing values, less predictive power)
df_cleaned = df.drop('Insulin', axis=1)

# 2. Drop 'SkinThickness' (also has ~30% missing, less critical for prediction)
df_cleaned = df_cleaned.drop('SkinThickness', axis=1)

# 3. For the remaining columns (e.g., Glucose, BMI), drop rows with missing values
# This preserves data quality for models that can't handle NaNs
df_cleaned = df_cleaned.dropna()


# Separate features (X) and target variable (y)
X = df_cleaned.drop('Outcome', axis=1)
y = df_cleaned['Outcome']


# Split data into training and testing sets (80-20 split)
# This allows us to evaluate model performance on unseen data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining set size: {X_train.shape[0]}")  # Number of samples for training
print(f"Testing set size: {X_test.shape[0]}")   # Number of samples for testing


# Normalize features (important for logistic regression)
# Standardization ensures all features contribute equally to the model
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Train Logistic Regression model
print("\n" + "="*50)
print("LOGISTIC REGRESSION MODEL")
print("="*50)

# Initialize and train the logistic regression model
# Logistic regression is a linear model for binary classification
lr_model = LogisticRegression(random_state=42, max_iter=1000)
print("Columns going into model:", X_train.columns.tolist())
lr_model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred_lr = lr_model.predict(X_test_scaled)
y_pred_proba_lr = lr_model.predict_proba(X_test_scaled)[:, 1]  # Probability estimates for ROC-AUC

# Evaluate model performance using common metrics
# Accuracy: Overall correctness
# Precision: Correct positive predictions / all positive predictions
# Recall: Correct positive predictions / all actual positives
# ROC-AUC: Discrimination ability between classes
print(f"Accuracy: {accuracy_score(y_test, y_pred_lr):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_lr):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_lr):.4f}")
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba_lr):.4f}")

# Feature importance for logistic regression (model coefficients)
# Positive coefficients increase the likelihood of diabetes, negative decrease it
print("\nFeature Importance (Coefficients):")
feature_importance_lr = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': lr_model.coef_[0]
}).sort_values('Coefficient', ascending=False)
print(feature_importance_lr)





# Train Random Forest model (can handle missing values natively)
print("\n" + "="*50)
print("RANDOM FOREST MODEL")
print("="*50)

# Use the original DataFrame (with NaNs) for Random Forest
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Split data into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")

# Initialize and train the Random Forest model
# Random Forest is an ensemble method that can handle missing values and does not require feature scaling
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
rf_model.fit(X_train, y_train)  # Random Forest doesn't need scaling

# Make predictions on the test set
y_pred_rf = rf_model.predict(X_test)
y_pred_proba_rf = rf_model.predict_proba(X_test)[:, 1]

# Evaluate model performance using the same metrics as before
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_rf):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_rf):.4f}")
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba_rf):.4f}")

# Feature importance for random forest
# Shows which features are most influential in the model's decisions
print("\nFeature Importance:")
feature_importance_rf = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)
print(feature_importance_rf)

# Confusion matrix for Random Forest predictions
# Shows the breakdown of true/false positives/negatives
print("\n" + "="*50)
print("CONFUSION MATRIX (Random Forest)")
print("="*50)
cm = confusion_matrix(y_test, y_pred_rf)
print(cm)
print(f"\nTrue Negatives: {cm[0,0]}, False Positives: {cm[0,1]}")
print(f"False Negatives: {cm[1,0]}, True Positives: {cm[1,1]}")