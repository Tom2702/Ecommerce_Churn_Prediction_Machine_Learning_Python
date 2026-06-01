# E-Commerce Churn Prediction | Machine Learning

## Contents

1. [Data Loading and Initial Inspection](#1-data-loading-and-initial-inspection)
2. [Feature Classification](#2-feature-classification)
3. [Data Distribution Analysis](#3-data-distribution-analysis)
4. [Missing Values and Data Cleaning](#4-missing-values-and-data-cleaning)
5. [Relationship Between Features and Churn](#5-relationship-between-features-and-churn)
6. [Churned User Behavior Analysis](#6-churned-user-behavior-analysis)
7. [Business Recommendations](#7-business-recommendations)
8. [Supervised Learning](#8-supervised-learning)
9. [Supervised Learning Results](#9-supervised-learning-results)
10. [Unsupervised Learning](#10-unsupervised-learning)
11. [Unsupervised Learning Results](#11-unsupervised-learning-results)
12. [Final Business Interpretation](#12-final-business-interpretation)
13. [How to Run](#13-how-to-run)

---

## 1. Data Loading and Initial Inspection

The notebook starts by importing the main Python libraries and loading the churn dataset.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

The dataset loader supports both Colab and local execution.

```python
from pathlib import Path

candidate_files = [
    Path("/content/drive/MyDrive/Churn_prediction.xlsx"),
    Path("/content/drive/MyDrive/churn_predict.csv"),
    Path("Churn_prediction.xlsx"),
    Path("churn_predict.csv"),
]

data_path = next((file for file in candidate_files if file.exists()), None)

if data_path is None:
    raise FileNotFoundError(
        "Dataset not found. Please place churn_predict.csv or Churn_prediction.xlsx "
        "in the notebook directory, or update candidate_files with the correct path."
    )

if data_path.suffix.lower() in [".xlsx", ".xls"]:
    df = pd.read_excel(data_path)
elif data_path.suffix.lower() == ".csv":
    df = pd.read_csv(data_path)
else:
    raise ValueError(f"Unsupported file format: {data_path.suffix}")

print(f"Loaded dataset: {data_path}")
df.head()
```

Initial inspection is performed using:

```python
df.head(5)
df.shape
df.info()
```

### Result Placeholder

```text
Dataset shape:
[Insert dataset shape here]

Data types and missing overview:
[Insert df.info() summary here]
```

---

## 2. Feature Classification

The notebook separates variables into ID columns, manual categorical columns, numerical columns, and categorical columns.

```python
id_cols = ["CustomerID"]

cat_manual = ["Churn", "Complain", "CityTier", "SatisfactionScore"]

num_cols = df.select_dtypes(include=np.number).columns.tolist()
num_cols = [col for col in num_cols if col not in id_cols and col not in cat_manual]

cat_cols = df.select_dtypes(include="object").columns.tolist()
cat_cols = cat_cols + [col for col in cat_manual if col in df.columns]
```

This step is important because numerical and categorical variables are handled differently during EDA, preprocessing, and modeling.

### Result Placeholder

```text
ID columns:
[Insert ID columns]

Numerical columns:
[Insert numerical columns]

Categorical columns:
[Insert categorical columns]
```

---

## 3. Data Distribution Analysis

The notebook analyzes the distribution of numerical and categorical features.

### 3.1 Numerical Variables

```python
df[num_cols].describe().T

df[num_cols].skew()

for col in num_cols:
    sns.histplot(df[col], kde=True)
    plt.title(f"Distribution of {col}")
    plt.show()
```

### 3.2 Categorical Variables

```python
for col in cat_cols:
    print(col)
    print(df[col].value_counts())
    print(df[col].value_counts(normalize=True) * 100)
```

### Result Placeholder

```markdown
![Numerical Distribution](images/numerical_distribution.png)
![Categorical Distribution](images/categorical_distribution.png)
```

```text
Key distribution insight 1:
[Insert insight]

Key distribution insight 2:
[Insert insight]
```

---

## 4. Missing Values and Data Cleaning

The notebook checks missing values and handles them based on feature type.

```python
df.isnull().mean() * 100
```

Numerical missing values are filled with median values.

```python
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())
```

Categorical missing values are filled with mode values.

```python
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])
```

Inconsistent category labels are standardized.

```python
if "PreferredLoginDevice" in df.columns:
    df["PreferredLoginDevice"] = df["PreferredLoginDevice"].replace({
        "Phone": "Mobile Phone",
        "Mobile": "Mobile Phone",
    })

if "PreferredPaymentMode" in df.columns:
    df["PreferredPaymentMode"] = df["PreferredPaymentMode"].replace({
        "COD": "Cash on Delivery",
        "CC": "Credit Card",
    })

order_cat_col = None
for possible_col in ["PreferedOrderCat", "PreferredOrderCat"]:
    if possible_col in df.columns:
        order_cat_col = possible_col
        break

if order_cat_col is not None:
    df[order_cat_col] = df[order_cat_col].replace({
        "Mobile": "Mobile Phone",
    })
```

Outliers are inspected using boxplots.

```python
for col in num_cols:
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")
    plt.show()
```

### Result Placeholder

```text
Missing value treatment summary:
[Insert summary]

Standardized category values:
[Insert summary]
```

---

## 5. Relationship Between Features and Churn

The notebook compares churned and non-churned customers across numerical and categorical variables.

### 5.1 Numerical Features vs Churn

```python
df.groupby("Churn")[num_cols].mean()
df.groupby("Churn")[num_cols].median()

for col in num_cols:
    sns.boxplot(data=df, x="Churn", y=col)
    plt.title(f"{col} by Churn")
    plt.show()
```

### 5.2 Correlation Heatmap

```python
plt.figure(figsize=(12, 8))
sns.heatmap(df[num_cols + ["Churn"]].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()
```

### 5.3 Churn Rate by Category

```python
for col in cat_cols:
    if col != "Churn":
        churn_rate = df.groupby(col)["Churn"].mean().sort_values(ascending=False)
        print(churn_rate)
```

### Result Placeholder

```markdown
![Correlation Heatmap](images/correlation_heatmap.png)
![Churn Rate by Category](images/churn_rate_by_category.png)
```

```text
Main churn driver 1:
[Insert insight]

Main churn driver 2:
[Insert insight]
```

---

## 6. Churned User Behavior Analysis

The notebook summarizes churned customer behavior after comparing churned and retained users.

Expected churned user patterns include:

- Shorter tenure
- More complaints
- Lower cashback amount
- Lower engagement
- Different order behavior
- Possible delivery inconvenience

### Result Placeholder

```text
Churned customer behavior summary:
[Insert final behavioral findings from notebook]
```

---

## 7. Business Recommendations

The notebook provides business recommendations before moving into machine learning.

Recommended retention actions include:

1. Welcome vouchers and onboarding support for new customers.
2. Faster complaint handling and compensation vouchers.
3. Personalized promotions for high-risk customers.
4. Delivery support for customers far from warehouses.
5. Win-back campaigns for valuable churned customers.

### Result Placeholder

```text
Business recommendation summary:
[Insert final recommendation text]
```

---

## 8. Supervised Learning

The notebook builds supervised learning models to predict churn.

### 8.1 Import Modeling Libraries

```python
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    balanced_accuracy_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)

import joblib
from pathlib import Path
```

### 8.2 One-Hot Encoding

```python
model_cat_cols = [col for col in cat_cols if col != "Churn"]
df_encoded = pd.get_dummies(df, columns=model_cat_cols, drop_first=True, dtype=int)
```

### 8.3 Define Features and Target

```python
x = df_encoded.drop(columns=["CustomerID", "Churn"])
y = df_encoded["Churn"]
```

### 8.4 Train-Test Split

```python
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)
```

### 8.5 Logistic Regression

```python
scaler = StandardScaler()

x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

log_model = LogisticRegression(max_iter=1000, class_weight="balanced")
log_model.fit(x_train_scaled, y_train)

y_test_pred = log_model.predict(x_test_scaled)

print("Logistic Regression")
print("Balanced Accuracy:", balanced_accuracy_score(y_test, y_test_pred))
print(confusion_matrix(y_test, y_test_pred))
print(classification_report(y_test, y_test_pred))
```

### 8.6 Random Forest

```python
rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",
)

rf_model.fit(x_train, y_train)

y_test_pred = rf_model.predict(x_test)

print("Random Forest")
print("Balanced Accuracy:", balanced_accuracy_score(y_test, y_test_pred))
print(confusion_matrix(y_test, y_test_pred))
print(classification_report(y_test, y_test_pred))
```

### 8.7 Hyperparameter Tuning

```python
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
    "bootstrap": [True, False],
}

grid_search = GridSearchCV(
    estimator=rf_model,
    param_grid=param_grid,
    cv=5,
    scoring="balanced_accuracy",
)

grid_search.fit(x_train, y_train)

print("Best Parameters:")
print(grid_search.best_params_)
```

### 8.8 Best Random Forest Evaluation

```python
best_rf_model = grid_search.best_estimator_

y_pred_best_rf = best_rf_model.predict(x_test)

print("Tuned Random Forest")
print("Balanced Accuracy:", balanced_accuracy_score(y_test, y_pred_best_rf))
print(confusion_matrix(y_test, y_pred_best_rf))
print(classification_report(y_test, y_pred_best_rf))
```

---

## 9. Supervised Learning Results

This section is reserved for final model outputs after running the notebook.

### 9.1 Model Comparison

| Model | Balanced Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Logistic Regression | `[Insert]` | `[Insert]` | `[Insert]` | `[Insert]` |
| Random Forest | `[Insert]` | `[Insert]` | `[Insert]` | `[Insert]` |
| Tuned Random Forest | `[Insert]` | `[Insert]` | `[Insert]` | `[Insert]` |

### 9.2 Best Model Parameters

```text
Best Parameters:
[Insert best parameters here]
```

### 9.3 Confusion Matrix and Classification Report

```text
Confusion Matrix:
[Insert confusion matrix here]

Classification Report:
[Insert classification report here]
```

### 9.4 Result Images

```markdown
![Model Comparison](images/model_comparison.png)
![Confusion Matrix](images/confusion_matrix.png)
![ROC and PR Curves](images/roc_pr_curves.png)
![Threshold Tuning](images/threshold_tuning.png)
![Feature Importance](images/feature_importance.png)
```

---

## 10. Unsupervised Learning

The notebook applies K-Means clustering only to churned customers.

### 10.1 Filter Churned Customers

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

churn_df = df[df["Churn"] == 1].copy()
churn_df.shape
```

### 10.2 Prepare Clustering Data

```python
cluster_data = churn_df.drop(columns=["CustomerID", "Churn"])

cluster_data_encoded = pd.get_dummies(
    cluster_data,
    drop_first=True,
    dtype=int,
)

scaler = StandardScaler()
cluster_data_scaled = scaler.fit_transform(cluster_data_encoded)
```

### 10.3 Elbow Method

```python
inertia = []
K_range = range(2, 20)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(cluster_data_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(K_range, inertia, marker="o")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.show()
```

### 10.4 Train K-Means

```python
optimal_k = 7

kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
churn_df["Segment"] = kmeans.fit_predict(cluster_data_scaled)
```

### 10.5 Cluster Validation

```python
silhouette_avg = silhouette_score(cluster_data_scaled, churn_df["Segment"])
print(f"Silhouette Score for k={optimal_k}: {silhouette_avg:.4f}")
```

PCA visualization:

```python
pca = PCA(n_components=2, random_state=42)
cluster_pca = pca.fit_transform(cluster_data_scaled)

pca_df = pd.DataFrame({
    "PCA1": cluster_pca[:, 0],
    "PCA2": cluster_pca[:, 1],
    "Segment": churn_df["Segment"].astype(str),
})

sns.scatterplot(data=pca_df, x="PCA1", y="PCA2", hue="Segment")
plt.title("PCA Visualization of Churned Customer Segments")
plt.show()
```

### 10.6 Segment Profiling

```python
churn_df["Segment"].value_counts(normalize=True) * 100

num_cols_cluster = churn_df.select_dtypes(include=np.number).columns.tolist()
num_cols_cluster = [
    col for col in num_cols_cluster
    if col not in ["CustomerID", "Churn", "Segment"]
]

churn_df.groupby("Segment")[num_cols_cluster].mean().T
```

Categorical profile:

```python
cat_cols_cluster = churn_df.select_dtypes(include="object").columns.tolist()

for col in cat_cols_cluster:
    print(f"\n{col}")
    print(pd.crosstab(churn_df["Segment"], churn_df[col], normalize="index") * 100)
```

---

## 11. Unsupervised Learning Results

This section is reserved for final clustering outputs after running the notebook.

### 11.1 Segment Distribution

```text
Segment distribution:
[Insert segment distribution here]
```

### 11.2 Segment Profile Table

| Segment | Main Characteristics | Suggested Action |
|---|---|---|
| Segment 0 | `[Insert]` | `[Insert]` |
| Segment 1 | `[Insert]` | `[Insert]` |
| Segment 2 | `[Insert]` | `[Insert]` |
| Segment 3 | `[Insert]` | `[Insert]` |
| Segment 4 | `[Insert]` | `[Insert]` |
| Segment 5 | `[Insert]` | `[Insert]` |
| Segment 6 | `[Insert]` | `[Insert]` |

### 11.3 Result Images

```markdown
![Elbow Method](images/elbow_method.png)
![Churned Customer Segments PCA](images/churned_customer_segments_pca.png)
```

---

## 12. Final Business Interpretation

The final section should summarize what the model and segmentation results mean for the business.

```text
[Insert final business interpretation here]
```

Suggested points:

- Which model performs best?
- Which metric matters most for churn prediction?
- Which customer groups are most likely to churn?
- Which churn segments should receive different retention actions?
- How can the company use churn probability to prioritize campaigns?

---

## 13. How to Run

Clone the repository:

```bash
git clone https://github.com/Tom2702/ML_Ecommerce_Churn_Prediction_Python.git
cd ML_Ecommerce_Churn_Prediction_Python
```

Install required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib openpyxl
```

Run the notebook:

```text
ML_Project.ipynb
```

Recommended workflow:

1. Run all notebook cells from top to bottom.
2. Export charts into the `images/` folder.
3. Fill in the placeholder sections in this README.
4. Upload the notebook, README, dataset, and images to GitHub.
