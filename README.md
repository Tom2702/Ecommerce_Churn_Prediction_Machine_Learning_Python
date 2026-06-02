# E-Commerce Churn Prediction & Segmentation For Retention Strategy | Machine Learning - Python

Author: Anh Tuan Nguyen  
Date: June 2026  
Tools Used: Python, pandas, numpy, matplotlib, seaborn, scikit-learn, Jupyter Notebook / Google Colab

![Project Banner](images/project_banner.png)

---

## Table of Contents

1. [Project Objective](#1-project-objective)
2. [Dataset Description](#2-dataset-description)
3. [Main Process: Source Code and Output](#3-main-process-source-code-and-output)
4. [Final Results](#4-final-results)
5. [Business Recommendations](#5-business-recommendations)
6. [Conclusion](#6-conclusion)
7. [How To Run](#7-how-to-run)
8. [Repository Structure](#8-repository-structure)

---

## 1. Project Objective

This project analyzes customer churn behavior for an e-commerce business and builds machine learning models to predict whether a customer is likely to churn. The project also segments churned customers into behavioral groups so that the business can design more targeted retention and win-back strategies.

The main objectives are:

- Understand customer behavior and churn patterns through exploratory data analysis.
- Clean missing values and standardize inconsistent categorical labels.
- Identify key factors related to churn, such as tenure, complaints, cashback, order category, payment behavior, and delivery distance.
- Build supervised machine learning models to predict customer churn.
- Tune the best model using GridSearchCV.
- Segment churned customers using K-Means clustering.
- Convert model and segmentation results into practical retention recommendations.

---

## 2. Dataset Description

### 2.1 Data Source

The dataset used in this project is:

```text
Churn_prediction.xlsx
```

The notebook also supports loading a CSV version if needed:

```text
churn_predict.csv
```

### 2.2 Dataset Overview

| Metric | Value |
|---|---:|
| Total rows | 5,630 |
| Total columns | 20 |
| Duplicate rows | 0 |
| Active customers (`Churn = 0`) | 4,682 |
| Churned customers (`Churn = 1`) | 948 |
| Churn rate | 16.84% |
| Main task | Binary classification and churned-customer segmentation |

### 2.3 Data Dictionary

| Column | Description |
|---|---|
| `CustomerID` | Unique customer identifier |
| `Churn` | Target variable, where 1 means churned and 0 means active |
| `Tenure` | Customer tenure with the company |
| `PreferredLoginDevice` | Preferred login device |
| `CityTier` | Customer city tier |
| `WarehouseToHome` | Distance from warehouse to customer home |
| `PreferredPaymentMode` | Preferred payment method |
| `Gender` | Customer gender |
| `HourSpendOnApp` | Hours spent on app or website |
| `NumberOfDeviceRegistered` | Number of registered devices |
| `PreferedOrderCat` | Preferred order category |
| `SatisfactionScore` | Customer satisfaction score |
| `MaritalStatus` | Customer marital status |
| `NumberOfAddress` | Number of registered addresses |
| `Complain` | Whether the customer raised a complaint |
| `OrderAmountHikeFromlastYear` | Increase in order amount compared with last year |
| `CouponUsed` | Number of coupons used |
| `OrderCount` | Number of orders placed |
| `DaySinceLastOrder` | Days since the last order |
| `CashbackAmount` | Cashback amount received by the customer |

---

## 3. Main Process: Source Code and Output

### 3.1 Data Loading and Initial Inspection

**Source code**

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

candidate_files = [
    Path('/content/drive/MyDrive/Churn_prediction.xlsx'),
    Path('/content/drive/MyDrive/churn_predict.csv'),
    Path('Churn_prediction.xlsx'),
    Path('churn_predict.csv'),
]

data_path = next((file for file in candidate_files if file.exists()), None)

if data_path is None:
    raise FileNotFoundError(
        'Dataset not found. Please place churn_predict.csv or Churn_prediction.xlsx '
        'in the notebook directory, or update candidate_files with the correct path.'
    )

if data_path.suffix.lower() in ['.xlsx', '.xls']:
    df = pd.read_excel(data_path)
elif data_path.suffix.lower() == '.csv':
    df = pd.read_csv(data_path)
else:
    raise ValueError(f'Unsupported file format: {data_path.suffix}')

print(f'Loaded dataset: {data_path}')
df.head()
```

**Output**

```text
Loaded dataset: Churn_prediction.xlsx
Dataset shape: (5630, 20)
```

```text
Data columns: 20 columns
Rows: 5,630
Dtypes: float64(8), int64(7), object(5)
Memory usage: 879.8+ KB
```

**Result image placeholder**

```markdown
![Dataset Preview](images/dataset_preview.png)
```

### 3.2 Feature Classification

**Source code**

```python
id_cols = ['CustomerID']
cat_manual = ['Churn', 'Complain', 'CityTier', 'SatisfactionScore']

num_cols = df.select_dtypes(include=np.number).columns.tolist()
num_cols = [col for col in num_cols if col not in id_cols and col not in cat_manual]

cat_cols = df.select_dtypes(include='object').columns.tolist()
cat_cols = cat_cols + [col for col in cat_manual if col in df.columns]

print('Numerical columns:', num_cols)
print('Categorical columns:', cat_cols)
```

**Output**

```text
Numerical columns:
Tenure, WarehouseToHome, HourSpendOnApp, NumberOfDeviceRegistered,
NumberOfAddress, OrderAmountHikeFromlastYear, CouponUsed, OrderCount,
DaySinceLastOrder, CashbackAmount

Categorical columns:
PreferredLoginDevice, PreferredPaymentMode, Gender, PreferedOrderCat,
MaritalStatus, Churn, Complain, CityTier, SatisfactionScore
```

### 3.3 Data Distribution Analysis

**Source code**

```python
df[num_cols].describe().T
```

**Output**

| Feature | Count | Mean | Std | Min | Median | Max |
|---|---:|---:|---:|---:|---:|---:|
| Tenure | 5,366 | 10.19 | 8.56 | 0.00 | 9.00 | 61.00 |
| WarehouseToHome | 5,379 | 15.64 | 8.53 | 5.00 | 14.00 | 127.00 |
| HourSpendOnApp | 5,375 | 2.93 | 0.72 | 0.00 | 3.00 | 5.00 |
| NumberOfDeviceRegistered | 5,630 | 3.69 | 1.02 | 1.00 | 4.00 | 6.00 |
| NumberOfAddress | 5,630 | 4.21 | 2.58 | 1.00 | 3.00 | 22.00 |
| OrderAmountHikeFromlastYear | 5,365 | 15.71 | 3.68 | 11.00 | 15.00 | 26.00 |
| CouponUsed | 5,374 | 1.75 | 1.89 | 0.00 | 1.00 | 16.00 |
| OrderCount | 5,372 | 3.01 | 2.94 | 1.00 | 2.00 | 16.00 |
| DaySinceLastOrder | 5,323 | 4.54 | 3.65 | 0.00 | 3.00 | 46.00 |
| CashbackAmount | 5,630 | 177.22 | 49.21 | 0.00 | 163.28 | 324.99 |

**Source code**

```python
df[num_cols].skew()
```

**Output**

| Feature | Skewness |
|---|---:|
| Tenure | 0.737 |
| WarehouseToHome | 1.619 |
| HourSpendOnApp | -0.027 |
| NumberOfDeviceRegistered | -0.397 |
| NumberOfAddress | 1.089 |
| OrderAmountHikeFromlastYear | 0.791 |
| CouponUsed | 2.546 |
| OrderCount | 2.196 |
| DaySinceLastOrder | 1.191 |
| CashbackAmount | 1.150 |

**Source code**

```python
for col in num_cols:
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.show()
```

**Result image placeholder**

```markdown
![Numerical Distribution](images/numerical_distribution.png)
```

**Source code**

```python
for col in cat_cols:
    print(col)
    print(df[col].value_counts())
    print(df[col].value_counts(normalize=True) * 100)
```

**Output summary**

| Variable | Main Output |
|---|---|
| `PreferredLoginDevice` | Mobile Phone is the largest group at 49.11%, followed by Computer at 29.02% and Phone at 21.87% |
| `PreferredPaymentMode` | Debit Card is the largest payment mode at 41.10%, followed by Credit Card at 26.66% |
| `Gender` | Male customers account for 60.11%, Female customers account for 39.89% |
| `PreferedOrderCat` | Laptop & Accessory is the largest category at 36.41%, followed by Mobile Phone at 22.58% |
| `MaritalStatus` | Married customers account for 53.04%, Single customers account for 31.90% |
| `Complain` | 28.49% of customers raised a complaint |
| `CityTier` | City Tier 1 accounts for 65.12% of customers |
| `SatisfactionScore` | Score 3 is the most common satisfaction score at 30.16% |

**Result image placeholder**

```markdown
![Categorical Distribution](images/categorical_distribution.png)
```

### 3.4 Missing Values and Data Cleaning

**Source code**

```python
df.isnull().mean() * 100
```

**Output**

| Column | Missing Rows | Missing Rate |
|---|---:|---:|
| DaySinceLastOrder | 307 | 5.45% |
| OrderAmountHikeFromlastYear | 265 | 4.71% |
| Tenure | 264 | 4.69% |
| OrderCount | 258 | 4.58% |
| CouponUsed | 256 | 4.55% |
| HourSpendOnApp | 255 | 4.53% |
| WarehouseToHome | 251 | 4.46% |

**Source code**

```python
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])
```

**Output**

```text
Missing value handling:
- Numerical missing values were filled with median values.
- No categorical columns had missing values.
- Missing values after cleaning: 0
```

**Source code**

```python
if 'PreferredLoginDevice' in df.columns:
    df['PreferredLoginDevice'] = df['PreferredLoginDevice'].replace({
        'Phone': 'Mobile Phone',
        'Mobile': 'Mobile Phone',
    })

if 'PreferredPaymentMode' in df.columns:
    df['PreferredPaymentMode'] = df['PreferredPaymentMode'].replace({
        'COD': 'Cash on Delivery',
        'CC': 'Credit Card',
    })

order_cat_col = None
for possible_col in ['PreferedOrderCat', 'PreferredOrderCat']:
    if possible_col in df.columns:
        order_cat_col = possible_col
        break

if order_cat_col is not None:
    df[order_cat_col] = df[order_cat_col].replace({
        'Mobile': 'Mobile Phone',
    })
```

**Output**

```text
Standardized category labels:
- PreferredLoginDevice: Phone/Mobile -> Mobile Phone
- PreferredPaymentMode: COD -> Cash on Delivery, CC -> Credit Card
- PreferedOrderCat: Mobile -> Mobile Phone
```

### 3.5 Outlier Detection

**Source code**

```python
for col in num_cols:
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot of {col}')
    plt.show()
```

**Output summary**

```text
Outlier review:
- WarehouseToHome has a strong right-skew pattern and a maximum value of 127.
- CouponUsed and OrderCount are right-skewed, indicating a small group of highly active coupon/order users.
- CashbackAmount is also right-skewed, suggesting some customers receive much higher cashback than the median.
```

**Result image placeholder**

```markdown
![Outlier Boxplots](images/outlier_boxplots.png)
```

### 3.6 Relationship Between Features and Churn

**Source code**

```python
df.groupby('Churn')[num_cols].mean()
df.groupby('Churn')[num_cols].median()
```

**Output: Mean comparison**

| Churn | Tenure | WarehouseToHome | HourSpendOnApp | Devices | Addresses | Amount Hike | Coupon Used | Order Count | Days Since Last Order | Cashback |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 11.40 | 15.31 | 2.93 | 3.64 | 4.16 | 15.69 | 1.72 | 2.99 | 4.71 | 180.64 |
| 1 | 3.86 | 16.86 | 2.96 | 3.93 | 4.47 | 15.62 | 1.71 | 2.81 | 3.22 | 160.37 |

**Output: Median comparison**

| Churn | Tenure | WarehouseToHome | HourSpendOnApp | Devices | Addresses | Amount Hike | Coupon Used | Order Count | Days Since Last Order | Cashback |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 10.00 | 14.00 | 3.00 | 4.00 | 3.00 | 15.00 | 1.00 | 2.00 | 3.00 | 166.12 |
| 1 | 1.00 | 14.00 | 3.00 | 4.00 | 3.00 | 15.00 | 1.00 | 2.00 | 2.50 | 149.66 |

**Source code**

```python
plt.figure(figsize=(12, 8))
sns.heatmap(df[num_cols + ['Churn']].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()
```

**Result image placeholder**

```markdown
![Correlation Heatmap](images/correlation_heatmap.png)
```

**Source code**

```python
for col in cat_cols:
    if col != 'Churn':
        churn_rate = df.groupby(col)['Churn'].mean().sort_values(ascending=False)
        print(churn_rate)
```

**Output: Churn rate by key categories**

| Category | Highest-Risk Group | Churn Rate |
|---|---|---:|
| PreferredLoginDevice | Computer | 19.83% |
| PreferredPaymentMode | Cash on Delivery | 24.90% |
| Gender | Male | 17.73% |
| PreferedOrderCat | Mobile Phone | 27.40% |
| MaritalStatus | Single | 26.73% |
| Complain | Complaint = 1 | 31.67% |
| CityTier | City Tier 3 | 21.37% |
| SatisfactionScore | Score 5 | 23.83% |

### 3.7 Churned User Behavior Analysis

**Source code**

```python
active_users = df[df['Churn'] == 0]
churned_users = df[df['Churn'] == 1]

print('Active users:', active_users.shape[0])
print('Churned users:', churned_users.shape[0])
print(df.groupby('Churn')[num_cols].mean())
```

**Output**

```text
Active users: 4,682
Churned users: 948

Key behavior patterns:
- Churned users have much shorter average tenure: 3.86 months versus 11.40 months for active users.
- Median tenure is 1 month for churned users and 10 months for active users.
- Churned users receive lower average cashback: 160.37 versus 180.64 for active users.
- Customers with complaints have a churn rate of 31.67%, compared with 10.93% for customers without complaints.
- Churned users should not be treated as one single group because their behavior differs by tenure, value, complaint status, payment mode, and delivery context.
```

### 3.8 Supervised Learning Preparation

**Source code**

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

model_cat_cols = [col for col in cat_cols if col != 'Churn']
df_encoded = pd.get_dummies(df, columns=model_cat_cols, drop_first=True, dtype=int)

x = df_encoded.drop(columns=['CustomerID', 'Churn'])
y = df_encoded['Churn']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
```

**Output**

```text
Train-test split:
- Train size: 4,504 rows
- Test size: 1,126 rows
- Stratified by target variable Churn
```

### 3.9 Logistic Regression Model

**Source code**

```python
log_model = LogisticRegression(max_iter=1000, class_weight='balanced')
log_model.fit(x_train_scaled, y_train)

y_test_pred = log_model.predict(x_test_scaled)

print('Logistic Regression')
print('Balanced Accuracy:', balanced_accuracy_score(y_test, y_test_pred))
print(confusion_matrix(y_test, y_test_pred))
print(classification_report(y_test, y_test_pred))
```

**Output**

```text
Logistic Regression
Balanced Accuracy: 0.8053699955015745

Confusion Matrix:
[[749, 187],
 [ 36, 154]]

Classification Report:
Class 0 precision: 0.95, recall: 0.80, f1-score: 0.87
Class 1 precision: 0.45, recall: 0.81, f1-score: 0.58
Accuracy: 0.80
```

### 3.10 Random Forest Model

**Source code**

```python
rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight='balanced'
)

rf_model.fit(x_train, y_train)

y_test_pred = rf_model.predict(x_test)

print('Random Forest')
print('Balanced Accuracy:', balanced_accuracy_score(y_test, y_test_pred))
print(confusion_matrix(y_test, y_test_pred))
print(classification_report(y_test, y_test_pred))
```

**Output**

```text
Random Forest
Balanced Accuracy: 0.9215080971659919

Confusion Matrix:
[[927, 9],
 [ 28, 162]]

Classification Report:
Class 0 precision: 0.97, recall: 0.99, f1-score: 0.98
Class 1 precision: 0.95, recall: 0.85, f1-score: 0.90
Accuracy: 0.97
```

### 3.11 Hyperparameter Tuning

**Source code**

```python
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'bootstrap': [True, False],
}

grid_search = GridSearchCV(
    estimator=rf_model,
    param_grid=param_grid,
    cv=5,
    scoring='balanced_accuracy'
)

grid_search.fit(x_train, y_train)

print('Best Parameters:')
print(grid_search.best_params_)
```

**Output**

```text
Best Parameters:
bootstrap: False
max_depth: None
min_samples_leaf: 2
min_samples_split: 5
n_estimators: 200
```

### 3.12 Tuned Random Forest Evaluation

**Source code**

```python
best_rf_model = grid_search.best_estimator_

y_pred_best_rf = best_rf_model.predict(x_test)

print('Tuned Random Forest')
print('Balanced Accuracy:', balanced_accuracy_score(y_test, y_pred_best_rf))
print(confusion_matrix(y_test, y_pred_best_rf))
print(classification_report(y_test, y_pred_best_rf))
```

**Output**

```text
Tuned Random Forest
Balanced Accuracy: 0.956174089068826

Confusion Matrix:
[[918, 18],
 [ 13, 177]]

Classification Report:
Class 0 precision: 0.99, recall: 0.98, f1-score: 0.98
Class 1 precision: 0.91, recall: 0.93, f1-score: 0.92
Accuracy: 0.97
```

### 3.13 Model Comparison and Advanced Evaluation

**Source code**

```python
model_predictions = {
    'Logistic Regression': {
        'y_pred': log_model.predict(x_test_scaled),
        'y_proba': log_model.predict_proba(x_test_scaled)[:, 1],
    },
    'Random Forest': {
        'y_pred': rf_model.predict(x_test),
        'y_proba': rf_model.predict_proba(x_test)[:, 1],
    },
    'Tuned Random Forest': {
        'y_pred': best_rf_model.predict(x_test),
        'y_proba': best_rf_model.predict_proba(x_test)[:, 1],
    },
}

model_results = []

for model_name, pred_data in model_predictions.items():
    y_pred = pred_data['y_pred']
    y_proba = pred_data['y_proba']

    model_results.append({
        'Model': model_name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Balanced Accuracy': balanced_accuracy_score(y_test, y_pred),
        'Precision (Churn=1)': precision_score(y_test, y_pred),
        'Recall (Churn=1)': recall_score(y_test, y_pred),
        'F1-score (Churn=1)': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_proba),
        'PR-AUC': average_precision_score(y_test, y_proba),
    })

model_results_df = pd.DataFrame(model_results).sort_values(
    'Balanced Accuracy', ascending=False
)
model_results_df
```

**Output**

| Model | Accuracy | Balanced Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.80 | 0.805 | 0.45 | 0.81 | 0.58 |
| Random Forest | 0.97 | 0.922 | 0.95 | 0.85 | 0.90 |
| Tuned Random Forest | 0.97 | 0.956 | 0.91 | 0.93 | 0.92 |

**Source code**

```python
ConfusionMatrixDisplay.from_estimator(
    best_rf_model,
    x_test,
    y_test,
    display_labels=['Not Churn', 'Churn'],
    cmap='Blues'
)
plt.title('Confusion Matrix - Tuned Random Forest')
plt.show()
```

**Result image placeholders**

```markdown
![Model Comparison](images/model_comparison.png)
![Confusion Matrix](images/confusion_matrix.png)
![ROC and PR Curves](images/roc_pr_curves.png)
![Threshold Tuning](images/threshold_tuning.png)
![Feature Importance](images/feature_importance.png)
```

### 3.14 Threshold Tuning

**Source code**

```python
y_proba_best_rf = best_rf_model.predict_proba(x_test)[:, 1]
thresholds = np.arange(0.10, 0.91, 0.05)
threshold_results = []

for threshold in thresholds:
    y_pred_threshold = (y_proba_best_rf >= threshold).astype(int)
    threshold_results.append({
        'Threshold': threshold,
        'Balanced Accuracy': balanced_accuracy_score(y_test, y_pred_threshold),
        'Precision (Churn=1)': precision_score(y_test, y_pred_threshold, zero_division=0),
        'Recall (Churn=1)': recall_score(y_test, y_pred_threshold),
        'F1-score (Churn=1)': f1_score(y_test, y_pred_threshold),
    })

threshold_results_df = pd.DataFrame(threshold_results)
threshold_results_df.sort_values('F1-score (Churn=1)', ascending=False).head(10)
```

**Output placeholder**

```markdown
![Threshold Tuning Result](images/threshold_tuning_result.png)
```

### 3.15 Churned Customer Segmentation

**Source code**

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

churn_df = df[df['Churn'] == 1].copy()
cluster_data = churn_df.drop(columns=['CustomerID', 'Churn'])
cluster_data_encoded = pd.get_dummies(cluster_data, drop_first=True, dtype=int)

scaler = StandardScaler()
cluster_data_scaled = scaler.fit_transform(cluster_data_encoded)
```

**Output**

```text
Churned customer dataset shape: (948, 20)
```

**Source code**

```python
inertia = []
K_range = range(2, 20)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(cluster_data_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(K_range, inertia, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()
```

**Result image placeholder**

```markdown
![Elbow Method](images/elbow_method.png)
```

**Source code**

```python
optimal_k = 7

kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
churn_df['Segment'] = kmeans.fit_predict(cluster_data_scaled)

churn_df['Segment'].value_counts(normalize=True) * 100
```

**Output**

| Segment | Share of Churned Customers |
|---|---:|
| Segment 3 | 31.65% |
| Segment 1 | 23.52% |
| Segment 2 | 19.51% |
| Segment 4 | 13.50% |
| Segment 6 | 7.59% |
| Segment 5 | 2.11% |
| Segment 0 | 2.11% |

**Source code**

```python
silhouette_avg = silhouette_score(cluster_data_scaled, churn_df['Segment'])
print(f'Silhouette Score for k={optimal_k}: {silhouette_avg:.4f}')
```

**Result output placeholder**

```markdown
![Silhouette Score Output](images/silhouette_score_output.png)
```

### 3.16 Segment Profiling

**Source code**

```python
num_cols_cluster = churn_df.select_dtypes(include=np.number).columns.tolist()
num_cols_cluster = [
    col for col in num_cols_cluster
    if col not in ['CustomerID', 'Churn', 'Segment']
]

churn_df.groupby('Segment')[num_cols_cluster].mean().T
```

**Output summary**

| Segment | Main Characteristics | Suggested Action |
|---|---|---|
| Segment 0 | Small high-value churn group with longer tenure, high order count, high cashback, and high coupon usage | Use premium win-back offers and direct recovery campaigns |
| Segment 1 | Medium-tenure churn group with longer warehouse-to-home distance and higher number of addresses | Provide delivery support, shipping incentives, and personalized coupons |
| Segment 2 | Newer churn group with short tenure, low cashback, and high complaint rate | Improve onboarding, complaint recovery, and early lifecycle incentives |
| Segment 3 | Largest churn group with short tenure, low cashback, and high complaint rate | Build broad retention campaigns focused on service recovery and second-purchase incentives |
| Segment 4 | City Tier 3 customers with long delivery distance and E-wallet payment behavior | Improve delivery communication and offer payment-specific promotions |
| Segment 5 | Small but higher-value group with longer tenure, high order count, and high cashback | Use premium reactivation offers and priority customer support |
| Segment 6 | Short-tenure UPI users with long delivery distance and high complaint rate | Use payment-specific incentives and delivery issue resolution |

**Source code**

```python
pca = PCA(n_components=2, random_state=42)
cluster_pca = pca.fit_transform(cluster_data_scaled)

pca_df = pd.DataFrame({
    'PCA1': cluster_pca[:, 0],
    'PCA2': cluster_pca[:, 1],
    'Segment': churn_df['Segment'].astype(str),
})

sns.scatterplot(data=pca_df, x='PCA1', y='PCA2', hue='Segment')
plt.title('PCA Visualization of Churned Customer Segments')
plt.show()
```

**Result image placeholder**

```markdown
![Churned Customer Segments PCA](images/churned_customer_segments_pca.png)
```

---

## 4. Final Results

### 4.1 Dataset and EDA Results

- The dataset contains 5,630 customers and 20 variables.
- The target variable is imbalanced: 83.16% active customers and 16.84% churned customers.
- 7 numerical columns contain missing values, with missing rates between 4.46% and 5.45%.
- Churned customers have much shorter tenure than active customers.
- Complaint behavior is one of the strongest churn indicators.
- Churned customers receive lower average cashback.
- Certain categories show higher churn risk, especially Mobile Phone order category, Single marital status, Cash on Delivery, E wallet, and City Tier 3.

### 4.2 Model Results

| Model | Accuracy | Balanced Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.80 | 0.805 | 0.45 | 0.81 | 0.58 |
| Random Forest | 0.97 | 0.922 | 0.95 | 0.85 | 0.90 |
| Tuned Random Forest | 0.97 | 0.956 | 0.91 | 0.93 | 0.92 |

Best model:

```text
Tuned Random Forest
Balanced Accuracy: 0.956
Recall for churned customers: 0.93
F1-score for churned customers: 0.92
```

### 4.3 Segmentation Results

- K-Means clustering was applied only to churned customers.
- The notebook uses `k = 7` segments.
- Segment 3 is the largest churned-customer group at 31.65%.
- Segment 1 and Segment 2 are also major churned groups, accounting for 23.52% and 19.51% respectively.
- The segmentation output shows that churned customers should receive different retention actions based on tenure, complaint behavior, delivery context, payment mode, and customer value.

---

## 5. Business Recommendations

1. **Reduce early lifecycle churn**

   Churned customers have an average tenure of only 3.86 months, while active customers have an average tenure of 11.40 months. The company should provide welcome vouchers, onboarding messages, first-month cashback, and second-purchase incentives for new users.

2. **Prioritize complaint recovery**

   Customers with complaints have a churn rate of 31.67%, compared with 10.93% for customers without complaints. Complaint cases should be treated as high-risk cases and followed up with faster resolution, service recovery, and compensation offers.

3. **Use the tuned Random Forest model for retention targeting**

   The tuned Random Forest model achieves the best balanced accuracy at 0.956 and recall of 0.93 for churned customers. This model can be used to score customers and prioritize retention campaigns.

4. **Target high-risk category and payment groups**

   Mobile Phone order category customers show a churn rate of 27.40%. Cash on Delivery and E wallet customers also show higher churn rates. The company should customize retention campaigns by order category and payment behavior.

5. **Improve delivery experience for high-risk locations**

   City Tier 3 customers show a higher churn rate at 21.37%. Customers with longer warehouse-to-home distance may also experience delivery inconvenience. The company should improve delivery estimates, shipping support, and warehouse allocation.

6. **Use segment-based win-back campaigns**

   Churned customers are not one single group. High-value churned customers should receive premium win-back offers, while short-tenure and complaint-heavy customers should receive onboarding and service recovery campaigns.

---

## 6. Conclusion

This project shows that churn prediction can help the business identify high-risk customers before they leave. The analysis indicates that churn is strongly related to tenure, complaint behavior, cashback, order category, payment mode, marital status, and city tier.

Among the tested models, the tuned Random Forest performs best, with a balanced accuracy of 0.956. In addition, K-Means clustering provides a more detailed view of churned customers by separating them into 7 behavioral segments. These results support a more targeted retention strategy instead of applying the same promotion to every customer.

---

## 7. How To Run

Clone the repository:

```bash
git clone https://github.com/Tom2702/ML_Ecommerce_Churn_Prediction_Python.git
cd ML_Ecommerce_Churn_Prediction_Python
```

Install dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib openpyxl
```

Run the notebook:

```text
ML_Project.ipynb
```

Recommended workflow:

1. Place `Churn_prediction.xlsx` or `churn_predict.csv` in the same directory as the notebook.
2. Run all notebook cells from top to bottom.
3. Export charts into the `images/` folder.
4. Replace the placeholder image paths in this README with the exported result images.
5. Upload the notebook, README, dataset, and result images to GitHub.

---

## 8. Repository Structure

```text
.
├── ML_Project.ipynb
├── Churn_prediction.xlsx
├── README.md
└── images/
    ├── project_banner.png
    ├── dataset_preview.png
    ├── numerical_distribution.png
    ├── categorical_distribution.png
    ├── outlier_boxplots.png
    ├── correlation_heatmap.png
    ├── model_comparison.png
    ├── confusion_matrix.png
    ├── roc_pr_curves.png
    ├── threshold_tuning.png
    ├── feature_importance.png
    ├── elbow_method.png
    └── churned_customer_segments_pca.png
```

