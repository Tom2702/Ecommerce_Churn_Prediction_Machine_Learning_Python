# E-Commerce Churn Prediction | Machine Learning

---

## 1. Introduction

This project analyzes customer churn behavior for an e-commerce company and builds machine learning models to predict whether a customer is likely to churn.

In this project, we will:

- Load and inspect the e-commerce churn dataset.
- Clean missing values and inconsistent category labels.
- Analyze churn behavior across numerical and categorical variables.
- Build classification models to predict churn.
- Tune the best model using GridSearchCV.
- Segment churned customers using K-Means clustering.
- Provide business recommendations for customer retention.

The main business goal is to help the company identify high-risk customers earlier and design more targeted retention campaigns.

---

## 2. Dataset

The dataset used in this project is:

```text
churn_predict.csv
```

or, in the original notebook environment:

```text
Churn_prediction.xlsx
```

Main fields include:

| Column | Description |
|---|---|
| `CustomerID` | Unique customer ID |
| `Churn` | Target variable, whether the customer churned |
| `Tenure` | Customer tenure with the company |
| `PreferredLoginDevice` | Preferred login device |
| `CityTier` | Customer city tier |
| `WarehouseToHome` | Distance from warehouse to customer home |
| `PreferredPaymentMode` | Preferred payment method |
| `Gender` | Customer gender |
| `HourSpendOnApp` | Time spent on app or website |
| `NumberOfDeviceRegistered` | Number of registered devices |
| `PreferedOrderCat` | Preferred order category |
| `SatisfactionScore` | Customer satisfaction score |
| `MaritalStatus` | Customer marital status |
| `NumberOfAddress` | Number of registered addresses |
| `Complain` | Whether the customer raised a complaint |
| `CouponUsed` | Number of coupons used |
| `OrderCount` | Number of orders |
| `DaySinceLastOrder` | Days since last order |
| `CashbackAmount` | Cashback received by the customer |

Target variable:

```text
Churn
```

- `0`: Not churned
- `1`: Churned

---

## 3. Repository Structure

This repo contains the following main files and folders:

```text
.
├── ML_Project.ipynb
└── README.md
```

Notes:

- `ML_Project.ipynb` contains the full analysis and model-building workflow.
- `images/` is reserved for charts exported from the notebook.
- `.pkl` files are optional outputs after running the model-saving cells.

---

## 4. Load Data

The notebook uses a flexible loader so the project can run in Google Colab or locally.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_excel("/content/drive/MyDrive/Churn_prediction.xlsx")
```
<img width="2956" height="347" alt="image" src="https://github.com/user-attachments/assets/e306e62a-8d3b-485d-bdaf-01b8ff70cec3" />

---

## 5. Data Inspection

Basic inspection is performed to understand dataset size, columns, data types, and missing values.

```python
df.head()
df.shape
df.info()
df.isnull().mean() * 100
```

Feature groups are separated into ID, numerical, categorical, and target columns.

```python
id_cols = ["CustomerID"]
cat_manual = ["Churn", "Complain", "CityTier", "SatisfactionScore"]

num_cols = df.select_dtypes(include="number").columns.tolist()
num_cols = [col for col in num_cols if col not in id_cols and col not in cat_manual]

cat_cols = df.select_dtypes(include="object").columns.tolist()
cat_cols = cat_cols + [col for col in cat_manual if col in df.columns]
```

---

## 6. Data Cleaning

Missing values are handled using median values for numerical columns and mode values for categorical columns.

```python
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

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

---

## 7. Exploratory Data Analysis

The EDA section analyzes both numerical and categorical variables.

### 7.1 Numerical Distribution

```python
df[num_cols].describe().T

df[num_cols].skew()

for col in num_cols:
    sns.histplot(df[col], kde=True)
    plt.title(f"Distribution of {col}")
    plt.show()
```

### 7.2 Categorical Distribution

```python
for col in cat_cols:
    print(col)
    print(df[col].value_counts())
    print(df[col].value_counts(normalize=True) * 100)
```

### 7.3 Relationship Between Features and Churn

```python
df.groupby("Churn")[num_cols].mean()
df.groupby("Churn")[num_cols].median()

for col in num_cols:
    sns.boxplot(data=df, x="Churn", y=col)
    plt.title(f"{col} by Churn")
    plt.show()
```

Correlation heatmap:

```python
plt.figure(figsize=(12, 8))
sns.heatmap(df[num_cols + ["Churn"]].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()
```

Churn rate by category:

```python
for col in cat_cols:
    if col != "Churn":
        churn_rate = df.groupby(col)["Churn"].mean().sort_values(ascending=False)
        print(churn_rate)
```

### 7.4 Result Placeholder

Add EDA charts here after exporting images from the notebook.

```markdown
![Churn Distribution](images/churn_distribution.png)
![Correlation Heatmap](images/correlation_heatmap.png)
![Churn Rate by Category](images/churn_rate_by_category.png)
```

---

## 8. Supervised Learning

The project uses supervised learning to predict customer churn.

Models used:

1. Logistic Regression
2. Random Forest Classifier
3. Tuned Random Forest Classifier

### 8.1 Train-Test Split

```python
from sklearn.model_selection import train_test_split

model_cat_cols = [col for col in cat_cols if col != "Churn"]
df_encoded = pd.get_dummies(df, columns=model_cat_cols, drop_first=True, dtype=int)

x = df_encoded.drop(columns=["CustomerID", "Churn"])
y = df_encoded["Churn"]

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)
```

### 8.2 Logistic Regression

```python
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

log_model = LogisticRegression(max_iter=1000, class_weight="balanced")
log_model.fit(x_train_scaled, y_train)
y_test_pred = log_model.predict(x_test_scaled)
```

### 8.3 Random Forest

```python
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",
)

rf_model.fit(x_train, y_train)
y_test_pred = rf_model.predict(x_test)
```

---

## 9. Hyperparameter Tuning

GridSearchCV is used to tune the Random Forest model.

```python
from sklearn.model_selection import GridSearchCV

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

best_rf_model = grid_search.best_estimator_
print(grid_search.best_params_)
```

Result placeholder:

```text
Best Parameters:
[Insert best parameters here]
```

---

## 10. Model Evaluation

Balanced accuracy is used because the churn dataset is imbalanced.

```python
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    classification_report,
)

y_pred_best_rf = best_rf_model.predict(x_test)

print("Balanced Accuracy:", balanced_accuracy_score(y_test, y_pred_best_rf))
print(confusion_matrix(y_test, y_pred_best_rf))
print(classification_report(y_test, y_pred_best_rf))
```

Additional evaluation code was added for:

- Confusion matrix
- ROC Curve
- Precision-Recall Curve
- Threshold tuning
- Feature importance

### 10.1 Threshold Tuning

```python
y_proba_best_rf = best_rf_model.predict_proba(x_test)[:, 1]
thresholds = np.arange(0.10, 0.91, 0.05)

threshold_results = []
for threshold in thresholds:
    y_pred_threshold = (y_proba_best_rf >= threshold).astype(int)
    threshold_results.append({
        "Threshold": threshold,
        "Balanced Accuracy": balanced_accuracy_score(y_test, y_pred_threshold),
        "Precision": precision_score(y_test, y_pred_threshold, zero_division=0),
        "Recall": recall_score(y_test, y_pred_threshold),
        "F1-score": f1_score(y_test, y_pred_threshold),
    })

threshold_results_df = pd.DataFrame(threshold_results)
threshold_results_df.head()
```

### 10.2 Feature Importance

```python
feature_importance_df = pd.DataFrame({
    "Feature": x_train.columns,
    "Importance": best_rf_model.feature_importances_,
}).sort_values("Importance", ascending=False)

feature_importance_df.head(20)
```

### 10.3 Save Model

```python
import joblib

joblib.dump(best_rf_model, "tuned_random_forest_churn_model.pkl")
joblib.dump(scaler, "standard_scaler.pkl")
```

---

## 11. Unsupervised Learning

K-Means clustering is applied only to churned customers.

```python
churn_df = df[df["Churn"] == 1].copy()
cluster_data = churn_df.drop(columns=["CustomerID", "Churn"])
cluster_data_encoded = pd.get_dummies(cluster_data, drop_first=True, dtype=int)
```

Scale features before K-Means:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
cluster_data_scaled = scaler.fit_transform(cluster_data_encoded)
```

Elbow Method:

```python
from sklearn.cluster import KMeans

inertia = []
K_range = range(2, 20)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(cluster_data_scaled)
    inertia.append(kmeans.inertia_)

plt.plot(K_range, inertia, marker="o")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.show()
```

Train final K-Means model:

```python
optimal_k = 7

kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
churn_df["Segment"] = kmeans.fit_predict(cluster_data_scaled)
```

Cluster profiling:

```python
churn_df["Segment"].value_counts(normalize=True) * 100

num_cols_cluster = churn_df.select_dtypes(include=np.number).columns.tolist()
num_cols_cluster = [
    col for col in num_cols_cluster
    if col not in ["CustomerID", "Churn", "Segment"]
]

churn_df.groupby("Segment")[num_cols_cluster].mean().T
```

---

## 12. Results

This section is intentionally left with placeholders so final charts and outputs can be added after running the notebook.

### 12.1 Dataset Result

```text
Dataset shape:
[Insert dataset shape here]

Churn distribution:
[Insert churn distribution here]
```

### 12.2 EDA Result

Add EDA charts here:

```markdown
![Churn Distribution](images/churn_distribution.png)
![Correlation Heatmap](images/correlation_heatmap.png)
![Churn Rate by Category](images/churn_rate_by_category.png)
```

Key EDA findings:

```text
[Insert main EDA insight 1]
[Insert main EDA insight 2]
[Insert main EDA insight 3]
```

### 12.3 Model Result

Model comparison table:

| Model | Balanced Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Logistic Regression | `[Insert]` | `[Insert]` | `[Insert]` | `[Insert]` |
| Random Forest | `[Insert]` | `[Insert]` | `[Insert]` | `[Insert]` |
| Tuned Random Forest | `[Insert]` | `[Insert]` | `[Insert]` | `[Insert]` |

Add model result images here:

```markdown
![Model Comparison](images/model_comparison.png)
![Confusion Matrix](images/confusion_matrix.png)
![ROC and PR Curves](images/roc_pr_curves.png)
![Threshold Tuning](images/threshold_tuning.png)
![Feature Importance](images/feature_importance.png)
```

### 12.4 Best Model Result

```text
Best Model:
[Insert best model name]

Best Parameters:
[Insert best model parameters]

Best Balanced Accuracy:
[Insert balanced accuracy]
```

### 12.5 Clustering Result

Add clustering result images here:

```markdown
![Elbow Method](images/elbow_method.png)
![Churned Customer Segments PCA](images/churned_customer_segments_pca.png)
```

Segment summary table:

| Segment | Main Characteristics | Suggested Action |
|---|---|---|
| Segment 0 | `[Insert]` | `[Insert]` |
| Segment 1 | `[Insert]` | `[Insert]` |
| Segment 2 | `[Insert]` | `[Insert]` |
| Segment 3 | `[Insert]` | `[Insert]` |
| Segment 4 | `[Insert]` | `[Insert]` |
| Segment 5 | `[Insert]` | `[Insert]` |
| Segment 6 | `[Insert]` | `[Insert]` |

---

## 13. Business Recommendations

Based on the churn analysis and model outputs, recommended actions include:

1. **New Customer Retention**
   - Target customers with short tenure using welcome vouchers, onboarding messages, and second-purchase incentives.

2. **Complaint Handling**
   - Prioritize customers who submitted complaints because complaint behavior is strongly related to churn risk.

3. **Personalized Promotions**
   - Use churn probability and customer behavior to send targeted coupons or cashback offers instead of mass promotions.

4. **Delivery Experience Improvement**
   - Review customers with high warehouse-to-home distance and offer delivery support where needed.

5. **High-Value Win-Back**
   - Identify churned customers with high historical value and send premium win-back offers.

---

## 14. How to Run

1. Clone this repository.
2. Place `churn_predict.csv` in the same directory as the notebook.
3. Open `ML_Project.ipynb` in Jupyter Notebook or Google Colab.
4. Run all cells from top to bottom.
5. Export final charts to the `images/` folder.
6. Update the `Results` section in this README with final outputs.

```bash
git clone https://github.com/Tom2702/ML_Ecommerce_Churn_Prediction_Python.git
cd ML_Ecommerce_Churn_Prediction_Python
```

Install required packages if needed:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib openpyxl
```
