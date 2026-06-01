# E-Commerce Churn Prediction | Machine Learning

Author: Anh Tuan Nguyen  
Tools Used: Python, pandas, numpy, matplotlib, seaborn, scikit-learn, Jupyter Notebook / Google Colab

## Table of Contents

1. [Background & Overview](#1-background--overview)
2. [Project Objectives](#2-project-objectives)
3. [Dataset Description](#3-dataset-description)
4. [Project Workflow](#4-project-workflow)
5. [Exploratory Data Analysis](#5-exploratory-data-analysis)
6. [Supervised Learning](#6-supervised-learning)
7. [Unsupervised Learning](#7-unsupervised-learning)
8. [Results](#8-results)
9. [Business Recommendations](#9-business-recommendations)
10. [Technologies Used](#10-technologies-used)
11. [How To Run](#11-how-to-run)
12. [Repository Structure](#12-repository-structure)
13. [Conclusion](#13-conclusion)

---

## 1. Background & Overview

Customer churn is one of the key challenges for e-commerce businesses. When customers stop purchasing or interacting with the platform, the company loses future revenue and must spend more on customer acquisition.

This project analyzes customer churn behavior for an e-commerce company and builds machine learning models to predict whether a customer is likely to churn. In addition to churn prediction, the project segments churned customers into behavioral groups to support targeted retention and win-back strategies.

The project combines:

- Exploratory data analysis
- Data cleaning and preprocessing
- Supervised machine learning for churn prediction
- Hyperparameter tuning
- Unsupervised learning for churned customer segmentation
- Business recommendations for customer retention

---

## 2. Project Objectives

The main objectives of this project are:

1. Perform exploratory data analysis to understand customer behavior.
2. Identify important factors related to customer churn.
3. Clean and preprocess numerical and categorical features.
4. Build supervised machine learning models to predict churn.
5. Tune the best model using hyperparameter tuning.
6. Segment churned customers using unsupervised learning.
7. Provide business recommendations to reduce churn and improve retention.

---

## 3. Dataset Description

The dataset used in this project is:

```text
churn_predict.csv
```

The dataset contains customer profile, behavior, transaction, complaint, and promotion-related information.

### 3.1 Dataset Summary

Based on the notebook output:

| Metric | Value |
|---|---:|
| Total records | `5,630` |
| Total columns | `20` |
| Non-churned customers | `4,682` |
| Churned customers | `948` |
| Churn rate | `16.84%` |
| Non-churn rate | `83.16%` |

The dataset is imbalanced, so model evaluation uses balanced accuracy and churn-class recall instead of relying only on normal accuracy.

### 3.2 Main Columns

| Column | Description |
|---|---|
| `CustomerID` | Unique customer ID |
| `Churn` | Target variable indicating whether a customer churned |
| `Tenure` | Customer tenure with the company |
| `PreferredLoginDevice` | Preferred login device |
| `CityTier` | Customer city tier |
| `WarehouseToHome` | Distance from warehouse to customer home |
| `PreferredPaymentMode` | Preferred payment method |
| `Gender` | Customer gender |
| `HourSpendOnApp` | Time spent on app or website |
| `NumberOfDeviceRegistered` | Number of registered devices |
| `PreferredOrderCat` | Preferred order category |
| `SatisfactionScore` | Customer satisfaction score |
| `MaritalStatus` | Customer marital status |
| `NumberOfAddress` | Number of registered addresses |
| `Complain` | Whether the customer raised a complaint |
| `CouponUsed` | Number of coupons used |
| `OrderCount` | Number of orders |
| `DaySinceLastOrder` | Days since last order |
| `CashbackAmount` | Cashback received by the customer |

### 3.3 Target Variable

The target variable is:

```text
Churn
```

It indicates whether a customer has churned:

- `0`: Not churned
- `1`: Churned

---

## 4. Project Workflow

### 4.1 Data Loading and Initial Inspection

The dataset is loaded and inspected using pandas functions such as:

- `head()`
- `shape`
- `info()`
- Missing value checks
- Duplicate checks
- Descriptive statistics

### 4.2 Feature Classification

Features are separated into:

- ID columns
- Numerical variables
- Categorical variables
- Target variable

The target variable is:

```text
Churn
```

### 4.3 Data Distribution Analysis

The project analyzes the distribution of numerical and categorical variables using:

- Descriptive statistics
- Skewness analysis
- Histograms
- Count plots
- Boxplots

This step helps identify skewed variables, outliers, class imbalance, and customer behavior patterns.

### 4.4 Missing Values and Data Cleaning

Missing values are handled using:

- Median values for numerical columns
- Mode values for categorical columns

Inconsistent category labels are standardized.

Examples:

| Original Value | Standardized Value |
|---|---|
| `COD` | `Cash on Delivery` |
| `CC` | `Credit Card` |
| `Mobile` | `Mobile Phone` |

### 4.5 Feature and Churn Relationship Analysis

The relationship between customer features and churn is analyzed using:

- Grouped descriptive statistics
- Boxplots by churn status
- Correlation heatmap
- Churn rate by category

Important churn-related factors include:

- Tenure
- Complaint status
- Cashback amount
- Warehouse-to-home distance
- Order behavior
- Days since last order
- Customer engagement

---

## 5. Exploratory Data Analysis

### 5.1 Data Quality Overview

The notebook identified missing values in several numerical columns. Missing values were handled using median imputation for numerical variables and mode imputation for categorical variables.

Highest missing value rates:

| Column | Missing Rate |
|---|---:|
| `DaySinceLastOrder` | `5.45%` |
| `OrderAmountHikeFromlastYear` | `4.71%` |
| `Tenure` | `4.69%` |
| `OrderCount` | `4.58%` |
| `CouponUsed` | `4.55%` |
| `HourSpendOnApp` | `4.53%` |
| `WarehouseToHome` | `4.46%` |

The missing value level is moderate and can be handled without dropping a large number of records.

### 5.2 Churned Customer Behavior

The analysis shows clear behavioral differences between churned and retained customers.

| Metric | Non-Churn Mean | Churn Mean | Interpretation |
|---|---:|---:|---|
| `Tenure` | `11.40` | `3.86` | Churned customers have much shorter tenure |
| `WarehouseToHome` | `15.31` | `16.86` | Churned customers live slightly farther from warehouse |
| `NumberOfDeviceRegistered` | `3.64` | `3.93` | Churned customers register slightly more devices |
| `Complain` | `0.27` | `0.54` | Churned customers complain more often |
| `DaySinceLastOrder` | `4.71` | `3.22` | Churned customers show different recent order behavior |
| `CashbackAmount` | `180.64` | `160.37` | Churned customers receive lower cashback on average |

The strongest difference is tenure: the median tenure of churned customers is `1`, compared with `10` for non-churned customers.

### 5.3 Churn Rate by Customer Attribute

Several customer attributes show higher churn risk.

| Feature | Highest-Risk Group | Churn Rate |
|---|---|---:|
| `Complain` | Customers with complaint | `31.67%` |
| `PreferredOrderCat` | Mobile Phone | `27.40%` |
| `MaritalStatus` | Single | `26.73%` |
| `PreferredPaymentMode` | Cash on Delivery | `24.90%` |
| `PreferredPaymentMode` | E wallet | `22.80%` |
| `CityTier` | Tier 3 | `21.37%` |
| `PreferredLoginDevice` | Computer | `19.83%` |
| `SatisfactionScore` | Score 5 | `23.83%` |

### 5.4 Business Interpretation

Churn risk is associated with a combination of short tenure, complaints, lower cashback, product preference, payment method, and city tier.

Key interpretation:

- Newer customers are more vulnerable to churn.
- Complaint history is a strong churn signal.
- Customers preferring Mobile Phone category show higher churn risk.
- Customers using Cash on Delivery or E-wallet show higher churn rates than credit/debit card users.
- Tier 3 city customers may need additional delivery or service support.

### 5.5 Placeholder for EDA Visuals

After running the notebook, add exported EDA images here if needed.

```markdown
![Churn Distribution](images/churn_distribution.png)
![Correlation Heatmap](images/correlation_heatmap.png)
![Churn Rate by Category](images/churn_rate_by_category.png)
```


---

## 6. Supervised Learning

The project uses supervised learning to predict whether a customer is likely to churn.

### 6.1 Models Used

The following models are tested:

1. Logistic Regression
2. Random Forest Classifier
3. Tuned Random Forest Classifier using GridSearchCV

### 6.2 Preprocessing

Preprocessing steps include:

- Removing `CustomerID` because it is only an identifier
- Using `Churn` as the target variable
- One-hot encoding categorical variables
- Applying standard scaling for Logistic Regression
- Splitting the dataset into training and testing sets
- Using stratified sampling to preserve churn class distribution

### 6.3 Evaluation Metric

Balanced accuracy is used because the churn classes are imbalanced.

Balanced accuracy is more suitable than normal accuracy when one class appears much more frequently than the other.

### 6.4 Model Performance

| Model | Balanced Accuracy |
|---|---:|
| Logistic Regression | 0.805 |
| Random Forest | 0.922 |
| Tuned Random Forest | 0.956 |

The tuned Random Forest model achieved the best performance.

### 6.5 Best Model Parameters

The best Random Forest parameters are:

```text
bootstrap: False
max_depth: None
min_samples_leaf: 2
min_samples_split: 5
n_estimators: 200
```

### 6.6 Model Result Interpretation

The tuned Random Forest model performs best because it can capture non-linear relationships between customer behavior and churn.

This is useful for churn prediction because churn behavior is usually influenced by multiple interacting factors such as tenure, complaint history, order activity, cashback, and engagement.

---

## 7. Unsupervised Learning

K-Means clustering is applied only to customers with:

```text
Churn == 1
```

The purpose is to segment churned customers into different behavioral groups.

### 7.1 Clustering Workflow

The clustering process includes:

1. Filter churned customers.
2. Remove ID and target columns.
3. One-hot encode categorical variables.
4. Scale features using `StandardScaler`.
5. Use the Elbow Method to select the number of clusters.
6. Train KMeans with `k = 7`.
7. Compare numerical and categorical characteristics across segments.

### 7.2 Segment Insights

The churned customer segments show different behavioral patterns:

1. Some segments contain new users with low tenure and low engagement.
2. Some segments have high complaint rates, suggesting poor customer experience.
3. Some groups have longer warehouse-to-home distances, indicating possible delivery inconvenience.
4. Some churned users are high-value customers with high order count, high cashback, and longer tenure.

### 7.3 Business Interpretation

The segmentation results show that churned customers should not be treated as one single group.

Different churned customer segments require different retention or win-back strategies:

- New users may need onboarding support or welcome incentives.
- Complaint-driven churners need service recovery and faster issue resolution.
- Distance-sensitive customers may need delivery support.
- High-value churned customers should receive personalized win-back offers.

---

## 8. Results

This section summarizes the actual outputs from the notebook and provides placeholders for GitHub result images.

### 8.1 Classification Model Results

The project tested three supervised learning models. Balanced accuracy was used because the churn class is imbalanced.

| Model | Balanced Accuracy | Churn Precision | Churn Recall | Churn F1-score |
|---|---:|---:|---:|---:|
| Logistic Regression | `0.805` | `0.45` | `0.81` | `0.58` |
| Random Forest | `0.922` | `0.95` | `0.85` | `0.90` |
| Tuned Random Forest | `0.956` | `0.91` | `0.93` | `0.92` |

The tuned Random Forest model achieved the best balanced accuracy and the strongest churn recall.

### 8.2 Best Model Result

Best model:

```text
Tuned Random Forest Classifier
```

Best balanced accuracy:

```text
0.956
```

Best Random Forest parameters:

```text
bootstrap: False
max_depth: None
min_samples_leaf: 2
min_samples_split: 5
n_estimators: 200
```

### 8.3 Confusion Matrix Results

#### Logistic Regression

```text
[[749 187]
 [ 36 154]]
```

Interpretation:

- Logistic Regression catches many churned customers with recall `0.81`.
- However, precision for churn is only `0.45`, meaning it creates many false churn alerts.

#### Random Forest

```text
[[927   9]
 [ 28 162]]
```

Interpretation:

- Random Forest improves both precision and recall.
- It reduces false positives significantly compared with Logistic Regression.

#### Tuned Random Forest

```text
[[918  18]
 [ 13 177]]
```

Interpretation:

- Tuned Random Forest correctly identifies `177` out of `190` churned customers in the test set.
- It misses only `13` churned customers.
- This is the best model for proactive churn prevention.

### 8.4 GitHub Result Image Placeholders

After running the upgraded notebook, add exported images to the `images/` folder and reference them here.

```markdown
![Model Comparison](images/model_comparison.png)
![Confusion Matrix](images/confusion_matrix.png)
![ROC and PR Curves](images/roc_pr_curves.png)
![Threshold Tuning](images/threshold_tuning.png)
![Feature Importance](images/feature_importance.png)
```

### 8.5 Feature Importance Result

The upgraded notebook includes a feature importance section for the tuned Random Forest model.

Use this section to explain:

- Which variables have the strongest influence on churn prediction
- Whether `Tenure`, `Complain`, `CashbackAmount`, `OrderCount`, or `DaySinceLastOrder` are key churn indicators
- How the business can use important features to build churn monitoring rules

Placeholder:

```markdown
![Feature Importance](images/feature_importance.png)
```

### 8.6 Churned Customer Segmentation Result

KMeans clustering was applied only to churned customers.

```text
Number of churned customers: 948
Number of clusters: 7
```

Segment distribution:

| Segment | Share of Churned Customers |
|---|---:|
| Segment 3 | `31.65%` |
| Segment 1 | `23.52%` |
| Segment 2 | `19.51%` |
| Segment 4 | `13.50%` |
| Segment 6 | `7.59%` |
| Segment 5 | `2.11%` |
| Segment 0 | `2.11%` |

### 8.7 Segment Profile and Suggested Actions

| Segment | Key Characteristics From Notebook | Suggested Action |
|---|---|---|
| Segment 0 | Small group, higher tenure, high order count, very high cashback | Premium win-back offer for valuable lost customers |
| Segment 1 | Larger group, moderate tenure, higher warehouse distance, moderate order count | Personalized retention offer and delivery support |
| Segment 2 | Low tenure, lower cashback, high complaint rate | Complaint recovery and early-life retention campaign |
| Segment 3 | Largest churned segment, low tenure, low cashback, low order count | Broad reactivation campaign with onboarding incentive |
| Segment 4 | City Tier 3, high warehouse distance, high complaint rate, E-wallet dominant | Delivery support and service recovery campaign |
| Segment 5 | Small group, higher order count, high cashback, longer inactivity | High-value reactivation with personalized category offer |
| Segment 6 | UPI dominant, high warehouse distance, high complaint rate | Payment-experience review and delivery compensation |

### 8.8 Clustering Result Image Placeholders

After running the upgraded notebook, add clustering visuals here.

```markdown
![Elbow Method](images/elbow_method.png)
![Churned Customer Segments PCA](images/churned_customer_segments_pca.png)
```


---

## 9. Business Recommendations

### 9.1 New Customer Retention

Customers with short tenure are more likely to churn.

Recommended actions:

- Offer welcome vouchers for first-time customers.
- Provide onboarding messages or product recommendations.
- Encourage second purchase through limited-time cashback.

### 9.2 Complaint Handling

Customers who complain have higher churn risk.

Recommended actions:

- Prioritize fast complaint resolution.
- Provide compensation vouchers after negative service experiences.
- Track complaint history as an important churn risk signal.

### 9.3 Personalized Promotion Strategy

Customers differ in promotion sensitivity and order behavior.

Recommended actions:

- Send personalized coupons to high-risk customers.
- Use cashback incentives for customers with low recent engagement.
- Avoid giving the same promotion to all customers.

### 9.4 Delivery Experience Improvement

Customers living farther from warehouses may experience delivery inconvenience.

Recommended actions:

- Provide delivery fee support for long-distance customers.
- Improve estimated delivery time communication.
- Prioritize logistics improvement for high-risk regions.

### 9.5 High-Value Customer Win-Back

Some churned customers have high order count, high cashback, or longer tenure.

Recommended actions:

- Create premium win-back campaigns for valuable churned customers.
- Offer personalized discounts based on past preferred categories.
- Assign higher retention priority to customers with strong historical value.

---

## 10. Technologies Used

- Python
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- Jupyter Notebook
- Google Colab

---

## 11. How To Run

1. Clone this repository or download the project files.
2. Place `churn_predict.csv` in the same directory as the notebook.
3. Open `ML_Project.ipynb` in Jupyter Notebook or Google Colab.
4. Run the cells from top to bottom.

---

## 12. Repository Structure

```text
.
├── ML_Project.ipynb
├── churn_predict.csv
├── images/
│   ├── model_comparison.png
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   └── churned_customer_segments.png
└── README.md
```

---

## 13. Conclusion

This project shows that machine learning can help an e-commerce company identify high-risk customers before they leave.

The dataset contains `5,630` customers, with `948` churned customers and a churn rate of `16.84%`. Because the target class is imbalanced, balanced accuracy and churn recall are more useful than normal accuracy alone.

The tuned Random Forest model achieved the best performance:

- Balanced Accuracy: `0.956`
- Churn Precision: `0.91`
- Churn Recall: `0.93`
- Churn F1-score: `0.92`

The model correctly identified most churned customers in the test set, missing only `13` churned customers. This makes it suitable for proactive retention targeting.

The clustering analysis shows that churned customers are not one homogeneous group. Different churned segments show different patterns, such as low tenure, high complaint rate, long warehouse distance, high cashback, or high historical order count.

Overall, the project demonstrates how supervised learning and customer segmentation can work together to support practical business actions:

- Predict customers likely to churn
- Understand churn drivers
- Prioritize high-risk customers
- Design targeted retention campaigns
- Build segment-specific win-back strategies

