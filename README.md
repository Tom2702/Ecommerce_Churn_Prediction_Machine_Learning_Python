# E-Commerce Churn Prediction

## Table of Contents

1. [Background & Overview](#1-background--overview)
2. [Project Objectives](#2-project-objectives)
3. [Dataset Description](#3-dataset-description)
4. [Project Workflow](#4-project-workflow)
5. [Exploratory Data Analysis](#5-exploratory-data-analysis)
6. [Supervised Learning](#6-supervised-learning)
7. [Unsupervised Learning](#7-unsupervised-learning)
8. [Business Recommendations](#8-business-recommendations)
9. [Technologies Used](#9-technologies-used)
10. [How To Run](#10-how-to-run)
11. [Conclusion](#11-conclusion)

---

## 1. Background & Overview

This project analyzes customer churn behavior for an e-commerce company and builds machine learning models to predict whether a customer is likely to churn.

In addition to churn prediction, the project segments churned customers into different behavioral groups using clustering. This helps the business design more targeted promotion, retention, and win-back strategies instead of treating all churned customers the same way.

The project combines:

- Exploratory data analysis
- Data cleaning and preprocessing
- Supervised machine learning for churn prediction
- Hyperparameter tuning
- Unsupervised learning for churned customer segmentation
- Business recommendations for retention strategy

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

### 3.1 Main Columns

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

### 3.2 Target Variable

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

Inconsistent category labels are also standardized.

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

### 5.1 Churned Customer Behavior

The analysis shows that churned users often have:

- Shorter tenure
- More complaints
- Lower cashback amount
- Lower engagement
- Different order behavior compared to retained customers
- Possible delivery inconvenience due to longer warehouse-to-home distance

### 5.2 Business Interpretation

These patterns suggest that churn is not caused by a single factor. Instead, churn may come from a combination of:

- Poor customer experience
- Low loyalty or short customer lifetime
- Delivery inconvenience
- Weak promotion or cashback engagement
- Lower app or website engagement

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

## 8. Business Recommendations

### 8.1 New Customer Retention

Customers with short tenure are more likely to churn.

Recommended actions:

- Offer welcome vouchers for first-time customers.
- Provide onboarding messages or product recommendations.
- Encourage second purchase through limited-time cashback.

### 8.2 Complaint Handling

Customers who complain have higher churn risk.

Recommended actions:

- Prioritize fast complaint resolution.
- Provide compensation vouchers after negative service experiences.
- Track complaint history as an important churn risk signal.

### 8.3 Personalized Promotion Strategy

Customers differ in promotion sensitivity and order behavior.

Recommended actions:

- Send personalized coupons to high-risk customers.
- Use cashback incentives for customers with low recent engagement.
- Avoid giving the same promotion to all customers.

### 8.4 Delivery Experience Improvement

Customers living farther from warehouses may experience delivery inconvenience.

Recommended actions:

- Provide delivery fee support for long-distance customers.
- Improve estimated delivery time communication.
- Prioritize logistics improvement for high-risk regions.

### 8.5 High-Value Customer Win-Back

Some churned customers have high order count, high cashback, or longer tenure.

Recommended actions:

- Create premium win-back campaigns for valuable churned customers.
- Offer personalized discounts based on past preferred categories.
- Assign higher retention priority to customers with strong historical value.

---

## 9. Technologies Used

- Python
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- Jupyter Notebook
- Google Colab

---

## 10. How To Run

1. Clone this repository or download the project files.
2. Place `churn_predict.csv` in the same directory as the notebook.
3. Open `ML_Project.ipynb` in Jupyter Notebook or Google Colab.
4. Run the cells from top to bottom.

---

## 11. Conclusion

This project shows that churn prediction can help the company identify high-risk customers before they leave.

The tuned Random Forest model achieved the best performance with a balanced accuracy of `0.956`, outperforming Logistic Regression and the baseline Random Forest model.

Customer segmentation further shows that churned customers have different behavioral patterns. Therefore, retention strategies should be targeted by churn segment rather than applying one general promotion to all churned users.

Overall, the project demonstrates how machine learning can support customer retention by combining predictive modeling, behavioral segmentation, and business-focused recommendations.

