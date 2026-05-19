# E-Commerce Churn Prediction

This project analyses customer churn behaviour for an e-commerce company and builds machine learning models to predict whether a customer is likely to churn. The project also segments churned customers into different behavioural groups to support targeted promotion and retention strategies.

## Project Objectives

- Perform exploratory data analysis to understand customer behaviour.
- Identify important factors related to customer churn.
- Build supervised machine learning models to predict churn.
- Tune the best model using hyperparameter tuning.
- Segment churned customers using unsupervised learning.
- Provide business recommendations to reduce churn.

## Dataset

The dataset used in this project is `churn_predict.csv`.

Main columns include:

- `CustomerID`: Unique customer ID
- `Churn`: Target variable indicating whether a customer churned
- `Tenure`: Customer tenure with the company
- `PreferredLoginDevice`: Preferred login device
- `CityTier`: Customer city tier
- `WarehouseToHome`: Distance from warehouse to customer home
- `PreferredPaymentMode`: Preferred payment method
- `Gender`: Customer gender
- `HourSpendOnApp`: Time spent on app or website
- `NumberOfDeviceRegistered`: Number of registered devices
- `PreferredOrderCat`: Preferred order category
- `SatisfactionScore`: Customer satisfaction score
- `MaritalStatus`: Customer marital status
- `NumberOfAddress`: Number of registered addresses
- `Complain`: Whether the customer raised a complaint
- `CouponUsed`: Number of coupons used
- `OrderCount`: Number of orders
- `DaySinceLastOrder`: Days since last order
- `CashbackAmount`: Cashback received by the customer

## Project Workflow

### 1. Data Loading and Initial Inspection

The dataset is loaded and inspected using basic pandas functions such as `head()`, `shape`, `info()`, and missing value checks.

### 2. Feature Classification

Features are separated into:

- ID columns
- Numerical variables
- Categorical variables
- Target variable: `Churn`

### 3. Data Distribution Analysis

The project analyses the distribution of numerical and categorical variables using descriptive statistics, skewness, histograms, count plots, and boxplots.

### 4. Missing Values and Data Cleaning

Missing values are handled using median values for numerical columns and mode values for categorical columns. Inconsistent category labels are also standardised.

Examples:

- `COD` is standardised to `Cash on Delivery`
- `CC` is standardised to `Credit Card`
- `Mobile` is standardised to `Mobile Phone`

### 5. Relationship Between Features and Churn

The relationship between customer features and churn is analysed using:

- Grouped descriptive statistics
- Boxplots
- Correlation heatmap
- Churn rate by category

Important churn-related factors include tenure, complaints, cashback amount, delivery distance, and order behaviour.

### 6. Churned User Behaviour Analysis

The analysis shows that churned users often have shorter tenure, more complaints, lower cashback, and lower engagement. Some churned customers also appear to be affected by delivery distance or poor customer experience.

### 7. Business Recommendations

Recommended retention actions include:

- Welcome vouchers and cashback for new customers
- Faster complaint handling and compensation vouchers
- Personalised promotions for high-risk customers
- Delivery support for customers far from warehouses
- Win-back campaigns for valuable churned customers

## Supervised Learning

The project uses supervised learning to predict customer churn.

### Models Used

- Logistic Regression
- Random Forest Classifier
- Tuned Random Forest Classifier using GridSearchCV

### Preprocessing

- Categorical variables are encoded using one-hot encoding.
- `CustomerID` is removed because it is only an identifier.
- `Churn` is used as the target variable.
- Standard scaling is applied for Logistic Regression.
- The dataset is split into training and testing sets using stratified sampling.

### Model Performance

Balanced accuracy is used because the churn classes are imbalanced.

| Model | Balanced Accuracy |
|---|---:|
| Logistic Regression | 0.805 |
| Random Forest | 0.922 |
| Tuned Random Forest | 0.956 |

The tuned Random Forest model achieved the best performance.

Best Random Forest parameters:

```text
bootstrap: False
max_depth: None
min_samples_leaf: 2
min_samples_split: 5
n_estimators: 200
```

## Unsupervised Learning

K-Means clustering is applied only to customers with `Churn == 1` to segment churned users into different groups.

### Clustering Workflow

- Filter churned customers.
- Remove ID and target columns.
- One-hot encode categorical variables.
- Scale features using `StandardScaler`.
- Use the Elbow Method to select the number of clusters.
- Train KMeans with `k = 7`.
- Compare numerical and categorical characteristics across segments.

### Segment Insights

The churned users show different behavioural patterns:

- Some segments contain new users with low tenure and low engagement.
- Some segments have high complaint rates, suggesting poor customer experience.
- Some groups have longer warehouse-to-home distances, indicating possible delivery inconvenience.
- Some churned users are high-value customers with high order count, high cashback, and longer tenure.

These differences show that churned customers should not be treated as one single group. Each segment should receive a different retention or win-back strategy.

## Technologies Used

- Python
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- Jupyter Notebook / Google Colab

## How to Run

1. Clone this repository or download the project files.
2. Place `churn_predict.csv` in the same directory as the notebook.
3. Open `ML_Project.ipynb` in Jupyter Notebook or Google Colab.
4. Run the cells from top to bottom.

## Repository Structure

```text
.
├── ML_Project.ipynb
└── README.md
```

## Conclusion

The project shows that churn prediction can help the company identify high-risk customers before they leave. Random Forest performs best among the tested models, and customer segmentation provides useful insights for designing targeted promotions and retention strategies.
