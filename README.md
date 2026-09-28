# E-Commerce Return Analysis

## Project Overview

This project analyzes why customers return products in an e-commerce environment. The analysis focuses on return rates, product categories, delivery patterns, customer history, and other order-related factors.

A Random Forest classification model is also trained to predict whether an order will be returned.

## Problem Statement

High product-return rates can increase operational costs and affect customer satisfaction. This project analyzes historical e-commerce order data to identify patterns associated with product returns and provide practical recommendations for reducing unnecessary returns.

## Objectives

- Calculate the overall product return rate
- Analyze return rates across product categories
- Study the relationship between delivery delay and returns
- Analyze customer purchase and return history
- Identify product and order-level patterns
- Train a Random Forest model to predict returns
- Evaluate the model using classification metrics
- Provide practical actions based on the analysis

## Dataset

The dataset contains 200,000 e-commerce order records and 16 columns.

### Features

- customer_age
- product_price
- discount_percent
- product_rating
- past_purchase_count
- past_return_rate
- delivery_delay_days
- session_length_minutes
- num_product_views
- device_type
- product_category
- shipping_method
- payment_method
- used_coupon
- returned
- order_id

The target variable is `returned`.

## Data Preprocessing

The preprocessing process included:

- Loading the raw dataset
- Checking dataset shape
- Checking column names
- Checking data types
- Checking missing values
- Checking duplicate records
- Removing duplicate records
- Saving the cleaned dataset

The dataset contained:

- 200,000 records
- 16 columns
- 0 missing values
- 0 duplicate records

The cleaned dataset is saved in:

`data/processed/cleaned_returns.csv`

## Exploratory Data Analysis

### Overall Return Rate

The overall return rate was:

**47.46%**

There were:

- 94,919 returned orders
- 105,081 non-returned orders

### Return Rate by Product Category

| Product Category | Return Rate |
|---|---:|
| Clothing | 53.05% |
| Toys | 50.31% |
| Beauty | 49.72% |
| Sports | 45.04% |
| Home | 44.54% |
| Electronics | 41.73% |

Clothing had the highest observed return rate, while electronics had the lowest.

### Return Rate by Delivery Group

Delivery delay was grouped into four categories:

- Early / On Time
- 0–2 Days
- 2–5 Days
- 5+ Days

Observed return rates:

| Delivery Group | Return Rate |
|---|---:|
| Early / On Time | 49.01% |
| 0–2 Days | 47.30% |
| 2–5 Days | 45.50% |
| 5+ Days | 43.65% |

The observed data does not show a simple increase in returns as delivery delay increases.

### Coupon Usage

| Coupon Usage | Return Rate |
|---|---:|
| No Coupon | 43.99% |
| Coupon Used | 49.63% |

Orders using coupons had a higher observed return rate in this dataset.

### Device Type

| Device | Return Rate |
|---|---:|
| Desktop | 51.91% |
| Tablet | 48.38% |
| Mobile | 45.12% |

Desktop orders had the highest observed return rate among the device categories.

### Shipping Method

| Shipping Method | Return Rate |
|---|---:|
| Express | 53.22% |
| Standard | 45.51% |
| Same Day | 44.29% |

Express shipping had the highest observed return rate.

## Visualizations

The project contains four required visualizations:

1. Overall Return Distribution
2. Return Rate by Product Category
3. Return Rate by Delivery Group
4. Customer History by Return Status

The visualizations are stored in:

`visualizations/`

## Random Forest Model

A Random Forest Classifier was trained to predict the `returned` variable.

Categorical features were encoded using One-Hot Encoding.

The dataset was divided into:

- 80% training data
- 20% testing data

The model used:

- 100 decision trees
- Random state: 42

## Model Evaluation

| Metric | Result |
|---|---:|
| Accuracy | 55.58% |
| Precision | 53.92% |
| Recall | 44.10% |
| F1 Score | 48.52% |

The model shows limited predictive performance on the test dataset.

## Feature Importance

The most important features identified by the Random Forest were:

| Feature | Importance |
|---|---:|
| Delivery Delay Days | 0.1110 |
| Product Price | 0.1107 |
| Past Return Rate | 0.1106 |
| Discount Percentage | 0.1104 |
| Session Length | 0.1103 |
| Product Rating | 0.1100 |
| Customer Age | 0.0955 |
| Number of Product Views | 0.0891 |
| Past Purchase Count | 0.0522 |

Feature importance indicates which variables contributed to model predictions. It does not establish that these variables directly cause product returns.

## Key Insights

1. The overall return rate is 47.46%, indicating that returns represent a substantial portion of the observed orders.

2. Clothing has the highest observed category return rate at 53.05%, followed by toys at 50.31%.

3. Electronics has the lowest observed category return rate at 41.73%.

4. Orders using coupons have a higher observed return rate than orders without coupons.

5. Express shipping has the highest observed return rate among the shipping methods in the dataset.

6. Customer past return rate and delivery delay are among the most important features in the Random Forest model.

7. The Random Forest model achieved 55.58% accuracy and 48.52% F1 score, indicating that the available variables provide limited predictive power for return prediction.

## Practical Action Plan

### 1. Analyze High-Return Categories

Investigate products within high-return categories such as clothing and toys to identify product-specific patterns.

### 2. Improve Product Information

Provide clearer product descriptions, specifications, images, and other relevant product information to help customers make informed purchase decisions.

### 3. Review Discount Strategies

Coupon users showed a higher observed return rate. Businesses can analyze whether certain discount campaigns are associated with higher return behavior.

### 4. Investigate Express Shipping Orders

Express shipping showed a higher observed return rate. Further analysis should determine whether product mix, customer type, or other factors explain this pattern.

### 5. Use Customer Return History

Past return rate was one of the important model features. Historical return behavior can be used as one input for customer-level return-risk analysis.

### 6. Improve Predictive Modeling

The current Random Forest model has limited predictive performance. Additional relevant features and higher-quality return-specific information could improve future models.

## Dataset Limitations

The assignment lists fields such as `Size`, `Customer_History`, `Delivery_Time`, and `Return_Reason`. The selected dataset does not contain all of these fields directly.

Therefore, the project does not invent missing values or return reasons. Instead, available fields such as `past_purchase_count`, `past_return_rate`, and `delivery_delay_days` are used for the corresponding analyses where appropriate.

The dataset also does not provide a direct `Return_Reason` field, so return-reason analysis could not be performed from the available data.

## Project Structure

```text
E-Commerce-Return-Analysis/
├── data/
│   ├── raw/
│   │   └── train.csv
│   └── processed/
│       └── cleaned_returns.csv
├── models/
│   └── random_forest_model.joblib
├── src/
│   ├── preprocessing.py
│   ├── eda.py
│   ├── visualizations.py
│   ├── model.py
│   └── feature_importance.py
├── visualizations/
│   ├── overall_return_distribution.png
│   ├── category_return_rate.png
│   ├── delivery_return_rate.png
│   ├── customer_history_return.png
│   └── feature_importance.png
├── .gitignore
├── requirements.txt
└── README.md