# Credit Risk Analysis

The dataset can be accessed by this link: https://drive.google.com/file/d/102q3Ki7qN0pAW2i-WJsOYP1dhO9VEIaW/view?usp=share_link

## Problem Description

The main objective is to predict which applicants to give loan to based on their characteristics, and determine the loan amount for each applicant in order to maximize the bank’s profits obtained from these loans in a credit risk game setting. 

The task is to classify client applications into default or non-default (binary classification problem), then predict the loan amount for those predicted as non-default (regression problem).

I apply methods to a dataset with a size of 1 million rows and 32 columns. The dataset contains client records and the results of lending money to some customers. Out of 1000000 applications, 50000 (5%) of them defaults. I have an imbalanced dataset.

## Methodology

I do data exploration to get some insights, fill missing values and visualize some features, check correlation among features, and apply undersampling with SMOTE technique to obtain a balanced dataset.

I develop a two-stage framework for the task. First, I build a binary classification model to estimate the probability of default for each individual. if the probability is greater than cutoff value, the loan amount will be greater than zero. Otherwise, the loan amount will be zero. 

Then, I build a regression model to predict the loan amount for each individual who are predicted to be non-defaulters. If the predicted loan amount is lower than the requested loan amount, then I give the predicted loan amount. If the predicted loan amount is higher than the requested loan amount, then I give the requested loan amount. The advantage of this method is that it is intuitive and easy to interpret.

## Models

### Classification
1. Logistic Regression (Baseline)
2. XGBoost (one of the most popular ML algorithms and is known to yield highly accurate results.)
3. Adaboost (usually a good classification method in cases with imbalanced data.)

I use F1 score as the performance measure which is known to provide robust results for imbalanced datasets.

### Regression
1. XGBoost
2. Blackboost (a gradient boosting method where regression trees are utilized as base-learners.)

In the classification task, logistic regression is the baseline because it is a simpler algorithm and does not require much time to build compared to other models. In logistic regression, I select variables using Lasso (L-1) regularization. I use XGBoost and Blackboost methods for the regression task.

## Results

### Variable Importance

![image](credit_risk_analysis_files/figure-gfm/unnamed-chunk-36-1.png)

Variable importance graph shows that some of the most important variables are:

- Total used amount of revolving credit (MNT_UTIL_REN),
- Value of financial assets (MNT_ACT),
- Value of financial liabilities (MNT_PASS), etc.

The results make sense; clients with high financial liabilities or credit utilization tend to have a higher risk of default, while the ratio of requested loan amount to value of the assets tend to give an accurate indication as to whether the client can payback their debt by their savings or assets.

### Metrics: F1 Score & RMSE

For the classification task, I obtain an F1 score of 96.3% found by Adaboost and 90% by the logistic regression. Thus, Adaboost achieves a 6% improvement on the baseline model.

For the regression task, I obtain an RMSE value of 10,100 found by the XGBoost and Blackboost. 
