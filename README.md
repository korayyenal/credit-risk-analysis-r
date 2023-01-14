# Credit Risk Analysis

## Problem Description
In order to minimize loss and maximize profits from lending, banks require careful assessment of their borrowers. This has led to credit risk computation becoming a commonly used application of statistics and data science.

In this notebook, I take a data set of loan applications and build a predictive model for making a decision as to whether to approve a loan and determine the loan amount based on the applicant’s characteristics in order to maximize the banks’ profits obtained from these loans. The dataset contains the records and results of lending money to some customers. The dataset contains 1,000,000 clients; 50,000 (5%) of them defaulted. I have an imbalanced dataset.

At first I do data exploration to get some insights, fill missing values and visualize some features, apply undersampling with SMOTE technique then I develop a two-stage framework for the prediction. First, I build a binary classification model to estimate the probability of default for each individual. Then, I build a regression model to predict the amount to loan each individual who are predicted to be non-defaulters.

## Modelling Approach

I adopt a two-stage approach, classification followed by regression. 

First, I obtain the probability of default for each client based on their characteristics. The decision rule for giving out a loan is the following: if the probability is greater than cutoff value, the loan amount will be greater than zero. Otherwise, the loan amount will be zero.

Second, for every client who is predicted as a non-defaulter, I predict a loan amount using the above regression models. If the predicted loan amount is lower than the requested loan amount, then I give the predicted loan amount. If the predicted loan amount is higher than the requested loan amount, then I give the requested loan amount. 

The advantage of this method is that it is intuitive and easy to interpret.

## Models

### Classification
1. Logistic Regression (Baseline)
2. XGBoost
3. Adaboost

### Regression
1. XGBoost
2. Blackboost

In the classification task, logistic regression is the baseline because it is a simpler algorithm and does not require much time to build compared to other models. In logistic regression, I select variables using Lasso (L-1) regularization. I use XGBoost and Blackboost methods for the regression task.

Adaboost is usually a good classification method in cases with imbalanced data.

XGBoost is one of the most popular ML algorithms and is known to yield highly accurate results.

Blackboost is a gradient boosting method where regression trees are utilized as base-learners.

## Performance Measure

Since I have an imbalanced dataset, using “accuracy” as a performance measure in a classification task would be misleading. As an alternative, I use F-1 score, which is a more suitable metric for imbalanced data.

