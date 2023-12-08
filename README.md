# Quantitative Analysis and Predictive Modeling of Concrete Strength Based on Composition

## Project Overview

This project focuses on predicting the compressive strength of concrete, which is a crucial measure in civil engineering. I utilize various machine learning models to understand the relationship between the concrete mixture constituents and its strength.

## Goals

- To explore the dataset and understand the factors affecting concrete strength.
- To develop a predictive model with the highest possible accuracy.
- To provide insights and recommendations for concrete mix optimization.

## Initial Hypotheses

- There a realtionship between the amount of cement in a cubic yard and concrete strength of a sample.
- There a realtionship between the amount of coarse aggregate in a cubic yard and concrete strength of a sample.
- There a realtionship between the amount of fine aggregate in a cubic yard and concrete strength of a sample.
- There a realtionship between the amount of total weight in a cubic yard and concrete strength of a sample.

## Project Plan

1. Acquire
- get the data into pandas
- look at it
- describe, info, head, shape
- understand what your data means
- know what each column is
- know what your target variable is
2. Wrangle
- clean the data
- handle nulls
- handle outliers
- correct datatypes
- univariate analysis (looking at only one variable)
- encode variables -- Preprocessing
- split into train, validate/, test
- scale data (after train/validate/test split) -- Preprocessing
- document how you're changing the data
3. Explore
- use train data
- use unscaled data
- establish relationships using multivariate analysis
- hypothesize
- visualize
- statistize
- summarize
- feature engineering
- use scaled data
4. Model
- use scaled/encoded data
- split into X_variables and y_variables
- X_train, y_train, X_validate, y_validate, X_test, y_test
- build models
- make
- fit (on train)
- use
- evaluate models on train and validate
- pick the best model and evaluate it on test
5. Test
- present results of the best model

## Data Dictionary

| Feature       | Definition                                 |
|---------------|--------------------------------------------|
| cement        | Quantity of cement in the sample mix (yd³)  |
| slag          | Quantity of slag in the samplemix (yd³)    |
| ash           | Quantity of fly ash in the samplemix (yd³) |
| water         | Quantity of water in the sample mix (yd³)   |
| superplastic  | Quantity of superplasticizer in the sample mix (yd³)   |
| coarseagg     | Quantity of coarse aggregates in the sample mix (yd³)  |
| fineagg       | Quantity of fine aggregates in the sample mix (yd³)    |
| age           | Age of concrete at testing (days)          |
| strength      | Compressive strength of concrete (psi)     |
| total_lbs_per_yd³ | Total weight of all components per cubic yard (yd³) |
| sample        | Identifier for the concrete sample         |


## Steps to Reproduce
1. Download concrete_strength.ipynb, wrgangle.py, and explore.py files in this repository.
2. Obtain the required .csv datased from 'https://www.kaggle.com/datasets/vinayakshanawad/cement-manufacturing-concrete-dataset'
3. Run it

## Takeaways
- The Polynomial Regression model significantly outperforms the baseline, explaining up to 47.92% of the variance in concrete strength.¶
- With 'total_lbs_per_yd^3' positively correlated and 'coarseagg' and 'fineagg' negatively correlated with strength, all significant at an alpha below 0.05.
### Recommendations
- Stakeholders should prioritize optimizing the total mix composition, particularly focusing on the 'total_lbs_per_yd^3' due to its strong positive impact on strength.
- A deeper analysis of 'coarseagg' and 'fineagg' proportions is advised to enhance the concrete's strength further.
### Next Steps
- Explore additional features or transformations that might better capture the relationships in the data, such as interaction terms between different types of aggregates.

##### Dataset Source: Kaggle - Cement Manufacturing Concrete Dataset