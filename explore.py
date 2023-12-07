## IMPORTS ##
##-------------------------------------------------------------------##
#tabular data imports :
import pandas as pd
import numpy as np
import env
from env import username, password, host
from pydataset import data

# visualization imports:
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.feature_selection import SelectKBest, RFE, f_regression, SequentialFeatureSelector
# success metrics from earlier in the week: mean squared error and r^2 explained variance
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler
import scipy.stats as stats
from scipy.stats import pearsonr, spearmanr
from scipy.stats import shapiro

import warnings
warnings.filterwarnings("ignore")
import wrangle as w
import os
directory = os.getcwd()





def split_data(df):
    '''
    take in a DataFrame and return train, validate, and test DataFrames.
    return train, validate, test DataFrames.
    '''
    
    # Create train_validate and test datasets
    train, validate_test = train_test_split(df, train_size=0.60, random_state=123)
    
    # Create train and validate datsets
    validate, test = train_test_split(validate_test, test_size=0.5, random_state=123)

    # Take a look at your split datasets

    print(f"""
    train -> {train.shape}
    validate -> {validate.shape}
    test -> {test.shape}""")
    
    return train, validate, test

def preprocess_concrete(df):
    '''
    preprocess_concrete will take in values in form of a single pandas dataframe
    and make the data ready for spatial modeling,
    including:
     - splitting the data
     - scaling continuous columns (excluding the target column 'strength')

    return: three pandas dataframes, ready for modeling structures.
    '''
    # split data:
    train, validate, test = split_data(df)

    # Separate target column 'strength' from features
    train_features = train.drop(columns='strength')
    validate_features = validate.drop(columns='strength')
    test_features = test.drop(columns='strength')
    train_target = train[['strength']]
    validate_target = validate[['strength']]
    test_target = test[['strength']]

    # scale continuous features (excluding 'strength'):
    scaler = MinMaxScaler()
    train_features_scaled = pd.DataFrame(
        scaler.fit_transform(train_features),
        index=train_features.index,
        columns=train_features.columns)
    validate_features_scaled = pd.DataFrame(
        scaler.transform(validate_features),
        index=validate_features.index,
        columns=validate_features.columns)
    test_features_scaled = pd.DataFrame(
        scaler.transform(test_features),
        index=test_features.index,
        columns=test_features.columns)

    # Recombine scaled features with the target column 'strength'
    train_scaled = pd.concat([train_features_scaled, train_target], axis=1)
    validate_scaled = pd.concat([validate_features_scaled, validate_target], axis=1)
    test_scaled = pd.concat([test_features_scaled, test_target], axis=1)

    return train_scaled, validate_scaled, test_scaled


def scale_data(train, 
               validate, 
               test, 
               to_scale):
    #make copies for scaling
    train_scaled = train.copy()
    validate_scaled = test.copy()
    test_scaled = test.copy()

    #make the thing
    scaler = MinMaxScaler()

    #fit the thing
    scaler.fit(train[to_scale])

    #use the thing
    train_scaled[to_scale] = scaler.transform(train[to_scale])
    validate_scaled[to_scale] = scaler.transform(validate[to_scale])
    test_scaled[to_scale] = scaler.transform(test[to_scale])
    
    return train_scaled, validate_scaled, test_scaled

def check_normality(data, column_name):
    # Graphical Method - Histogram
    plt.hist(data[column_name], bins=20)
    plt.title(f'Histogram of {column_name}')
    plt.show()

    # Graphical Method - Q-Q Plot
    stats.probplot(data[column_name], dist="norm", plot=plt)
    plt.title(f'Q-Q Plot of {column_name}')
    plt.show()

    # Statistical Test - Shapiro-Wilk Test
    shapiro_test = stats.shapiro(data[column_name])
    print(f'Shapiro-Wilk Test for {column_name}: Statistic={shapiro_test.statistic:.3f}, p-value={shapiro_test.pvalue:.3f}')

    # Statistical Test - D’Agostino’s K^2 Test
    k2, p = stats.normaltest(data[column_name])
    print(f'D’Agostino’s K^2 Test for {column_name}: Statistic={k2:.3f}, p-value={p:.3f}')

    # Statistical Test - Anderson-Darling Test
    anderson_test = stats.anderson(data[column_name], dist='norm')
    print(f'Anderson-Darling Test for {column_name}: Statistic={anderson_test.statistic:.3f}')
    for i in range(len(anderson_test.critical_values)):
        sl, cv = anderson_test.significance_level[i], anderson_test.critical_values[i]
        if anderson_test.statistic < cv:
            print(f'{sl:.3f}: {cv:.3f}, data looks normal (fail to reject H0)')
        else:
            print(f'{sl:.3f}: {cv:.3f}, data does not look normal (reject H0)')
 
    
def eval_Spearman(r, p, α=0.05):
    if p < α:
        return print(f"""We reject H₀, there is a monotonic relationship.
Spearman’s r: {r:2f}
P-value: {p}""")
    else:
        return print(f"""We fail to reject H₀: that there is a monotonic relationship.
Spearman’s r: {r:2f}
P-value: {p}""")


def evaluate_reg(y, yhat):
    '''
    based on two series, y_act, y_pred, (y, yhat), we
    evaluate and return the root mean squared error
    as well as the explained variance for the data.
    
    returns: rmse (float), rmse (float)
    '''
    rmse = mean_squared_error(y, yhat, squared=False)
    r2 = r2_score(y, yhat)
    return rmse, r2

def plot_all_histograms(df, individual_fig_size=(8, 4)):
    """
    Plot histograms for each numerical column in the dataframe.
    
    Parameters:
    df (DataFrame): The dataframe containing the data.
    individual_fig_size (tuple): Figure size for individual histograms.
    """
    numeric_cols = df.select_dtypes(include=['number']).columns  # This will select only numeric columns

    for col in numeric_cols:
        plt.figure(figsize=individual_fig_size)
        plt.title(col.upper())  # Convert title to uppercase
        df[col].hist(bins=50)
        plt.xticks(rotation=30)  # Rotate x-axis labels by 30 degrees
        plt.grid(False)  # Hide the grid
        plt.show()  # Show each plot individually
