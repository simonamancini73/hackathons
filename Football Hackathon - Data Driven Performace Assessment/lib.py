# Importo Librerie
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate, cross_val_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_regression
from sklearn.exceptions import UndefinedMetricWarning
from imblearn.pipeline import Pipeline
from imblearn import FunctionSampler
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
#import lux
import datetime
from warnings import filterwarnings
filterwarnings('ignore')
from lib import *


class LoggerHead:
    
    def __init__(self, n=5):
        self.n = n
    
    def fit(self, X, y):
        return self
    
    def transform(self, X):
        print(pd.DataFrame(X).head(self.n))
        return X


class LoggerShape:
    
    def fit(self, X, y):
        return self
    
    def transform(self, X):
        print(X.shape)
        return X


class Logger:
    
    def fit(self, X, y):
        print("Logger.fit: " + str(datetime.datetime.now()))
        return self
    
    def transform(self, X):
        print("Logger.transform: " + str(datetime.datetime.now()))
        return X


class AbsValue:
    
    def fit(self, X, y):
        return self
    
    def transform(self, X):
        return X.abs()


class LogValue:
    
    def fit(self, X, y):
        return self
    
    def transform(self, X):
        np.seterr(divide = 'ignore')
        arr = np.where(X > 0, np.log(X), 0)
        np.seterr(divide = 'warn') 
        X = pd.DataFrame(arr, columns = X.columns)
        return X
    
    
'''
mmsc = MinMaxScalerColumn('Fare')
mmsc.fit(X_train, y_train)
mmsc.transform(X_train, y_train)  # X_train ha solo numeri tra 0 e 1
mmsc.transform(X_test, y_test)
'''
class MinMaxScalerColumn:
    
    def __init__(self, c):
        self.column = c
    
    def fit(self, X, y):
        self.min = X[self.column].min()
        self.max = X[self.column].max()
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        X_columns = X_copy.columns
        
        X_column = X_copy[self.column]
        
        X_column = np.where(X_column > self.max, self.max, X_column)
        X_column = np.where(X_column < self.min, self.min, X_column)
        X_column = (X_column - self.min) / (self.max - self.min)
        
        X_copy[self.column] = X_column
        
        X_ret = pd.DataFrame(X_copy, columns = X_columns)
        return X_ret


class OneHotEncoderColumn:
    
    def __init__(self, c):
        self.column = c
        
    def getColumn(self):
        return self.column
    
    def fit(self, X, y):
        X_column = X[[self.column]]
        self.ohe = OneHotEncoder(categories = "auto", handle_unknown = "ignore")
        self.ohe.fit(X_column.values.reshape(-1, 1))
        return self
    
    def transform(self, X):
        X_column = X[[self.column]]
        
        X_ohe_pd = pd.DataFrame(
            self.ohe.transform(X_column.values.reshape(-1, 1)).toarray(),
            columns = [self.column + '_' + x for x in self.ohe.categories_[0]]
        )
        X_ohe_pd.index = X.index
        
        X_out = pd.concat([X.drop([self.column], axis = 1), X_ohe_pd], axis = 1)
        
        return X_out
    

class Judge:
    
    def __init__(self):
        self.params = {}
        return None
    
    def set_algorithms(self, algorithms):
        self.algorithms = algorithms
        return self
    
    def set_data(self, X, y):
        self.X = X
        self.y = y
        return self
    
    def set_metrics(self, metrics):
        self.metrics = metrics
        return self
    
    def set_params(self, params):
        self.params = params
        return self
    
    def get_performance(self, metric, algorithm, grid):
        best_algorithm = algorithm
        if bool(grid):
            best_algorithm = GridSearchCV(estimator = algorithm, param_grid = grid, n_jobs = -1)  # inner loop
        scores = cross_validate(best_algorithm, X = self.X, y = self.y, scoring = metric, cv = 2)  # outer_loop
        score = np.mean(scores['test_score'])
        score = round(score, 2)
        
        return score
        
    def get_table(self):
        metrics_results = {}
        for metric in self.metrics:
            algorithms_results = {}
            for label, algorithm in self.algorithms.items():
                grid = {}
                if label in self.params.keys():
                    grid = self.params[label]
                algorithms_results[label] = self.get_performance(metric, algorithm, grid)
            metrics_results[metric] = algorithms_results
            
        df = pd.DataFrame.from_dict(metrics_results)
                
        return df
    

# One-hot encoding
def ohe(data, column):
    d = pd.get_dummies(data[column], prefix = column)
    data_d = pd.concat([data, d], axis = 1)
    data_d = data_d.drop([column], axis = 1)
    return data_d


def null_count(data):
    return data.isnull().sum()

def missing_zero_values_table(df):
        zero_val = (df == 0.00).astype(int).sum(axis=0)
        mis_val = df.isnull().sum()
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        mz_table = pd.concat([zero_val, mis_val, mis_val_percent], axis=1)
        mz_table = mz_table.rename(
        columns = {0 : 'Zero Values', 1 : 'Missing Values', 2 : '% of Total Values'})
        mz_table['Total Zero Missing Values'] = mz_table['Zero Values'] + mz_table['Missing Values']
        mz_table['% Total Zero Missing Values'] = 100 * mz_table['Total Zero Missing Values'] / len(df)
        mz_table['Data Type'] = df.dtypes
        mz_table = mz_table[
            mz_table.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns and " + str(df.shape[0]) + " Rows.\n"      
            "There are " + str(mz_table.shape[0]) +
              " columns that have missing values.")
        return mz_table


def min_max_scaler_single_column(column):
    col_min = np.min(column)
    col_max = np.max(column)
    column = (column - col_min) / (col_max - col_min)
    return column

def min_max_scaler_dt(dataframe):
    for column in dataframe.columns:
        column_min = np.min(dataframe[column])
        column_max = np.max(dataframe[column])
        dataframe[column] = (dataframe[column] - column_min) / (column_max - column_min)
    return dataframe

def remove_n_line(X, y, n):
    X_prep = np.delete(X, list(range(5)), 0)
    y_prep = y[n:]
    return X_prep, y_prep