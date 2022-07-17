from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate, cross_val_score
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import pandas as pd
import numpy as np
from lib import *
from sklearn.pipeline import Pipeline
import datetime

df = pd.read_csv('data/titanic.csv')
df = df.dropna()

X = df[['Pclass', 'SibSp', 'Parch', 'Fare', 'Embarked']]
y = df['Survived']

pipe_dt_01 = Pipeline([
    ('ohec', OneHotEncoderColumn('Embarked')),
    ('remove_n_lines', FunctionSampler(func=remove_n_line, kw_args={'n': 5})),
    ('abs', AbsValue()),
    ('log', LogValue()),
    ('minmax_fare', MinMaxScalerColumn('Fare')),
    ('fs', SelectKBest(chi2)),
    ('dt', DecisionTreeClassifier())
], verbose = False)

pipelines = {
    'pipe_dt': pipe_dt_01
}

# https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
metrics = [
    'accuracy',
    'roc_auc',
    'precision',
    'recall'
]

params = {
    'pipe_dt': {
        'dt__min_samples_leaf': [1],
        'dt__max_depth': [3, 5, 10],
        'fs__k': [1, 2]
    }
}

judge = Judge()
judge.set_data(X, y)
judge.set_algorithms(pipelines)
judge.set_metrics(metrics)
judge.set_params(params)
print(judge.get_table())

best_pipe = pipe
grid = params['pipe_dt']

clf = GridSearchCV(best_pipe, param_grid = grid)
clf.fit(X, y)
print(clf.best_estimator_)