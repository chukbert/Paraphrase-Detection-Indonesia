import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))

from evaluation_matrics import print_validasi, print_score_uji, print_score_opt_uji

import optuna
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def objective_svm(trial, X, y):
    C = trial.suggest_float('svc__C', 1e-3, 1000, log=True)
    kernel = trial.suggest_categorical('svc__kernel', ['linear', 'rbf', 'poly'])
    gamma = trial.suggest_float('svc__gamma', 1e-4, 1e1, log=True)
    degree = trial.suggest_int('svc__degree', 2, 5) if kernel == 'poly' else trial.suggest_int('svc__degree', 3, 3)

    svm = SVC(C=C, kernel=kernel, gamma=gamma, degree=degree, random_state=13)
    pipeline = make_pipeline(SMOTE(random_state=42), svm)
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    try:
        return cross_val_score(pipeline, X, y, cv=kf, scoring='f1_macro', n_jobs=-1).mean()
    except ValueError as e:
        if 'dual coefficients or intercepts are not finite' in str(e):
            return np.nan
        raise

def find_best_param_svm(X_train, y_train, n_trials=100, timeout=300, list_param=None):
    study = optuna.create_study(direction='maximize')
    study.enqueue_trial({'svc__C': 1, 'svc__kernel': 'linear', 'svc__gamma': 1, 'svc__degree': 3})
    if list_param:
        for param in list_param:
            study.enqueue_trial(param)    
    
    study.optimize(lambda trial: objective_svm(trial, X_train, y_train), n_trials=n_trials, timeout=timeout)

    best_params = study.best_params
    best_svm = SVC(C=best_params['svc__C'], kernel=best_params['svc__kernel'], 
                    gamma=best_params['svc__gamma'], degree=best_params['svc__degree'], 
                    random_state=13)
    best_pipeline = make_pipeline(SMOTE(random_state=42), best_svm)
    best_pipeline.fit(X_train, y_train)
    print_validasi(best_pipeline, X_train, y_train)
    print(best_params)

    return study, best_pipeline

default_param = {'svc__C': [0.21910254866942946],
                'svc__kernel': ['rbf'],
                'svc__gamma': [0.001128369711239445],
                'svc__degree': [3]}

def fit_and_test_svm(X_train, y_train, X_test, y_test, param=default_param, train_score=False):
    svm = SVC(random_state=13)
    param_grid = param
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    pipeline = make_pipeline(SMOTE(random_state=42), svm)
    grid_search = GridSearchCV(pipeline, param_grid, cv=kf, scoring='f1_macro', return_train_score=train_score, verbose=2)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    # y_pred = post_process(X_train, y_pred)
    print_score_uji(y_test, y_pred, grid_search)
    return grid_search, best_model

def fit_train_svm(X_train, y_train, param=default_param, train_score=False):
    svm = SVC(random_state=13)
    param_grid = param
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    pipeline = make_pipeline(SMOTE(random_state=42), svm)
    grid_search = GridSearchCV(pipeline, param_grid, cv=kf, scoring='f1_macro', return_train_score=train_score, verbose=2)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    print_validasi(best_model, X_train, y_train)
    
    return grid_search, best_model

def fit_train_loop_svm(X_train, y_train, param=default_param, train_score=False):
    svm = SVC(random_state=13)
    param_grid = param
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    pipeline = make_pipeline(SMOTE(random_state=42), svm)
    grid_search = GridSearchCV(pipeline, param_grid, cv=kf, scoring='f1_macro', return_train_score=train_score, verbose=2)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='macro')
    return grid_search, best_model, f1