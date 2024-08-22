import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))

from evaluation_matrics import print_validasi, print_score_uji, print_score_opt_uji

import optuna, xgboost
from xgboost import XGBClassifier, plot_tree
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import numpy as np
import random
import matplotlib.pyplot as plt
import graphviz
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


random.seed(42)
np.random.seed(42)

def objective_xgb(trial, X, y):
    eta = trial.suggest_float('xgbclassifier__eta', 0.01, 0.3)
    n_estimators = trial.suggest_int('xgbclassifier__n_estimators', 10, 1000)
    max_depth = trial.suggest_int('xgbclassifier__max_depth', 2, 10)
    learning_rate = trial.suggest_float('xgbclassifier__learning_rate', 0.01, 0.3)
    subsample = trial.suggest_float('xgbclassifier__subsample', 0.5, 1.0)
    colsample_bytree = trial.suggest_float('xgbclassifier__colsample_bytree', 0.5, 1.0)
    min_child_weight = trial.suggest_int('xgbclassifier__min_child_weight', 1, 10)
    gamma = trial.suggest_float('xgbclassifier__gamma', 0, 5.0)
    reg_lambda = trial.suggest_float('xgbclassifier__reg_lambda', 0, 10.0)
    objective = trial.suggest_categorical('xgbclassifier__objective', ['binary:logistic', 'reg:squarederror', 'reg:logistic', 'binary:hinge'])

    xgb = XGBClassifier(eta=eta,
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        learning_rate=learning_rate,
                        subsample=subsample,
                        colsample_bytree=colsample_bytree,
                        min_child_weight=min_child_weight,
                        gamma=gamma,
                        reg_lambda=reg_lambda,
                        objective=objective,
                        random_state=42,
                        use_label_encoder=False)
    pipeline = make_pipeline(SMOTE(random_state=42), xgb)
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    return cross_val_score(pipeline, X, y, cv=kf, scoring='f1_macro', n_jobs=-1).mean()

def find_best_param_xgb(X_train, y_train, n_trials=100, timeout=300, list_param=None):
    study = optuna.create_study(direction='maximize')
    study.enqueue_trial({'xgbclassifier__eta': 0.3, 'xgbclassifier__n_estimators': 100, 'xgbclassifier__max_depth': 6, 'xgbclassifier__learning_rate': 0.3, 'xgbclassifier__subsample': 1, 'xgbclassifier__colsample_bytree': 1, 'xgbclassifier__min_child_weight': 1, 'xgbclassifier__gamma': 0, 'xgbclassifier__reg_lambda': 1, 'xgbclassifier__objective': 'reg:squarederror'})

    if list_param:
        for param in list_param:
            study.enqueue_trial(param)
    
    study.optimize(lambda trial: objective_xgb(trial, X_train, y_train), n_trials=n_trials, timeout=timeout)

    best_params = study.best_params
    best_xgb = XGBClassifier(eta=best_params['xgbclassifier__eta'], n_estimators=best_params['xgbclassifier__n_estimators'],
                             max_depth=best_params['xgbclassifier__max_depth'], learning_rate=best_params['xgbclassifier__learning_rate'],
                             subsample=best_params['xgbclassifier__subsample'], colsample_bytree=best_params['xgbclassifier__colsample_bytree'],
                             min_child_weight=best_params['xgbclassifier__min_child_weight'], gamma=best_params['xgbclassifier__gamma'],
                             reg_lambda=best_params['xgbclassifier__reg_lambda'], objective=best_params['xgbclassifier__objective'],
                             random_state=42,
                             use_label_encoder=False)
    best_pipeline = make_pipeline(SMOTE(random_state=42), best_xgb)
    best_pipeline.fit(X_train, y_train)
    print_validasi(best_pipeline, X_train, y_train)
    print(best_params)
    
    return study, best_pipeline

default_param = {'xgbclassifier__subsample': [0.7210103240835263],
 'xgbclassifier__reg_lambda': [30.571961951412387],
 'xgbclassifier__objective': ['binary:hinge'],
 'xgbclassifier__n_estimators': [49],
 'xgbclassifier__min_child_weight': [5.148736081496221],
 'xgbclassifier__max_depth': [14],
 'xgbclassifier__learning_rate': [0.020971884833061095],
 'xgbclassifier__gamma': [0.5181639744254346],
 'xgbclassifier__eta': [0.24065854342316315],
 'xgbclassifier__colsample_bytree': [0.5444278765166607]}

def fit_and_test_xgb(X_train, y_train, X_test, y_test, param=default_param, train_score=False):
    xgb = XGBClassifier(random_state=42, n_jobs=1)
    param_grid = param
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    pipeline = make_pipeline(SMOTE(random_state=42), xgb)
    grid_search = GridSearchCV(pipeline, param_grid, cv=kf, scoring='f1_macro', return_train_score=train_score, verbose=2)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    # y_pred = post_process(X_train, y_pred)
    print_score_uji(y_test, y_pred, grid_search)
    best_xgb_model = best_model.named_steps['xgbclassifier']
    plt.figure(figsize=(30, 30))
    plot_tree(best_xgb_model, num_trees=300, rankdir='LR')
    plt.savefig("xgb_tree_high_res.png", dpi=300)
    plt.show()
    # plot_tree(best_xgb_model, num_trees=1, rankdir='LR')
    # plt.show()
    return grid_search, best_model

def fit_train_xgb(X_train, y_train, param=default_param, train_score=False):
    xgb = XGBClassifier(random_state=42,  n_jobs=1)
    param_grid = param
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    pipeline = make_pipeline(SMOTE(random_state=42), xgb)
    grid_search = GridSearchCV(pipeline, param_grid, cv=kf, scoring='f1_macro', return_train_score=train_score, verbose=2)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    print_validasi(best_model, X_train, y_train)

    return grid_search, best_model

def fit_train_loop_xgb(X_train, y_train, param=default_param, train_score=False):
    xgb = XGBClassifier(random_state=42)
    param_grid = param
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    pipeline = make_pipeline(SMOTE(random_state=42), xgb)
    grid_search = GridSearchCV(pipeline, param_grid, cv=kf, scoring='f1_macro', return_train_score=train_score, verbose=2)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='macro')
    return grid_search, best_model, f1

def fit_train_xgb_doang(X_train, y_train, param=default_param, train_score=False):
    xgb = XGBClassifier(random_state=42,  n_jobs=1)
    param_grid = param
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    pipeline = make_pipeline(SMOTE(random_state=42), xgb)
    grid_search = GridSearchCV(pipeline, param_grid, cv=kf, scoring='f1_macro', return_train_score=train_score, verbose=2)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    
    y_pred = best_model.predict(X_train)
    accuracy = accuracy_score(y_train, y_pred)
    precision = precision_score(y_train, y_pred, average='macro')
    recall = recall_score(y_train, y_pred, average='macro')
    f1 = f1_score(y_train, y_pred, average='macro')

    print("Akurasi: %0.3f" % accuracy)
    print("Presisi: %0.3f" % precision)
    print("Recall: %0.3f" % recall)
    print("F1 validasi: %0.3f" % f1)
