import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))

from evaluation_matrics import print_validasi, print_score_uji, print_score_opt_uji

import optuna
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

def objective_rf(trial, X, y):
    n_estimators = trial.suggest_int('randomforestclassifier__n_estimators', 10, 1000)
    max_features = trial.suggest_categorical('randomforestclassifier__max_features', ['sqrt', 'log2', None])
    max_depth = trial.suggest_int('randomforestclassifier__max_depth', 10, 100, log=True)
    criterion = trial.suggest_categorical('randomforestclassifier__criterion', ['gini', 'entropy'])
    max_leaf_nodes = trial.suggest_int('randomforestclassifier__max_leaf_nodes', 10, 1000, log=True)
    bootstrap = trial.suggest_categorical('randomforestclassifier__bootstrap', [True, False])

    rf = RandomForestClassifier(n_estimators=n_estimators, max_features=max_features,
                                max_depth=max_depth, criterion=criterion,
                                max_leaf_nodes=max_leaf_nodes, bootstrap=bootstrap,
                                random_state=42)
    pipeline = make_pipeline(SMOTE(random_state=42), rf)
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    return cross_val_score(pipeline, X, y, cv=kf, scoring='f1_macro', n_jobs=-1).mean()

def find_best_param_rf(X_train, y_train, n_trials=100, timeout=300, list_param=None):
    study = optuna.create_study(direction='maximize')
    study.enqueue_trial({'randomforestclassifier__n_estimators': 100, 'randomforestclassifier__max_features': 'sqrt', 'randomforestclassifier__max_depth': 100, 'randomforestclassifier__criterion': 'gini', 'randomforestclassifier__max_leaf_nodes': 1000, 'randomforestclassifier__bootstrap': True})
    # study.enqueue_trial({'randomforestclassifier__n_estimators': 997, 'randomforestclassifier__max_features': 'log2', 'randomforestclassifier__max_depth': 11, 'randomforestclassifier__criterion': 'entropy', 'randomforestclassifier__max_leaf_nodes': 46, 'randomforestclassifier__bootstrap': True})
    if list_param:
        for param in list_param:
            study.enqueue_trial(param)
    study.optimize(lambda trial: objective_rf(trial, X_train, y_train), n_trials=n_trials, timeout=timeout)

    best_params = study.best_params
    best_rf = RandomForestClassifier(n_estimators=best_params['randomforestclassifier__n_estimators'],
                                     max_features=best_params['randomforestclassifier__max_features'],
                                     max_depth=best_params['randomforestclassifier__max_depth'],
                                     criterion=best_params['randomforestclassifier__criterion'],
                                     max_leaf_nodes=best_params['randomforestclassifier__max_leaf_nodes'],
                                     bootstrap=best_params['randomforestclassifier__bootstrap'],
                                     random_state=42)
    best_pipeline = make_pipeline(SMOTE(random_state=42), best_rf)
    best_pipeline.fit(X_train, y_train)
    print_validasi(best_pipeline, X_train, y_train)
    print(best_params)
    return study, best_pipeline

default_param = {}

def fit_and_test_rf(X_train, y_train, X_test, y_test, param=default_param, train_score=False):
    rf = RandomForestClassifier(random_state=42)
    param_grid = param
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    pipeline = make_pipeline(SMOTE(random_state=42), rf)
    grid_search = GridSearchCV(pipeline, param_grid, cv=kf, scoring='f1_macro', return_train_score=train_score, verbose=2)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    # y_pred = post_process(X_train, y_pred)
    print_score_uji(y_test, y_pred, grid_search)
    return grid_search, best_model

def fit_train_rf(X_train, y_train, param=default_param, train_score=False):
    rf = RandomForestClassifier(random_state=42)
    param_grid = param
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    pipeline = make_pipeline(SMOTE(random_state=42), rf)
    grid_search = GridSearchCV(pipeline, param_grid, cv=kf, scoring='f1_macro', return_train_score=train_score, verbose=2)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    print_validasi(best_model, X_train, y_train)

    return grid_search, best_model

def fit_train_loop_rf(X_train, y_train, param=default_param, train_score=False):
    rf = RandomForestClassifier(random_state=42)
    param_grid = param
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    pipeline = make_pipeline(SMOTE(random_state=42), rf)
    grid_search = GridSearchCV(pipeline, param_grid, cv=kf, scoring='f1_macro', return_train_score=train_score, verbose=2)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='macro')
    return grid_search, best_model, f1