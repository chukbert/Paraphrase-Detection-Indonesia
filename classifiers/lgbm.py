import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))

from evaluation_matrics import print_validasi, print_score_uji, print_score_opt_uji


import optuna
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
        
def objective_lgbm(trial, X, y):
    num_leaves = trial.suggest_int('lgbmclassifier__num_leaves', 20, 150)
    min_data_in_leaf = trial.suggest_int('lgbmclassifier__min_data_in_leaf', 10, 100)
    max_depth = trial.suggest_int('lgbmclassifier__max_depth', -1, 15)
    learning_rate = trial.suggest_float('lgbmclassifier__learning_rate', 0.01, 0.3)
    n_estimators = trial.suggest_int('lgbmclassifier__n_estimators', 10, 1000)
    subsample = trial.suggest_float('lgbmclassifier__subsample', 0.5, 1.0)
    colsample_bytree = trial.suggest_float('lgbmclassifier__colsample_bytree', 0.5, 1.0)
    reg_alpha = trial.suggest_float('lgbmclassifier__reg_alpha', 0, 10.0)
    reg_lambda = trial.suggest_float('lgbmclassifier__reg_lambda', 0, 10.0)
    boosting = trial.suggest_categorical('lgbmclassifier__boosting', ['gbdt', 'rf', 'dart', 'goss'])

    lgbm = LGBMClassifier(n_jobs=1, verbose=-1, num_leaves=num_leaves, min_data_in_leaf=min_data_in_leaf, max_depth=max_depth, learning_rate=learning_rate, n_estimators=n_estimators, subsample=subsample, colsample_bytree=colsample_bytree, reg_alpha=reg_alpha, reg_lambda=reg_lambda, objective='binary', boosting_type=boosting, random_state=42)
    pipeline = make_pipeline(SMOTE(random_state=42), lgbm)
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    return cross_val_score(pipeline, X, y, cv=kf, scoring='f1_macro', n_jobs=-1).mean()

def find_best_param_lgbm(X_train, y_train, n_trials=100, timeout=100, list_param=None):
    study = optuna.create_study(direction='maximize')
    study.enqueue_trial({'lgbmclassifier__num_leaves': 31, 'lgbmclassifier__min_data_in_leaf': 20, 'lgbmclassifier__max_depth': -1, 'lgbmclassifier__learning_rate': 0.1, 'lgbmclassifier__n_estimators': 100, 'lgbmclassifier__subsample': 1.0, 'lgbmclassifier__colsample_bytree': 1.0, 'lgbmclassifier__reg_alpha': 0.0, 'lgbmclassifier__reg_lambda': 0.0, 'lgbmclassifier__boosting': 'gbdt'})
    if list_param:
        for param in list_param:
            study.enqueue_trial(param)
    study.optimize(lambda trial: objective_lgbm(trial, X_train, y_train), n_trials=n_trials, timeout=timeout)

    best_params = study.best_params
    best_lgbm = LGBMClassifier(verbose=-1,
                               num_leaves=best_params['lgbmclassifier__num_leaves'],
                               min_data_in_leaf=best_params['lgbmclassifier__min_data_in_leaf'],
                               max_depth=best_params['lgbmclassifier__max_depth'],
                               learning_rate=best_params['lgbmclassifier__learning_rate'],
                               n_estimators=best_params['lgbmclassifier__n_estimators'],
                               subsample=best_params['lgbmclassifier__subsample'],
                               colsample_bytree=best_params['lgbmclassifier__colsample_bytree'],
                               reg_alpha=best_params['lgbmclassifier__reg_alpha'],
                               reg_lambda=best_params['lgbmclassifier__reg_lambda'],
                               objective='binary', boosting_type=best_params['lgbmclassifier__boosting'], random_state=42)
    best_pipeline = make_pipeline(SMOTE(random_state=42), best_lgbm)
    best_pipeline.fit(X_train, y_train)
    print_validasi(best_pipeline, X_train, y_train)
    print(best_params)
    
    return study, best_pipeline

default_param = {'lgbmclassifier__boosting': 'dart', 'lgbmclassifier__colsample_bytree': 0.8062362343663648, 
                 'lgbmclassifier__learning_rate': 0.3156157906329412, 'lgbmclassifier__max_depth': 9, 
                 'lgbmclassifier__min_data_in_leaf': 30, 'lgbmclassifier__n_estimators': 996, 
                 'lgbmclassifier__num_leaves': 64, 'lgbmclassifier__reg_alpha': 0.7847950336434518, 
                 'lgbmclassifier__reg_lambda': 60.13383354494465, 'lgbmclassifier__subsample': 0.38926772788483466}
default_param = {k: [v] for k, v in default_param.items()}

def fit_and_test_lgbm(X_train, y_train, X_test, y_test, param=default_param, train_score=False):
    lgbm = LGBMClassifier(verbose=-1, random_state=42)
    param_grid = param
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    pipeline = make_pipeline(SMOTE(random_state=42), lgbm)
    grid_search = GridSearchCV(pipeline, param_grid, cv=kf, scoring='f1_macro', return_train_score=train_score, verbose=2)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    # y_pred = post_process(X_train, y_pred)
    
    print_score_uji(y_test, y_pred, grid_search)
    return grid_search, best_model

def fit_train_lgbm(X_train, y_train, param=default_param, train_score=False):
    lgbm = LGBMClassifier(verbose=-1, random_state=42)
    param_grid = param
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    pipeline = make_pipeline(SMOTE(random_state=42), lgbm)
    grid_search = GridSearchCV(pipeline, param_grid, cv=kf, scoring='f1_macro', return_train_score=train_score, verbose=2)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    print_validasi(best_model, X_train, y_train)

    return grid_search, best_model

def fit_train_loop_lgbm(X_train, y_train, param=default_param, train_score=False):
    lgbm = LGBMClassifier(verbose=-1, random_state=42)
    param_grid = param
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    pipeline = make_pipeline(SMOTE(random_state=42), lgbm)
    grid_search = GridSearchCV(pipeline, param_grid, cv=kf, scoring='f1_macro', return_train_score=train_score, verbose=2)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='macro')
    return grid_search, best_model, f1