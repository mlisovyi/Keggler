__all__ = ['StackingClassifier']

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn import model_selection
from sklearn.base import BaseEstimator, MetaEstimatorMixin, ClassifierMixin
from sklearn.base import clone
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold, StratifiedKFold
import lightgbm as lgb
from joblib import Parallel, delayed
import time

def _fit_and_score(model, X, y, trn_idx, val_idx, metric, predict_str='predict', verbose=True):
    '''
    A helper function that is used for parallel model fiting in the CV loop.
    The fitting procedure in the case of lightgbm.LGBMClassifier model uses early stopping
    based on the ROC AUC score evaluated on the validation subsample.
    In this special scenario the evaluation metric is also printed every 100 fit iteration
    if `verbose==True`.

    Parameters
    -----------
    model : the sklearn-style model to be fitted
    X : pandas.DataFrame, (n_rows, n_cols)
        The dataset that is used in CV loop
    y : pandas.Series, (n_rows, 1)
        The target that correspond to the provided X
    trn_idx : array-like, (n_rows_train, 1)
        The indices of events to be used for training 
    val_idx : array-like, (n_rows_valid, 1)
        The indices of eents to be used for evaluation of model performance
    metric : a collable
        The metric function that will be evaluated on the vadiation subset
        The interface should be `metric(y_true, y_pred)`
    predict_str: str [default: 'predict']
        The name of the `model` method to be used for prediction
    verbose: bool [default: True]
        Enable extended output dump

    Returns
    ---------
    model: sklearn-style model 
        The fitted model
    y_pred_val: array like, (n_rows_valid, 1)
        The out-of-fold predictions on the validation subset
    score_fold: float
        The metric evaluated on the validation subset
    '''
    X_trn, y_trn = X.iloc[trn_idx, :], y.iloc[trn_idx]
    X_val, y_val = X.iloc[val_idx, :], y.iloc[val_idx]

    fit_params={}
    # In the case of LGBMClassifier add early stopping using the validation subset
    if isinstance(model, lgb.LGBMClassifier):
        fit_params={'eval_set': [(X_trn, y_trn), (X_val, y_val)], 
                    'eval_metric':'auc', 
                    'verbose':100 if verbose else False, 
                    'early_stopping_rounds':100}
    model = model.fit(X_trn, y_trn, **fit_params)
    
    # OOF prediction
    y_pred_val = getattr(model, predict_str)(X_val)
    if predict_str != 'predict':
        y_pred_val = y_pred_val[:,1]
    # performance fo the meta model on the validation fold
    score_fold = metric(y_val, y_pred_val)
    
    return (model, y_pred_val, score_fold)


class BaseStackingEstimator(BaseEstimator, MetaEstimatorMixin):
    '''
    A base class for building a stack-/meta-model.
    In contrast to several other stacking implementations,
    this class does not perform a fit of daugher models.
    In contrast, it operated on provided pre-computed out-of-fold, OOF,
    predictions as well as predictions for the full test/submission sample.
    In practise it stores in `self.meta_models_folds` a set of meta-models,
    that were fit on individual CV iterations.

    Parameters
    ----------
    meta_model : sklearn-style model
        The meta-model to be used in fitting.
    oof_preds : a dictionary of {name: array-like OOF predictions} or a pandas.DataFrame, (n_rows_train, n_models)
        The OOF predictions for input model.
    test_preds :  a dictionary of {name: array-like test predictions} or a pandas.DataFrame, (n_rows_test, n_models)
        The test predictions for input models. 
        The dictionary keys or the DataFrame columns has to match those of `oof_preds`.
    cv : sklearn-style CV object
        The object with `.split(X,y)` method that returns a generator of training and validation indices.
    metric : callable
        The evaluation metric to be used.
        Expected interface is  `metric(y_true, y_pred)`.
    n_jobs : int [default: 1]
        The number of CV jobs to be run in parallel.
    use_base_features : bool [default: False]
        Fit a model on a concatenation of the provided lower-level OOF model predictions
        with the `X` dataset provided in the `fit()` method.
    verbose : bool [default: False]
        Enable extended output dump. 
        In particular, prints `metric` evaluated for the input OOF predictions for each model.
    use_ranks : bool [default: False]
        Use ranks instead of the raw OOF and test inputs.
        Shows better performance in classification problems with ROC AUC score.
    scale_proba : bool [default: False]
        Use scaling of all provided OOF and test predictions to [0,1] interval.
        Can be used as an alternative to `use_ranks`, 
        but in a couple of tested examples has shown worse ROC AUC performance.
        This preprocessing is relevant, as various models populate different ranges of probability values.
    use_aggs : array-like structure of strings or string or `None` [default: None]
        A list of or a single aggregation to be computed across OOF and test inputs.
    score_categories : bool [default: False]
        Evaluate and dump metric for different categories in the dataset provided in the `fit()` method
    score_categories_cols : list of strings or None [default: None]
        The list of columns to be used if `score_categories == True`.
        `None` corresponds to all columns in the dataset.
    '''
    def __init__(self, meta_model, oof_preds, test_preds, cv, metric, n_jobs=1,
                 use_base_features=False, verbose=False, 
                 use_ranks=False, scale_proba=False, use_aggs=None,
                 score_categories=False, score_categories_cols=None):
        self.meta_model = meta_model
        self.oof_preds = pd.DataFrame(oof_preds)
        self.test_preds = pd.DataFrame(test_preds)
        self.cv = cv
        self.metric = metric
        self.n_jobs = n_jobs
        self.use_base_features = use_base_features
        self.verbose = verbose
        self.use_ranks = use_ranks
        self.scale_proba = scale_proba
        self.use_aggs = use_aggs
        self.meta_oof = None
        self.meta_models_folds = []
        self.predict_str = 'predict'
        self.score_categories = score_categories
        self.score_categories_cols = score_categories_cols
        if self.use_ranks and self.scale_proba:
            print('Can not use both ranks and scaling. Will set both options to False')
            self.use_ranks = False
            self.scale_proba = False

    def fit(self, X, y=None, **fit_params):  
        '''
        Fit the meta-model.
        `X` is *not* used unless `use_base_features` is set in the constructor
        '''
        X_fit = self._get_X(X, self.oof_preds)
        
        self.eval_results_ = {}
        self.eval_results_['meta_score_folds'] = []
        self.meta_oof = np.zeros(X.shape[0])
        # clear the list of models in the case of refit
        del self.meta_models_folds[:]
        print('Shape of datafit in fiting: {}'.format(X_fit.shape))
        
        t_start = time.time()
        # train model on the folds in parallel
        out = Parallel(
            n_jobs=self.n_jobs, verbose=self.verbose
                      )(delayed(_fit_and_score)(clone(self.meta_model), 
                                                X_fit, y, 
                                                trn_idx, val_idx,
                                                metric=self.metric,
                                                predict_str=self.predict_str,
                                                verbose=self.verbose)
                        for (trn_idx, val_idx) in self.cv.split(X_fit, y))
        # unpack outputs
        self.meta_models_folds, oof_preds, self.eval_results_['meta_score_folds'] = zip(*out)
        
        # store the oof predictions according to their indices
        for i, (_, val_idx) in enumerate(self.cv.split(X_fit, y)):
            self.meta_oof[val_idx] = oof_preds[i]
        
        # save timing performance
        t_end = time.time()        
        self.eval_results_['fit_time'] = t_end-t_start
        print('Total execution time of the kfold learning: {:.1f} sec'.format(self.eval_results_['fit_time']))
        
        # evaluation score within categories
        if self.score_categories_cols is not None:
            cols_2proc = self.score_categories_cols
        else:
            cols_2proc = [c for c in X.columns if c != 'SK_ID_CURR']
        if self.score_categories:
            # loop over columns in the input data
            for col in cols_2proc:
                # find unique values within the column
                categories = X[col].unique()
                # prepare data structure to store scores
                eval_name = '{}_score_folds'.format(col)
                self.eval_results_[eval_name] = {c: [] for c in categories}
                #loop over folds
                for i, (_, val_idx) in enumerate(self.cv.split(X, y)):
                    # C_val is the column from the input X 
                    C_val, y_val = X.iloc[val_idx, X.columns.get_loc(col)], y.iloc[val_idx]
                    for cat in categories:
                        # the list of booleans that map which entries belong to category
                        X_cat = (C_val == cat)

                        if y_val[X_cat].shape[0] == 0:
                            # assign -1 if category is not present in the validation set
                            self.eval_results_[eval_name][cat].append(-1)
                        elif y_val[X_cat].nunique() == 1:
                            # assign -2 if the target of the category in the validation set is the same (only 0 or only 1)
                            self.eval_results_[eval_name][cat].append(-2)
                        else:
                            # meaningful case of metric evaluation
                            self.eval_results_[eval_name][cat].append(self.metric(y_val[X_cat], oof_preds[i][X_cat]))
            
        # store average performance of the meta model
        self.eval_results_['meta_score_mean'] = np.mean(self.eval_results_['meta_score_folds'])
        self.eval_results_['meta_score_std'] = np.std(self.eval_results_['meta_score_folds'])
        # store performances of input models
        for model_name in self.oof_preds.columns:
            self.eval_results_['{}_score'.format(model_name)] = self.metric(y, self.oof_preds[model_name])

        # Info dump
        if self.verbose:
            for model_name in self.oof_preds.columns:
                print('{} score: {:.5f}'.format(model_name, 
                                                self.eval_results_['{}_score'.format(model_name)]))
        if self.score_categories:
            print('Scoring metric withing categories of input data')
            for col in cols_2proc:
                print('------- {} -------------'.format(col))
                for k,l in self.eval_results_['{}_score_folds'.format(col)].items():
                    print('{}: {:.5f}+-{:.5f} , {}'.format(k, np.mean(l), np.std(l), l))
        print('Scores in individual folds: {}'.format(self.eval_results_['meta_score_folds']))
        print('Stacked score: {:.5f} ± {:.5f}'.format(self.eval_results_['meta_score_mean'],
                                                      self.eval_results_['meta_score_std'] ))

        return self
    
    # Obsolete?
    #def _predict_oof(self, model, X):
    #    return model.predict(X)
    
    def _get_X(self, X, X_models):
        '''
        Retrieve dataset for fiting.
        Can merge provided `X` with `self.oof_preds` if `use_base_features==True` in the constructor.
        Can add aggregates for each row if `use_aggs` is set in the constructor.
        '''
        # preprocess X_models
        X_tmp = X_models.copy(deep=True)
        if self.use_ranks:
            for f_ in X_tmp.columns:
                X_tmp[f_] = X_tmp[f_].argsort().argsort() / X_tmp[f_].shape[0]
        if self.scale_proba:
            X_tmp = pd.DataFrame(MinMaxScaler().fit_transform(X_tmp), columns=X_tmp.columns)

        # Add aggregates 
        if self.use_aggs is not None:
            X_agg = pd.DataFrame(index=X_models.index)
            if isinstance(self.use_aggs, str):
                self.use_aggs = list(self.use_aggs)
            if isinstance(self.use_aggs, list):
                for l in self.use_aggs:
                    X_agg['agg_{}'.format(l)] = getattr(X_models, l)(axis=1)
                X_tmp = pd.concat([X_agg, X_tmp], axis=1)
            else:
                print('The `use_aggs` type is not recognised. No aggregation will be added.')
        
        # oof_preds should have as many rows as there are in X and as many
        # columns as there are models.  
        if self.use_base_features:
            if isinstance(X, pd.DataFrame):
                X_out = pd.concat([X_tmp, X], axis=1)
            else:
                X_out = pd.concat([X_tmp, pd.DataFrame(X, columns=list(range(X.shape[1])))], axis=1)
        else:
            X_out = X_tmp.copy(deep=True)
            
        del X_tmp
        return X_out

    def predict(self, X):
        '''
        Dummy implementation. To be completed, once a meta-regressor is implemented.
        '''
        return None


class StackingClassifier(BaseStackingEstimator, ClassifierMixin):
    '''
    A meta-classifier.
    Uses stratified folds.
    Most parameters follow the base-class implementation.
    Only additional will be described below.

    Parameters
    ----------
    n_cv : int [default: 5]
        The number of folds to be used in split.
    random_state: int or None [default: None]
        The random state to be used in the folds splitting.
    '''
    def __init__(self, meta_model, oof_preds, test_preds, n_cv=5, random_state=None, n_jobs=1,
                 metric=metrics.roc_auc_score, use_base_features=False, verbose=False, 
                 use_ranks=False, scale_proba=False, use_aggs=None,
                 score_categories=False, score_categories_cols=None):
        self.n_cv = n_cv
        self.random_state = random_state
        super().__init__(
            meta_model=meta_model,
            oof_preds=oof_preds,
            test_preds=test_preds,
            cv=StratifiedKFold(n_splits=n_cv, shuffle=True, random_state=random_state),
            n_jobs=n_jobs,
            metric=metric,
            use_base_features=use_base_features,
            verbose=verbose,
            use_ranks=use_ranks,
            scale_proba=scale_proba,
            use_aggs=use_aggs,
            score_categories = score_categories,
            score_categories_cols = score_categories_cols
        )
        self.predict_str='predict_proba'

    # Obsolete?
    #def _predict_oof(self, model, X):
    #    return model.predict_proba(X)[:,1]
    
    def predict_proba(self, X):
        '''
        Since the meta-model has no access to the input models,
        predictions can be made only for the OOF and test samples,
        where model predictions are already provided.
        An average of stored meta-model predictions is returned.
        '''
        if X.shape[0] == self.oof_preds.shape[0]:
            if self.verbose:
                print('Predicting for the TRAIN sample')
            X_oof = self.oof_preds
        elif X.shape[0] == self.test_preds.shape[0]:
            if self.verbose:
                print('Predicting for the SUBMISSION sample')
            X_oof = self.test_preds
        else:
            print('Predicting for neither TRAIN nor SUBMISSION is not supported. Exiting...')
            return None

        X_pred = self._get_X(X, X_oof)

        preds = np.zeros((X_pred.shape[0],2))
        for i in range(self.cv.n_splits):
            preds += self.meta_models_folds[i].predict_proba(X_pred)
        preds /= self.cv.n_splits
        
        return preds

