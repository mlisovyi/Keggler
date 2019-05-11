__all__=[
         'train_model_in_nestedCV', 'print_nested_perf_clf',
         'train_model_in_CV', 'print_perf_clf'
        ]

from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone, ClassifierMixin, RegressorMixin
import pandas as pd
import numpy as np
import time

from keggler.ensemble import *

def train_single_model(clf_, X_, y_, random_state_=314, opt_parameters_={}, fit_params_={}):
    '''
    Single-liner to fit a copy of `clf_` model with `opt_parameters_` parameters and 
    `fit_params_` parameters passed to the `fit` method
    
    Returns
    ---------
    c: Fitted model
    '''
    c = clone(clf_)
    c.set_params(**opt_parameters_)
    c.set_params(random_state=random_state_)
    return c.fit(X_, y_, **fit_params_)


def train_model_in_CV(model, X, y, metric, metric_args={},
                            model_name='xmodel',
                            seed=31416, n=5,
                            opt_parameters_={}, fit_params_={},
                            verbose=True,
                            groups=None, y_eval=None,
                            mlf=None, mlf_metric_name=None,
                            do_stratifiedkfold=True
                     ):
    '''
    Single-liner to do performance evaluation of multi-class `model` in a single stratified CV.
    The base model is are evaluated in the CV loop.

    Parameters
    -----------
    model: a model complying with sklearn API (has to work with `clone(model)`)
    X, y: pandas DataFrame and Series
        Training data and the target
    metric: func
        The function to evaluate the model
    metric_args: dict [default: {}]
        Additional arguments to feed into the metric evaluation.
        Options: `sample_weight`- weights per row, `index_weight`- weight per target class,
        `sqrt`- take sqrt of the evaluate metric.
    model_name: string
        The string to be used to identify model in the output dictionaries.
    seed: int [default: 31416]
        The random number seed to be used in CV.
    n: int [default: 5]
        The number of folds in CV.
    opt_parameters_, fit_params_: dict [default: {}]
        The parameters to be used by the model (set via `model.set_params()`)
        as well as parameters used during the fit (`model.fit()`)
    verbose: int, bool, None [default: True]
        The verbosity level in model fitting (applicable for GBMs)
    groups: pandas Series, None [default: None]
        Group to which each instance belongs (if there are groups within the dataset).
        The length has to be the same as `y`.
        Note, that groups have to be not interrupted. 
        Triggers usage of `GroupKFold` in CV.
    y_eval: pandas Series, None [default: None]
        Target that was preprocessed in some way to fit into the chosen metric.
        For example, use binary target in ROC AUC while the actual task is a regression problem.
    mlf: MLflow client instance, None [default: None]
        The MLflow client to be used for metrics tracking, if required.
        MLflow run has to be started in advance.
    mlf_metric_name: string, None [default: {}]
        The string preffix for the metric name to be used in tracking.
    do_stratifiedkfold: bool [default: True]
        Use stratified or regular kfold.
    
    Returns
    -----------
    clfs: A list of `model` copies fitted in the outer CV loop. 
        These can be used to implement voting, as they wre fit on partially different data.
    perf_eval: A dictionary storing evaluation metric calculated at different stages.
        The dictionary contains the following keys: 
        `['score_o_early_stop_vc_w0_hard', 'score_o_early_stop_vc_w0_soft', 'score_o_early_stop', 
        'score_i_ave', 'score_i_std', 'score_i_oof', 'score_i_early_stop_ave']`.
    y_full_oof: the OOF predictions for the full dataset predicted by the models in the outer CV loop.
    features: pandas DataFrame
        Feature importances in each CV fold.
    '''
    # the list of classifiers for voting ensable
    clfs = []
    # performance 
    perf_eval = {'score_i_oof': 0,
                 'score_i_ave': 0,
                 'score_i_std': 0,
                 'score_i': [],
                 'fit_time': []
                }
    # full-sample oof prediction
    y_full_oof = pd.Series(np.zeros(shape=(y.shape[0],)), 
                          index=y.index)
    
    sample_weight=None
    if 'sample_weight' in metric_args:
        sample_weight=metric_args['sample_weight']
        
    index_weight=None
    if 'index_weight' in metric_args:
        index_weight=metric_args['index_weight']
        del metric_args['index_weight']
        
    doSqrt=False
    if 'sqrt' in metric_args:
        doSqrt=True
        del metric_args['sqrt']

    if groups is None:
        if do_stratifiedkfold:
            cv = StratifiedKFold(n, shuffle=True, random_state=seed)
        else:
            cv = KFold(n, shuffle=True, random_state=seed)
    else:
        cv = GroupKFold(n)
    # The out-of-fold (oof) prediction for the k-1 sample in the outer CV loop
    y_oof = pd.Series(np.zeros(shape=(X.shape[0],)), 
                      index=X.index)
    scores = []
    clfs = []
    feature_importances = []

    for n_fold, (trn_idx, val_idx) in enumerate(cv.split(X, (y!=0).astype(np.int8), groups=groups)):
        X_trn, y_trn = X.iloc[trn_idx], y.iloc[trn_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        if 'LGBMRanker' in type(model).__name__ and groups is not None:
            G_trn, G_val = groups.iloc[trn_idx], groups.iloc[val_idx]        

        if fit_params_:
            # use _stp data for early stopping
            fit_params_["eval_set"] = [(X_trn,y_trn), (X_val,y_val)]
            fit_params_['verbose'] = verbose
            if index_weight is not None:
                fit_params_["sample_weight"] = y_trn.index.map(index_weight).values
                fit_params_["eval_sample_weight"] = [None, y_val.index.map(index_weight).values]
            if 'LGBMRanker' in type(model).__name__ and groups is not None:
                fit_params_['group'] = G_trn.groupby(G_trn, sort=False).count()
                fit_params_['eval_group'] = [G_trn.groupby(G_trn, sort=False).count(),
                                             G_val.groupby(G_val, sort=False).count()]

        #display(y_trn.head())
        t = time.time()
        clf = train_single_model(model, X_trn, y_trn, 314+n_fold, opt_parameters_, fit_params_)
        perf_eval['fit_time'].append(time.time()-t)

        clfs.append(('{}{}'.format(model_name,n_fold), clf))
        # oof predictions
        if isinstance(clf, RegressorMixin):
            y_oof.iloc[val_idx] = clf.predict(X_val)
        elif isinstance(clf, ClassifierMixin):
            y_oof.iloc[val_idx] = clf.predict_proba(X_val)[:,1]
        else:
            y_oof.iloc[val_idx] = clf.predict(X_val)
        # prepare weights for evaluation
        if sample_weight is not None:
            metric_args['sample_weight'] = y_val.map(sample_weight)
        elif index_weight is not None:
            metric_args['sample_weight'] = y_val.index.map(index_weight).values
        # prepare target values
        y_true_tmp = y_val if 'LGBMRanker' not in type(model).__name__  and y_eval is None else y_eval.iloc[val_idx]
        y_pred_tmp = y_oof.iloc[val_idx] if y_eval is None else y_oof.iloc[val_idx]        
        #store evaluated metric
        metric_value = metric(y_true_tmp, y_pred_tmp, **metric_args)
        scores.append(metric_value)
        if mlf is not None:
            mlf.log_metric("{}_Fold{}".format(mlf_metric_name, n_fold), metric_value)
        #
        fi_tmp = pd.DataFrame()
        fi_tmp["feature"] = X.columns
        if hasattr(clf, 'feature_importances_'):
            fi_tmp["importance"] = clf.feature_importances_
        fi_tmp["fold"] = n_fold + 1
        feature_importances.append(fi_tmp)
        #cleanup
        del X_trn, y_trn, X_val, y_val, y_true_tmp, y_pred_tmp

    # Store performance info for this CV
    if sample_weight is not None:
        metric_args['sample_weight'] = y_oof.map(sample_weight)
    elif index_weight is not None:
        metric_args['sample_weight'] = y_oof.index.map(index_weight).values
    perf_eval['score_i_oof'] = metric(y, y_oof, **metric_args)
    perf_eval['score_i'] = scores
    
    if doSqrt:
        for k in perf_eval.keys():
            if 'score' in k:
                perf_eval[k] = np.sqrt(perf_eval[k])
        scores = np.sqrt(scores)
            
    perf_eval['score_i_ave'] = np.mean(scores)
    perf_eval['score_i_std'] = np.std(scores)

    return clfs, perf_eval, y_oof, pd.concat(feature_importances, axis=0)



def train_model_in_nestedCV(model, X, y, metric, metric_args={},
                            model_name='xmodel',
                            inner_seed=31416, inner_n=10, outer_seed=314, outer_n=10,
                            opt_parameters_={}, fit_params_={},
                            verbose=True):
    '''
    Single-liner to do performance evaluation of multi-class `model` in a nested stratified CV.
    In addition to the base model, also Voting Classifiers are evaluated in the outer CV loop.
    
    Parameters
    -----------
    model: a model complying with sklearn API (has to work with `clone(model)`)
    X, y: pandas DataFrame and Series
        Training data and the target
    metric: func
        The function to evaluate the model
    metric_args: dict [default: {}]
        Additional arguments to feed into the metric evaluation.
        Options: `sample_weight`- weights per row, `index_weight`- weight per target class,
        `sqrt`- take sqrt of the evaluate metric.
    model_name: string
        The string to be used to identify model in the output dictionaries.
    inner_seed, outer_seed: int [default: 31416 and 314]
        The random number seeds to be used in the inner and outer CV splits.
    inner_n, outer_n: int [default: 10 and 10]
        The number of folds in CV.
    opt_parameters_, fit_params_: dict [default: {}]
        The parameters to be used by the model (set via `model.set_params()`)
        as well as parameters used during the fit (`model.fit()`).
    verbose: int, bool, None [default: True]
        The verbosity level in model fitting (applicable for GBMs).
    mlf: MLflow client instance, None [default: None]
        The MLflow client to be used for metrics tracking, if required.
        MLflow run has to be started in advance.
    mlf_metric_name: string, None [default: {}]
        The string preffix for the metric name to be used in tracking.
    do_stratifiedkfold: bool [default: True]
        Use stratified or regular kfold.

    Returns
    -----------
    clfs: A list of `model` copies fitted in the outer CV loop. 
        These can be used to implement voting, as they wre fit on partially different data
    perf_eval: A dictionary storing evaluation metric calculated at different stages.
        The dictionary contains the following keys: 
        `['score_o_early_stop_vc_w0_hard', 'score_o_early_stop_vc_w0_soft', 'score_o_early_stop', 
        'score_i_ave', 'score_i_std', 'score_i_oof', 'score_i_early_stop_ave']`
    y_full_oof: the OOF predictions for the full dataset predicted by the models in the outer CV loop
    '''
    # the list of classifiers for voting ensable
    clfs = []
    # performance 
    perf_eval = {'score_i_oof': [],
                 'score_i_ave': [],
                 'score_i_std': [],
                 'score_i_early_stop_ave': [],
                 'score_o_early_stop': [],
                 'score_o_early_stop_vc_w0_soft': [],
                 'score_o_early_stop_vc_w0_hard': []
                }
    # full-sample oof prediction
    y_full_oof = pd.Series(np.zeros(shape=(y.shape[0],)), 
                          index=y.index)
    
    if 'sample_weight' in metric_args:
        sample_weight=metric_args['sample_weight']

    outer_cv = StratifiedKFold(outer_n, shuffle=True, random_state=outer_seed)
    for n_outer_fold, (outer_trn_idx, outer_val_idx) in enumerate(outer_cv.split(X,y)):
        print('--- Outer loop iteration: {} ---'.format(n_outer_fold))
        X_out, y_out = X.iloc[outer_trn_idx], y.iloc[outer_trn_idx]
        X_stp, y_stp = X.iloc[outer_val_idx], y.iloc[outer_val_idx]
        
        if inner_n:
            inner_cv = StratifiedKFold(inner_n, shuffle=True, random_state=inner_seed+n_outer_fold)
            # The out-of-fold (oof) prediction for the k-1 sample in the outer CV loop
            y_outer_oof = pd.Series(np.zeros(shape=(X_out.shape[0],)), 
                                      index=X_out.index)
            scores_inner = []
            clfs_inner = []

            for n_inner_fold, (inner_trn_idx, inner_val_idx) in enumerate(inner_cv.split(X_out,y_out)):
                X_trn, y_trn = X_out.iloc[inner_trn_idx], y_out.iloc[inner_trn_idx]
                X_val, y_val = X_out.iloc[inner_val_idx], y_out.iloc[inner_val_idx]

                if fit_params_:
                    # use _stp data for early stopping
                    fit_params_["eval_set"] = [(X_trn,y_trn), (X_stp,y_stp)]
                    fit_params_['verbose'] = False

                clf = train_single_model(model, X_trn, y_trn, 314+n_inner_fold, opt_parameters_, fit_params_)

                clfs_inner.append(('{}{}_inner'.format(model_name,n_inner_fold), clf))
                # evaluate performance
                y_outer_oof.iloc[inner_val_idx] = clf.predict(X_val)
                if 'sample_weight' in metric_args:
                    metric_args['sample_weight'] = y_val.map(sample_weight)
                scores_inner.append(metric(y_val, y_outer_oof.iloc[inner_val_idx], **metric_args))
                #cleanup
                del X_trn, y_trn, X_val, y_val

            # Store performance info for this outer fold
            if 'sample_weight' in metric_args:
                metric_args['sample_weight'] = y_outer_oof.map(sample_weight)
            perf_eval['score_i_oof'].append(metric(y_out, y_outer_oof, **metric_args))
            perf_eval['score_i_ave'].append(np.mean(scores_inner))
            perf_eval['score_i_std'].append(np.std(scores_inner))
        
            # Do the predictions for early-stop sub-sample for comparison with VotingPrefitClassifier
            if 'sample_weight' in metric_args:
                metric_args['sample_weight'] = y_stp.map(sample_weight)
            score_inner_early_stop = [metric(y_stp, clf_.predict(X_stp), **metric_args)
                                       for _,clf_ in clfs_inner]
            perf_eval['score_i_early_stop_ave'].append(np.mean(score_inner_early_stop))
        
            # Record performance of Voting classifiers
            w = np.array(scores_inner)
            for w_, w_name_ in [(None, '_w0')#,
                                #(w/w.sum(), '_w1'),
                                #((w**2)/np.sum(w**2), '_w2')
                               ]:
                vc = VotingPrefitClassifier(clfs_inner, weights=w_).fit(X_stp, y_stp)
                for vote_type in ['soft', 'hard']:
                    vc.voting = vote_type
                    if 'sample_weight' in metric_args:
                        metric_args['sample_weight'] = y_stp.map(sample_weight)
                    perf_eval['score_o_early_stop_vc{}_{}'.format(w_name_, vote_type)].append(metric(y_stp, vc.predict(X_stp), **metric_args))
            # inner loop cleanup
            del inner_cv, clfs_inner

        if fit_params_:
            # Train main model for the voting average
            fit_params_["eval_set"] = [(X_out,y_out), (X_stp,y_stp)]
            if verbose:
                fit_params_['verbose'] = 200
        #print('Fit the final model on the outer loop iteration: ')
        clf = train_single_model(model, X_out, y_out, 314+n_outer_fold, opt_parameters_, fit_params_)
        if 'sample_weight' in metric_args:
            metric_args['sample_weight'] = y_stp.map(sample_weight)
        perf_eval['score_o_early_stop'].append(metric(y_stp, clf.predict(X_stp), **metric_args))
        clfs.append(('{}{}'.format(model_name,n_outer_fold), clf))
        y_full_oof.iloc[outer_val_idx] = clf.predict(X_stp)
        # cleanup
        del X_out, y_out, X_stp, y_stp

    return clfs, perf_eval, y_full_oof

#TODO: finish this version and test it
def train_model_in_nestedCV_USING_train_model_in_CV(model, X, y, metric, metric_args={},
                            model_name='xmodel',
                            inner_seed=31416, inner_n=10, outer_seed=314, outer_n=10,
                            opt_parameters_={}, fit_params_={},
                            verbose=True):
    '''
    Single-liner to do performance evaluation of multi-class `model` in a nested stratified CV.
    In addition to the base model, also Voting Classifiers are evaluated in the outer CV loop.
    
    Parameters
    -----------
    model: a model complying with sklearn API (has to work with `clone(model)`)
    X, y: pandas DataFrame and Series
        Training data and the target
    metric: func
        The function to evaluate the model
    metric_args: dict [default: {}]
        Additional arguments to feed into the metric evaluation.
        Options: `sample_weight`- weights per row, `index_weight`- weight per target class,
        `sqrt`- take sqrt of the evaluate metric.
    model_name: string
        The string to be used to identify model in the output dictionaries.
    inner_seed, outer_seed: int [default: 31416 and 314]
        The random number seeds to be used in the inner and outer CV splits.
    inner_n, outer_n: int [default: 10 and 10]
        The number of folds in CV.
    opt_parameters_, fit_params_: dict [default: {}]
        The parameters to be used by the model (set via `model.set_params()`)
        as well as parameters used during the fit (`model.fit()`).
    verbose: int, bool, None [default: True]
        The verbosity level in model fitting (applicable for GBMs).
    mlf: MLflow client instance, None [default: None]
        The MLflow client to be used for metrics tracking, if required.
        MLflow run has to be started in advance.
    mlf_metric_name: string, None [default: {}]
        The string preffix for the metric name to be used in tracking.
    do_stratifiedkfold: bool [default: True]
        Use stratified or regular kfold.

    Returns
    -----------
    clfs: A list of `model` copies fitted in the outer CV loop. 
        These can be used to implement voting, as they wre fit on partially different data
    perf_eval: A dictionary storing evaluation metric calculated at different stages.
        The dictionary contains the following keys: 
        `['score_o_early_stop_vc_w0_hard', 'score_o_early_stop_vc_w0_soft', 'score_o_early_stop', 
        'score_i_ave', 'score_i_std', 'score_i_oof', 'score_i_early_stop_ave']`
    y_full_oof: the OOF predictions for the full dataset predicted by the models in the outer CV loop
    '''
    # the list of classifiers for voting ensable
    clfs = []
    # performance 
    perf_eval = {'score_i_oof': [],
                 'score_i_ave': [],
                 'score_i_std': [],
                 'score_i_early_stop_ave': [],
                 'score_o_early_stop': [],
                 'score_o_early_stop_vc_w0_soft': [],
                 'score_o_early_stop_vc_w0_hard': []
                }
    # full-sample oof prediction
    y_full_oof = pd.Series(np.zeros(shape=(y.shape[0],)), 
                          index=y.index)
    
    if 'sample_weight' in metric_args:
        sample_weight=metric_args['sample_weight']

    outer_cv = StratifiedKFold(outer_n, shuffle=True, random_state=outer_seed)
    for n_outer_fold, (outer_trn_idx, outer_val_idx) in enumerate(outer_cv.split(X,y)):
        print('--- Outer loop iteration: {} ---'.format(n_outer_fold))
        X_out, y_out = X.iloc[outer_trn_idx], y.iloc[outer_trn_idx]
        X_stp, y_stp = X.iloc[outer_val_idx], y.iloc[outer_val_idx]
        
        if inner_n:
            clfs_inner, perf_eval, y_outer_oof, _ = train_model_in_CV(model, X_out, y_out, metric, metric_args, model_name,
                    seed=inner_seed, n=inner_n,
                    opt_parameters_=opt_parameters_, fit_params_=fit_params_,
                    verbose=verbose,
                    groups=None, y_eval=None, do_stratifiedkfold=True
                    )
#            scores_inner = []

#               scores_inner.append(metric(y_val, y_outer_oof.iloc[inner_val_idx], **metric_args))

            # Store performance info for this outer fold
            if 'sample_weight' in metric_args:
                metric_args['sample_weight'] = y_outer_oof.map(sample_weight)
            perf_eval['score_i_oof'].append(metric(y_out, y_outer_oof, **metric_args))
            perf_eval['score_i_ave'].append(np.mean(scores_inner))
            perf_eval['score_i_std'].append(np.std(scores_inner))
        
            # Do the predictions for early-stop sub-sample for comparison with VotingPrefitClassifier
            if 'sample_weight' in metric_args:
                metric_args['sample_weight'] = y_stp.map(sample_weight)
            score_inner_early_stop = [metric(y_stp, clf_.predict(X_stp), **metric_args)
                                       for _,clf_ in clfs_inner]
            perf_eval['score_i_early_stop_ave'].append(np.mean(score_inner_early_stop))
        
            # Record performance of Voting classifiers
            w = np.array(scores_inner)
            for w_, w_name_ in [(None, '_w0')#,
                                #(w/w.sum(), '_w1'),
                                #((w**2)/np.sum(w**2), '_w2')
                               ]:
                vc = VotingPrefitClassifier(clfs_inner, weights=w_).fit(X_stp, y_stp)
                for vote_type in ['soft', 'hard']:
                    vc.voting = vote_type
                    if 'sample_weight' in metric_args:
                        metric_args['sample_weight'] = y_stp.map(sample_weight)
                    perf_eval['score_o_early_stop_vc{}_{}'.format(w_name_, vote_type)].append(metric(y_stp, vc.predict(X_stp), **metric_args))
            # inner loop cleanup
            del inner_cv, clfs_inner

        if fit_params_:
            # Train main model for the voting average
            fit_params_["eval_set"] = [(X_out,y_out), (X_stp,y_stp)]
            if verbose:
                fit_params_['verbose'] = 200
        #print('Fit the final model on the outer loop iteration: ')
        clf = train_single_model(model, X_out, y_out, 314+n_outer_fold, opt_parameters_, fit_params_)
        if 'sample_weight' in metric_args:
            metric_args['sample_weight'] = y_stp.map(sample_weight)
        perf_eval['score_o_early_stop'].append(metric(y_stp, clf.predict(X_stp), **metric_args))
        clfs.append(('{}{}'.format(model_name,n_outer_fold), clf))
        y_full_oof.iloc[outer_val_idx] = clf.predict(X_stp)
        # cleanup
        del X_out, y_out, X_stp, y_stp

    return clfs, perf_eval, y_full_oof



def print_nested_perf_clf(name, perf_eval):
    '''
    Single-liner to dump to the screen performance of the trained models
    
    Parameters
    ----------
    name: string
        The name of the model. Is used for printouts only
    perf_eval: dict
        The dictionary storing info on model performance. 
        The dictionary should be similar to the one output by the `train_model_in_nestedCV` function
    '''
    isNestedLoop = (len(perf_eval['score_i_ave']) > 0)
    if isNestedLoop:
        print('Performance of the inner-loop model (the two should agree):')
        print('  Mean(mean(Val)) score inner {} Classifier: {:.4f}+-{:.4f}'.format(name, 
                                                                          np.mean(perf_eval['score_i_ave']),
                                                                          np.std(perf_eval['score_i_ave'])
                                                                         ))
        print('  Mean(mean(EarlyStop)) score inner {} Classifier: {:.4f}+-{:.4f}'.format(name, 
                                                                          np.mean(perf_eval['score_i_early_stop_ave']),
                                                                          np.std(perf_eval['score_i_early_stop_ave'])
                                                                         ))
        print('Mean(inner OOF) score inner {} Classifier: {:.4f}+-{:.4f}'.format(name, 
                                                                           np.mean(perf_eval['score_i_oof']), 
                                                                           np.std(perf_eval['score_i_oof'])
                                                                          ))
    # outer loop performance
    print('Mean(EarlyStop) score outer {} Classifier: {:.4f}+-{:.4f}'.format(name, 
                                                                      np.mean(perf_eval['score_o_early_stop']),
                                                                      np.std(perf_eval['score_o_early_stop'])
                                                                     ))
    if isNestedLoop:
        print('Mean(EarlyStop) outer VotingPrefit SOFT: {:.4f}+-{:.4f}'.format(np.mean(perf_eval['score_o_early_stop_vc_w0_soft']),
                                                                               np.std(perf_eval['score_o_early_stop_vc_w0_soft'])
                                                                              ))
        print('Mean(EarlyStop) outer VotingPrefit HARD: {:.4f}+-{:.4f}'.format(np.mean(perf_eval['score_o_early_stop_vc_w0_hard']),
                                                                               np.std(perf_eval['score_o_early_stop_vc_w0_hard'])
                                                                                ))


def print_perf_clf(name, perf_eval):
    print('Performance of the model:')    
    print('Mean(Val) score inner {} Classifier: {:.4f}+-{:.4f}'.format(name, 
                                                                      perf_eval['score_i_ave'],
                                                                      perf_eval['score_i_std']
                                                                     ))
    print('Min/max scores on folds: {:.4f} / {:.4f}'.format(np.min(perf_eval['score_i']),
                                                            np.max(perf_eval['score_i'])))
    print('OOF score inner {} Classifier: {:.4f}'.format(name, perf_eval['score_i_oof']))
    print('Scores in individual folds: {}'.format(['{:.4f}'.format(x) for x in perf_eval['score_i']]))
