__all__=[
         'train_model_in_nestedCV', 'print_nested_perf_clf'
        ]

from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
import pandas as pd
import numpy as np
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



def train_model_in_nestedCV(model, X, y, metric, metric_args={},
                            model_name='xmodel',
                            inner_seed=31416, inner_n=10, outer_seed=314, outer_n=10,
                            opt_parameters_={}, fit_params_={},
                            verbose=True):
    '''
    Single-liner to do performance evaluation of multi-class `model` in a nested stratified CV.
    In addition to the base model, also Voting Classifiers are evaluated in the outer CV loop.
    
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
