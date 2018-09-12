import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin

class FeatureEncoder(TransformerMixin):
    '''
    Calculates significance of diviation (or other diviation metrics) 
    from the mean of a numerical variable 
    within classes of categorical variables

    Parameters
    ----------
    enc_list : the list of categorical variables in which to do grouping
    var_list : the default list of numerical variables, for which
        diviations will be evaluated
    enc_dict : the dictionary of variables with non-default `var_list`.
        The keys should match the elements in `enc_list`
    tr_types : the list of transformations to be performed 
        (default: ['sign'], to calculate significance of diviation)            
    '''
    def __init__(self, enc_list, var_list, enc_dict, tr_types=['sign']):
        self.enc_list = enc_list
        self.var_list = var_list
        self.enc_dict = enc_dict
        self.tr_types = tr_types
        self.enc_map  = dict()
        super(FeatureEncoder, self).__init__()
            
    def transform(self, X, **transform_params):
        """Calculate the diviations

        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            A copy of the input data will be returned after transformation.
        """
        #X_ = X.copy(deep=True)
        X_ = X
        for f_enc in self.enc_list:
            var_list_ = self.enc_dict.get(f_enc, self.var_list)
            for f in var_list_:
                if 'sign' in self.tr_types:
                    X_['{}_ENC_{}_SIGN'.format(f_enc, f)] = (
                        (X_[f] - X_[f_enc].map(self.enc_map[f_enc]['{}_MEAN'.format(f)])) / 
                        X_[f_enc].map(self.enc_map[f_enc]['{}_STD'.format(f)])
                    )
        return X_
    
    def fit(self, X, y=None, **fit_params):
        """Compute the mean and std withing each categorical

        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
        """
        for f_enc in self.enc_list:
            var_list_ = self.enc_dict.get(f_enc, self.var_list)
            enc_agg = X.groupby(f_enc)[var_list_].agg(['mean', 'std'])
            enc_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in enc_agg.columns.tolist()])
            self.enc_map[f_enc] = enc_agg
        return self
