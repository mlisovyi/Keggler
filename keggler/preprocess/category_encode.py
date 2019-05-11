import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.base import TransformerMixin

class TargetEncoder_Base(TransformerMixin):
    '''
    Base class of target/entity/frequency encoding
    Implements a simple frequency encoding on the full sample without regularisation.
    So far only binary classification is implemented.
    
    Parameters
    ----------
    cat_cols: list of strings, None [default: None]
        The list of columns to encode.
        If `None` encode all `object` and `category` columns
    y_name: string [default: TARGET]
        The name of the feature holding the target
    tr_type: string [default: basic]
        The name of transformation to use for encoding. 
        Only the simple frequency (`basic`) is implemented so far.
    random_state: int [default: 0]
        Random seed.
        Is used for
        
    '''
    def __init__(self, cat_cols=None, y_name='TARGET', tr_type='basic', random_state=0, prefix='ENC', prefix_sep='_'):
        self.cat_cols = cat_cols
        self.gb       = dict()
        self.y_name   = y_name
        self.tr_type = tr_type
        self.random_state = random_state
        self.prefix   = '{}{}'.format(prefix, prefix_sep)
        self.prior    = -1
        super(TargetEncoder_Base, self).__init__()
            
    def transform(self, X, **transform_params):
        X_ = X.copy(deep=True)
        for f_ in self.cat_cols:
            X_ [self.prefix + f_] = X_[f_].map(self.gb[f_]).astype(np.float32)
            X_ [self.prefix + f_].fillna(self.prior, inplace=True)
            del X_[f_]
        return X_
    
    def fit(self, X, y=None, **fit_params):
        self._prefit(X, y)
                
        #concatenate X and y to simplify usage (temporary object)
        XY = self._getXY(X, y)
        
        if self.tr_type == 'basic':
            # learn encodings from the full sample
            for f_ in self.cat_cols:
                self.gb[f_] = XY.groupby(f_)[self.y_name].mean()
        else:
            raise ValueError('Unknown value tr_type = {}'.format(self.tr_type))
        
        del XY   
        return self
    
    def _prefit(self, X, y=None):
        if y is None:
            raise RuntimeError('TargetEncoder_KFold needs y to learn the transform')
            
        # deduce categorical columns, if user did not speficy anything
        if self.cat_cols == None:
            self.cat_cols = X.select_dtypes(include=['category', 'object']).columns.tolist()
        # make sure that we store the list of categorical columns as a list
        if not isinstance(self.cat_cols, list):
            try:
                self.cat_cols = self.cat_cols.tolist()
            except:
                RuntimeError('TargetEncoder_KFold fails to convert `cat_cols` into a list')
                
        #store the full sample mean for encoding of rare categories
        self.prior = y.mean()
        
    def _getXY(self, X, y):
        return pd.concat([X[self.cat_cols], y], axis=1)
    
    

class TargetEncoder_KFold(TargetEncoder_Base):
    def __init__(self, cv=5, **kwargs):
        super(TargetEncoder_KFold, self).__init__(**kwargs)
        self.cv       = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
    
    def regularise(self, X, y=None, **transform_params):
        # a dataframe to store OOF target encodings
        oof = pd.DataFrame(np.zeros(shape=(X.shape[0], len(self.cat_cols))),
                           index=X.index,
                           columns=self.cat_cols)
        #concatenate X and y to simplify usage (temporary object)
        XY = self._getXY(X, y)
        
        # iterate over folds
        for trn_idx, val_idx in self.cv.split(X, y):
            trn = XY.iloc[trn_idx]
            val = XY.iloc[val_idx]
            # iterate over categorical features
            for f_ in self.cat_cols:
                # get target means for each class within category
                te = trn.groupby(f_)[self.y_name].mean()
                # encode the OOF partion
                oof.iloc[val_idx, oof.columns.get_loc(f_)] = val[f_].map(te).astype(np.float32)
        # do finla cosmetics and fill NAN
        oof = oof.add_prefix(self.prefix).fillna(self.prior)
        del XY
        
        X_ = X.drop(self.cat_cols, axis=1)
        return pd.concat([X_, oof], axis=1)



class TargetEncoder_ExpandingMean(TargetEncoder_Base):
    def __init__(self, **kwargs):
        super(TargetEncoder_ExpandingMean, self).__init__(**kwargs)
    
    def regularise(self, X, y=None, **transform_params):
        X_ = X.copy(deep=True)
        
        # iterate over categorical features
        for f_ in self.cat_cols:
            gb = self._getXY(X_, y).groupby(f_)[self.y_name]
            # calculate expanding mean
            X_[self.prefix + f_] = ((gb.cumsum() - y) / gb.cumcount()).astype(np.float32)
            del gb
        
        X_.drop(self.cat_cols, axis=1, inplace=True)
        return X_



# use this if you want to convert categorical features to dummies
def cat_to_dummy(train, test, target_name='TARGET'):
    train_d = pd.get_dummies(train, drop_first=False)
    test_d = pd.get_dummies(test, drop_first=False)
    # make sure that the number of features in train and test should be same
    for i in train_d.columns:
        if i not in test_d.columns:
            if i != target_name:
                train_d = train_d.drop(i, axis=1)
    for j in test_d.columns:
        if j not in train_d.columns:
            if j != target_name:
                test_d = test_d.drop(j, axis=1)
    print('Memory usage of train increases from {:.2f} to {:.2f} MB'.format(train.memory_usage().sum() / 1024**2, 
                                                                            train_d.memory_usage().sum() / 1024**2))
    print('Memory usage of test increases from {:.2f} to {:.2f} MB'.format(test.memory_usage().sum() / 1024**2, 
                                                                            test_d.memory_usage().sum() / 1024**2))
    return train_d, test_d



