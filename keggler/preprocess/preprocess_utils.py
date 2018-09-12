import numpy as np
import pandas as pd
from  itertools import combinations

def join_permuted_text_cols(df_, f_list):
    '''
    Computes interactions features as a joint of string-type inputs.
    Modifies the input pandas.DataFrame

    Parameters
    -----------
    df_ : pd.DataFrame
        Input DataFrame. New features will be added to it.
    f_list : list of strings
        The list of columns from the DataFrame,
        that will be used to build interactions.
        Missing values will be replaced with `'XNA'`
        These as well as added columns will be casted into `Category` type.
    '''
    df_[f_list] = df_[f_list].astype('object').fillna('XNA')    
    for L in range(2, len(f_list)+1):
        for cc in combinations(f_list, L):
            print(cc)
            df_['__'.join(cc)] = df_[list(cc)].apply(lambda x: '__'.join(x), axis=1).astype('category')
    df_[f_list] = df_[f_list].astype('category')



def convert_OHE2LE(df, cat_cols=[]):
    '''
    Convert One-Hot/Binary Encoding(OHE/BE) into an integer-label encoding.
    The function will check that OHE is complete, i.e. that there are no rows with all zeros.
    If such OHE group is discovered, an additional column will be added 
    to represent such missing dummy category before label encoding.

    Parameters
    ----------
    df : pd.DataFrame
        The datafarme to process.
    cat_cols : list of strings [default: []]
        The list of strings with which category names start.

    Returns
    --------
    tmp_df : pd.DataFrame
        A deep copy of the input dataframe with requested OHE columns 
        replaced with LE columns with `'_LE'` prefix.
    '''
    tmp_df = df.copy(deep=True)
    for s_ in cat_cols:
        cols_s_ = [f_ for f_ in df.columns if f_.startswith(s_)]
        sum_ohe = tmp_df[cols_s_].sum(axis=1).unique()
        #deal with those OHE, where there is a sum over columns == 0
        if 0 in sum_ohe:
            print('The OHE in {} is incomplete. A new column will be added before label encoding'
                  .format(s_))
            # dummy colmn name to be added
            col_dummy = s_+'_dummy'
            # add the column to the dataframe
            tmp_df[col_dummy] = (tmp_df[cols_s_].sum(axis=1) == 0).astype(np.int8)
            # add the name to the list of columns to be label-encoded
            cols_s_.append(col_dummy)
            # proof-check, that now the category is complete
            sum_ohe = tmp_df[cols_s_].sum(axis=1).unique()
            if 0 in sum_ohe:
                 print("The category completion did not work")
        tmp_df[s_ + '_LE'] = tmp_df[cols_s_].idxmax(axis=1).str.replace(s_,'').astype(np.uint16)
        tmp_df.drop(cols_s_, axis=1, inplace=True)
    return tmp_df



def train_test_apply_func(train_, test_, func_, **kwargs):
    '''
    Apply `func_` function on a dataset that is created by joining `train_` and `test_` dataframes

    Parameters
    ----------
    train_ : pd.DataFrame, (n_rows1, n_cols)
        A dataset, that typcally will be the train dataset
    test_ : pd.DataFrame, (n_rows2, n_cols)
        A dataset, that typically will be the test dataset
    func_ : function
        A function with interface `func_(pd.DataFrame, **kwargs)`, which in turn returns a pd.DataFrame
    kwargs : additional arguments to be passed to `func_`
    '''
    xx = pd.concat([train_, test_])
    xx_func = func_(xx)
    train_ = xx_func.iloc[:train_.shape[0], :]
    test_  = xx_func.iloc[train_.shape[0]:, :]

    del xx, xx_func
    return train_, test_
