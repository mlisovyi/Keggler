import pandas as pd
import numpy as np

def reduce_mem_usage(df, force_obj_in_category=True, debug=True):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage. 
        This function originates from https://www.kaggle.com/gemartin/load-data-reduce-memory-usage
    """
    if debug:
        start_mem = df.memory_usage(deep=True).sum() / 1024**2
        print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object and df[col].dtype.name != 'category' and 'datetime' not in col_type.name:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                for i_type in [np.int8, np.int16, np.int32, np.int64]:
                    if c_min > np.iinfo(i_type).min and c_max < np.iinfo(i_type).max:
                        df[col] = df[col].astype(i_type)
                        break
            elif str(col_type)[:4] == 'uint':
                for i_type in [np.uint8, np.uint16, np.uint32, np.uint64]:
                    if c_max < np.iinfo(i_type).max:
                        df[col] = df[col].astype(i_type)
                        break
            elif col_type == bool:
                df[col] = df[col].astype(np.uint8)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        elif force_obj_in_category and 'datetime' not in col_type.name:
            df[col] = df[col].astype('category')

    if debug:
        end_mem = df.memory_usage(deep=True).sum() / 1024**2
        print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
        print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df


def import_data(file, key=''):
    """create a dataframe and optimize its memory usage"""
    if '.csv' in file:
        df = pd.read_csv(file, parse_dates=True, keep_date_col=True)#, nrows=10000)
        df = reduce_mem_usage(df)
    elif '.h5' in file:
        df = pd.read_hdf(file, key=key)
    return df
