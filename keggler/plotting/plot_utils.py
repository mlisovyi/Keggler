__all_ = ['plot_var_for2classes', 
          'plot3D_from_df', 'plot3D_basic',
          'display_importances',
          'plot_confusion_matrix'
         ]

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D # needed for 3D scatter plots
import itertools

def plot_var_for2classes(data, var_name, target_name='TARGET', xlim=None, bins=100, figsize=(15,6), normalise=True, lw=1.5, lc=('b', 'r')):
    '''
    Plot comparison of `data[var_name]` histograms for `data[target_name]` equals 0 and 1
    
    Parameters
    ----------
    data: pandas.DataFrame
        The dataframe to be used for plotting
    var_name: str
        The name of the variable to fill the histograms
    target_name: str
        The name of the target variable that will be used to split into two histograms
    xlim: tuple, None
        The X range to be used for selection and plotting.
        `None` can be used to use the full range automatically
    bins: int
        The number of bins in the histograms
    figsize: tuple
        The plot size
    normalise: bool
        Shall the two histograms be normalised to unity?
    lw: float
        The line width
    lc: tuple
        The line colours
    '''
    plt.figure(figsize=figsize)
    q_str = ''
    if xlim is not None:
        xmin,xmax = xlim
        q_str='{0} >= @xmin and {0} <= @xmax and '.format(var_name)
    for i in [0,1]:
        data.query('{} {} == {}'.format(q_str, target_name, i))[var_name].hist(bins=bins, 
                                                                                density=normalise, 
                                                                                histtype='step', 
                                                                                lw=lw, 
                                                                                label=str(i),
                                                                                color=lc[i])
    plt.legend()



def plot3D_from_df(X_, y_, x_str, y_str, z_str, figsize=(17,8), fout_name=None):
    '''
    Make a 3D scatter plot using columns from the X DataFrame.
    Set `y_=None` to get colour to represent Z axis value

    Parameters
    ----------
    X_ : pd.DataFrame, (n_rows, n_cols)
        The dataframe used for plotting
    y_ : pd.Series, (n_rows, 1) or None
        The y labels used to colour points.
        If `None`, use the z-axis values for colouring
    x_str, y_str, z_str : str
        The column names to be used to pick features for the X, Y and Z axis, respectively.
    figsize : tuple of floats, [default: (17,8)]
        The size of the figure
    fout_name : str or None [default: None]
        The name of the file to dump the figure. 
        If `None`, no file is created (to be used in notebooks)
    '''
    plot3D_basic(X_[x_str], X_[y_str], X_[z_str], y_, figsize, fout_name,
            x_str, y_str, z_str)


def plot3D_basic(x, y, z, c, figsize=(17,8), fout_name=None, 
        x_title='', y_title='', z_title=''):
    '''
    Make a 3D scatter plot using arrays.

    Parameters
    ----------
    x, y, z : array-like, (n_rows, 1)
        The values to be used for plotting
    c : array-like, (n_rows, 1) or None
        The y labels used to colour points.
        If `None`, use the z-axis values for colouring
    figsize : tuple of floats, [default: (17,8)]
        The size of the figure
    fout_name : str or None [default: None]
        The name of the file to dump the figure. 
        If `None`, no file is created (to be used in notebooks)
    '''
    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection='3d')
    p = ax.scatter(x, y, z, c=(c if c is not None else z))
    plt.colorbar(p, ax=ax)

    _ = ax.set_xlabel(x_title)
    _ = ax.set_ylabel(y_title)
    _ = ax.set_zlabel(z_title)

    if fout_name is not None:
        plt.savefig(fout_name)


####################################################################################

def display_importances(feature_importance_df_, n_feat=20, 
        silent=False, dump_strs=[], 
        fout_name=None, title='Features (avg over folds)'):
    '''
    Make a plot of most important features from a tree-based model

    Parameters
    ----------
    feature_importance_df_ : pd.DataFrame
        The input dataframe. 
        Must contain columns `'feature'` and `'importance'`.
        The dataframe will be first grouped by `'feature'` and the mean `'importance'` will be calculated.
        This allows to calculate and plot importance averaged over folds, 
        when the same features appear in the dataframe as many time as there are folds in CV.
    n_feats : int [default: 20]
        The maximum number of the top features to be plotted
    silent : bool [default: False]
        Dump additionsl information, in particular the mean importances for features 
        defined by `dump_strs` and the features with zero (<1e-3) importance
    dump_strs : list of strings [default: []]
        Features containing either of these srings will be printed to the screen
    fout_name : str or None [default: None]
        The name of the file to dump the figure. 
        If `None`, no file is created (to be used in notebooks)
    title : str
        The title to be assigned to the plot
    '''
    # Plot feature importances
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(
            by="importance", ascending=False)[:n_feat].index  
    
    mean_imp = feature_importance_df_[["feature", "importance"]].groupby("feature").mean()
    df_2_neglect = mean_imp[mean_imp['importance'] < 1e-3]
    
    if not silent:
        print('The list of features with 0 importance: ')
        print(df_2_neglect.index.values.tolist())

        pd.set_option('display.max_rows', 500)
        pd.set_option('display.max_columns', 500)
        for feat_prefix in dump_strs:
            feat_names = [x for x in mean_imp.index if feat_prefix in x]
            print(mean_imp.loc[feat_names].sort_values(by='importance', ascending=False))
    del mean_imp, df_2_neglect
    
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
    
    plt.figure(figsize=(8,10))
    sns.barplot(x="importance", y="feature", 
                data=best_features.sort_values(by="importance", ascending=False))
    plt.title(title)
    plt.tight_layout()

    if fout_name is not None:
        plt.savefig(fout_name)

        
###################################################################################

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    
    np.set_printoptions(precision=2)
    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)#, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
