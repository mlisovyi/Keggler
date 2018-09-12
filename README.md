# Keggler

A data science toolkit for kaggle and beyond

## Objective

The package is a collection of tools developed in various kaggle competitions.
It implements various common pre-processing steps as well as 
additional high-level models like stacking and voting ensembles.

## HowTo

The documentation of various tools is done directly in the code. 
Later on it will be propagate to `readthedocs` or something alike.
Only a list of tools is provided below.

- **Preprocessing**:
 - `FeatureEncoder`: evaluate diviation of a numeric feature 
   with respect to the mean within a class of a categorical feature.
 - `TargetEncoder_Base`, `TargetEncoder_KFold`, `TargetEncoder_ExpandingMean`:
   target (=entity) frequiency encoding either with basic method or with KFold or expanding mean regularisation.
   The implementation follows XXX.
 - `cat_to_dummy`: do dummy encoding on training and testing set droping those OHE columns, 
   that are unique to either.
 - `convert_OHE2LE`: convert OHE to LE.
 - `train_test_apply_func`: apply a function to a merge of training and testing samples.

- **Data I/O**:
 - `import_data`: read csv/h5 files and reduce memory footprint using 
  `reduce_mem_usage` in the case of csv.
 - `reduce_mem_usage`: minimise memory footprint based on feature values. 
  Non-numeric features will be turned into `category`.

- **Feature engineering**:
 - `join_permuted_text_cols`: make interactions of string features as a joint.

- **Ensemble _classification_ models**:
 - `StackingClassifier`: a stacker that builds an arbitrary model on pre-computed OOF and test predictions.
   Various data manipulations are available. 
   _Developed for binary classification only._
 - `VotingPrefitClassifier`: a voting (multi-class) classifier that operates on pre-computed models,
   instead of fiting them as sklearn implementation does.
 - further tools, that still have to be described.
