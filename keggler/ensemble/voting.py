__all__=['VotingPrefitClassifier' 
        ]

from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import LabelEncoder



class VotingPrefitClassifier(VotingClassifier):
    '''
    The class implements the VotingClassifier with prefitted classifiers
    `X` is **NOT used**, so it can be fit on any data
    '''
    def fit(self, X, y, sample_weight=None, **fit_params):
        self.estimators_ = [x[1] for x in self.estimators]
        self.le_ = LabelEncoder().fit(y)
        self.classes_ = self.le_.classes_
        
        return self
