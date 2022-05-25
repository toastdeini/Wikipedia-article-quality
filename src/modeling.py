import numpy as np
import seaborn as sns
from sklearn.model_selection import cross_validate, cross_val_score

# 1.) Model scoring class

class ModelForScoring():
    '''
    Class structure to save a model and print out scores.
    
    Lifted and modified from Flatiron DS Live lecture #51, "Workflow With Pipelines."
    '''
    
    def __init__(self, model, model_name, X, y, cv_now='simple'):
        self.model = model
        self.name = model_name
        self.X = X
        self.y = y
        
        # for CV results
        self.cv_results = None
        self.cv_mean = None
        self.cv_median = None
        self.cv_std = None
        
        # Cross-validate now?
        if cv_now == 'simple':
            self.cv_simple()
        elif cv_now == 'multi':
            self.cv_multi()
        else:
            pass
           
            
    def cv_simple(self, X=None, y=None, kfolds=5):
        '''
        Simple results of cross-validation.
        
        X, y: Optional, training data. Otherwise use X, y from object
        kfolds: Optional, # of folds for CV. Default is 5, bump up to 10 if necessary
        '''
        
        cv_X = X if X else self.X
        cv_y = y if y else self.y
        
        self.cv_results = cross_val_score(self.model, cv_X, cv_y, cv=kfolds)
        self.cv_mean = np.mean(self.cv_results)
        self.cv_median = np.median(self.cv_results)
        self.cv_std = np.std(self.cv_results)
        
        
    def cv_multi(self, X=None, y=None, s_metrics=None, kfolds=5, verbose=1):
        '''
        Multi-metric results of cross-validation,
        with training scores for comparison.
        
        Parameters
        ----------
        X, y : optional
            Training data - use X, y from
            object if not specified.
        s_metrics : string or list-like
            Scoring metrics to use. If none,
            use accuracy and F1 macro.
        kfolds : int, optional
            Optional, # of folds for cross-validation.
            Default is 5.
        verbose : int, optional
            Display cross-validation time. Set to 0
            for cleaner print.
        '''
        
        cv_X = X if X else self.X
        cv_y = y if y else self.y
        scoring_metrics = s_metrics if s_metrics else ['accuracy', 'f1_macro']
        
        self.cv_results = cross_validate(estimator=self.model,
                                         X=cv_X,
                                         y=cv_y,
                                         scoring=scoring_metrics,
                                         cv=kfolds,
                                         verbose=verbose,
                                         return_train_score=True)
        
        if 'accuracy' in scoring_metrics:
            self.cv_mean = np.mean(self.cv_results['test_accuracy'])
            self.cv_median = np.median(self.cv_results['test_accuracy'])
            self.cv_std = np.std(self.cv_results['test_accuracy'])
        else:
            pass
        
                                         
# 2.) Global scoring function

def pretty_cv(result):
    '''
    Generic function to print 'pretty' results from the results of multi-metric `cross_validate`.
    
    Can only be used when `scoring` set to `accuracy` and `f1_macro`,
    and `return_train_score` set to True.
    '''
    if 'test_accuracy' and 'test_f1_macro' in result.keys():
        print('CV Results')
        print('='*32)
        print('Accuracy')
        print('-'*32)
        print(f"Training accuracy: {result['train_accuracy'].mean():.3f}")
        print(f"Test accuracy:     {result['test_accuracy'].mean():.3f}")
        print('F-1 Score')
        print('-'*32)
        print(f"Training F1 score: {result['train_f1_macro'].mean():.3f}")
        print(f"Test F1 score:     {result['test_f1_macro'].mean():.3f}")
    else:
        print("Error: Metrics do not match.")