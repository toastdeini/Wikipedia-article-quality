import pandas as pd
import numpy as np

def pretty_cv(results):
    '''
    Generic function to print 'pretty' results from the results of `cross_validate`.
    
    :param results: Output of the sklearn `cross_validate` method.
    :return: Printout of accuracy and F1 scores on both training and validation sets.
    '''
    print('CV Results')
    print('='*32)
    print('Accuracy')
    print('-'*32)
    print(f"Training accuracy: {cv_metric['train_accuracy'].mean():.3f}")
    print(f"Test accuracy:     {cv_metric['test_accuracy'].mean():.3f}")
    print('F-1 Score')
    print('-'*32)
    print(f"Training F1 score: {cv_metric['train_f1_macro'].mean():.3f}")
    print(f"Test F1 score:     {cv_metric['test_f1_macro'].mean():.3f}")