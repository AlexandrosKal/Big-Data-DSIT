import numpy as np
from sklearn.model_selection import cross_validate


def cross_val(clf, X, y, to_file, alg, k=5):
    # perform k fold cross validation and print the requested metrics
    # if to_file = True write the requested metrics to a file
    print("Attempting 5-fold cross validation...")

    scoring = {
        'acc': 'accuracy',
        'prec_macro': 'precision_macro',
        'rec_macro': 'recall_macro',
        'f1_macro': 'f1_macro'

    }
    scores = cross_validate(clf, X, y, cv=k, scoring=scoring, return_train_score=False, n_jobs=-1)
    print('Accuracy:', np.mean(scores['test_acc']), scores['test_acc'])
    print('Precision:', np.mean(scores['test_prec_macro']), scores['test_prec_macro'])
    print('Recall:', np.mean(scores['test_rec_macro']), scores['test_rec_macro'])
    print('F-Measure:', np.mean(scores['test_f1_macro']), scores['test_f1_macro'])
    print('Fit-Time:', np.mean(scores['fit_time']), scores['fit_time'])
    if to_file:
        f = open(alg+'_metrics', 'w')
        print('Accuracy:', np.mean(scores['test_acc']), scores['test_acc'], file=f)
        print('Precision:', np.mean(scores['test_prec_macro']), scores['test_prec_macro'], file=f)
        print('Recall:', np.mean(scores['test_rec_macro']), scores['test_rec_macro'], file=f)
        print('F-Measure:', np.mean(scores['test_f1_macro']), scores['test_f1_macro'], file=f)
        print('Fit-Time:', np.mean(scores['fit_time']), scores['fit_time'], file=f)
        f.close()
