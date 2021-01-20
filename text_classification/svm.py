import pandas as pd
import preprocessing as pp
import cross_val as cv
import timeit
from sklearn.svm import LinearSVC


def svm_clf(train_set_df, vectorizer, use_svd):
    print("Preprocessing data...")
    start = timeit.default_timer()
    X,y = pp.preprocess_data(train_set_df, vectorizer, use_svd)
    end = timeit.default_timer()
    pp_time = end - start
    clf = LinearSVC(random_state=0, tol=1e-4, C=1, loss='hinge', max_iter=20000)
    print('Preprocessing time:', pp_time)
    cv.cross_val(clf, X, y, to_file=False, alg='SVM')


if __name__ == "__main__":

    # read train_set.csv
    path = '../train.csv'
    train_set_df = pd.read_csv(path, sep=',')
    svm_clf(train_set_df, vectorizer=1, use_svd=False)
    svm_clf(train_set_df, vectorizer=2, use_svd=False)
    svm_clf(train_set_df, vectorizer=1, use_svd=True)
    svm_clf(train_set_df, vectorizer=2, use_svd=True)


