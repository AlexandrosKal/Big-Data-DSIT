import pandas as pd
import preprocessing as pp
import cross_val as cv
import timeit
from sklearn.svm import LinearSVC


def svm_clf_i(train_set_df, vectorizer, use_svd):
    # SVM with improved preprocessing
    print("Preprocessing data...")
    start = timeit.default_timer()
    X, y, _, _ = pp.preprocess_data_i(train_set_df, vectorizer, use_svd)
    end = timeit.default_timer()
    pp_time = end - start
    print('Preprocessing time:', pp_time)
    clf = LinearSVC(random_state=0, tol=1e-4, C=1, loss='hinge', max_iter=20000, class_weight='balanced')
    cv.cross_val(clf, X, y, to_file=False, alg='SVM')


def svm_clf_i_pred(train_set_df, test_set_df, vectorizer, use_svd):
    # SVM with improved preprocessing used to predict on a specific test set
    print("Preprocessing data...")
    start = timeit.default_timer()
    X, y, vc, transformer = pp.preprocess_data_i(train_set_df, vectorizer, use_svd)
    X_test = pp.preprocess_test_data(test_set_df, vc, transformer)
    end = timeit.default_timer()
    pp_time = end - start
    print('Preprocessing time:', pp_time)
    clf = LinearSVC(random_state=0, tol=1e-4, C=1, loss='hinge', max_iter=20000, class_weight='balanced')
    print('Fitting model...')
    start = timeit.default_timer()
    clf.fit(X, y)
    end = timeit.default_timer()
    print('Time to fit:', end-start)
    y_pred =  clf.predict(X_test)
    return y_pred


if __name__ == "__main__":

    # read train and test sets
    path = '../train.csv'
    path_test = '../test_without_labels.csv'
    train_set_df = pd.read_csv(path, sep=',')
    test_set_df = pd.read_csv(path_test, sep=',')
    svm_clf_i(train_set_df, vectorizer=1, use_svd=False)
    svm_clf_i(train_set_df, vectorizer=2, use_svd=False)

    # create the csv for kaggle submission

    # le = preprocessing.LabelEncoder()
    # le.fit_transform(train_set_df["Label"])
    # y_pred = svm_clf_i_pred(train_set_df, test_set_df, vectorizer=2, use_svd=False)
    # prediction = pd.DataFrame(data={"Predicted": le.inverse_transform(y_pred)}, index=test_set_df['Id'])
    # prediction.to_csv('testSet_categories.csv')


