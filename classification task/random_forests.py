import pandas as pd
import preprocessing as pp
import cross_val as cv
import timeit
from sklearn.ensemble import RandomForestClassifier


def random_forest(train_set_df, vectorizer, use_svd):
    print("Preprocessing data...")
    start = timeit.default_timer()
    X,y = pp.preprocess_data(train_set_df, vectorizer, use_svd)
    end = timeit.default_timer()
    pp_time = end - start
    clf = RandomForestClassifier(random_state=0, n_jobs=-1)
    print('Preprocessing time:', pp_time)
    cv.cross_val(clf, X, y, to_file=False, alg='Random_forest')


if __name__ == "__main__":

    # read train_set.csv
    path = '../train.csv'
    train_set_df = pd.read_csv(path, sep=',')
    random_forest(train_set_df, vectorizer=1, use_svd=False)
    random_forest(train_set_df, vectorizer=2, use_svd=False)
    random_forest(train_set_df, vectorizer=1, use_svd=True)
    random_forest(train_set_df, vectorizer=2, use_svd=True)


