import pandas as pd
import timeit
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.svm import LinearSVC
from scipy.sparse import hstack, coo_matrix
from text_classification import cross_val as cv

def duplicate_detection(train_set_df, use_custom_feats):
    stopwords = ENGLISH_STOP_WORDS
    # Vectorization
    # vc = CountVectorizer(stop_words=stopwords, ngram_range=(1, 2))
    # vc = CountVectorizer(stop_words=stopwords)
    vc = HashingVectorizer(stop_words=stopwords, ngram_range=(1, 3))


    # fit on the combined questions
    X = vc.fit_transform(train_set_df['Question1'].astype('str') + ' ' + train_set_df['Question2'].astype('str'))
    transformer = TfidfTransformer()
    X = transformer.fit_transform(X)
    print(X.shape)


    q1_train = vc.transform(train_set_df['Question1'].astype('str'))
    q2_train = vc.transform(train_set_df['Question2'].astype('str'))
    q1_train = transformer.transform(q1_train)
    q2_train = transformer.transform(q2_train)

    if use_custom_feats:
        J_sim = coo_matrix(train_set_df['J_sim'].to_numpy().reshape(-1, 1))
        Fuzzy = coo_matrix(train_set_df['Fuzzy'].to_numpy().reshape(-1, 1))
        custom_feats = hstack((J_sim, Fuzzy))
        normalizer = preprocessing.Normalizer()
        norm_custom_feats = normalizer.fit_transform(custom_feats)
        final_train_set = hstack([q1_train, q2_train, norm_custom_feats])
    else:
        final_train_set = hstack((q1_train, q2_train))



    # Labels to integers encoding
    le = preprocessing.LabelEncoder()
    y = le.fit_transform(train_set_df["IsDuplicate"])

    print('Shape of training set: ', final_train_set.shape)
    clf = LinearSVC(random_state=0, tol=1e-5, C=0.1, loss='hinge', max_iter=100000, class_weight='balanced')
    cv.cross_val(clf, final_train_set, y, to_file=True, alg='duplicate_detection_svm_tfidf_hash_feats')


def duplicate_detection_pred(train_set_df, test_set_df, use_custom_feats):
    stopwords = ENGLISH_STOP_WORDS
    # Vectorization
    # vc = CountVectorizer(stop_words=stopwords, ngram_range=(1, 2))
    # vc = CountVectorizer(stop_words=stopwords)
    vc = HashingVectorizer(stop_words=stopwords, ngram_range=(1, 3))

    # fit on the combined questions
    X = vc.fit_transform(train_set_df['Question1'].astype('str') + ' ' + train_set_df['Question2'].astype('str'))
    transformer = TfidfTransformer()
    X = transformer.fit_transform(X)
    print(X.shape)

    q1_train = vc.transform(train_set_df['Question1'].astype('str'))
    q2_train = vc.transform(train_set_df['Question2'].astype('str'))
    q1_test = vc.transform(test_set_df['Question1'].astype('str'))
    q2_test = vc.transform(test_set_df['Question2'].astype('str'))

    q1_train = transformer.transform(q1_train)
    q2_train = transformer.transform(q2_train)
    q1_test = transformer.transform(q1_test)
    q2_test = transformer.transform(q2_test)

    if use_custom_feats:
        J_sim = coo_matrix(train_set_df['J_sim'].to_numpy().reshape(-1, 1))
        Fuzzy = coo_matrix(train_set_df['Fuzzy'].to_numpy().reshape(-1, 1))
        J_sim_test = coo_matrix(test_set_df['J_sim'].to_numpy().reshape(-1, 1))
        Fuzzy_test = coo_matrix(test_set_df['Fuzzy'].to_numpy().reshape(-1, 1))
        custom_feats = hstack((J_sim, Fuzzy))
        custom_feats_test = hstack((J_sim_test, Fuzzy_test))
        normalizer = preprocessing.Normalizer()
        norm_custom_feats = normalizer.fit_transform(custom_feats)
        norm_custom_feats_test = normalizer.fit_transform(custom_feats_test)
        final_train_set = hstack([q1_train, q2_train, norm_custom_feats])
        final_test_set = hstack([q1_test, q2_test, norm_custom_feats_test])
    else:
        final_train_set = hstack((q1_train, q2_train))
        final_test_set = hstack((q1_test, q2_test))

    # Labels to integers encoding
    # le = preprocessing.LabelEncoder()
    # y = le.fit_transform(train_set_df["IsDuplicate"])

    print('Shape of training set: ', final_train_set.shape)
    print('Shape of test set: ', final_test_set.shape)
    print('Fitting model...')
    clf = LinearSVC(random_state=0, tol=1e-5, C=0.1, loss='hinge', max_iter=100000, class_weight='balanced')
    start = timeit.default_timer()
    clf.fit(final_train_set, train_set_df['IsDuplicate'])
    end = timeit.default_timer()
    print('Time to fit:', end-start)
    pred = clf.predict(final_test_set)
    return pred


if __name__ == "__main__":

    # read train and test sets
    use_custom_feats = True
    if use_custom_feats:
        path = './train_features.csv'
        path_test = './test_features.csv'
    else :
        path = './train.csv'
        path_test = './test_without_labels.csv'
    train_set_df = pd.read_csv(path, sep=',')
    test_set_df = pd.read_csv(path_test, sep=',')
    train_set_df = train_set_df
    duplicate_detection(train_set_df, use_custom_feats)

    #generate predictions for kaggle
    #pred = duplicate_detection_pred(train_set_df, test_set_df, use_custom_feats)
    #pred_df = pd.DataFrame(data={"Predicted": pred}, index=test_set_df['Id'])
    #pred_df.to_csv('testSet_categories.csv')
