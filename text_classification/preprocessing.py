import nltk
from sklearn import preprocessing
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.stem.snowball import SnowballStemmer


def preprocess_data(train_data, vectorizer, use_svd):
    # Preprocess data in order to transform into BoW representation
    stopwords = ENGLISH_STOP_WORDS
    # Vectorization
    if vectorizer == 1:
        vc = CountVectorizer(stop_words=stopwords)
    elif vectorizer == 2:
        vc = HashingVectorizer(n_features=10**5, stop_words=stopwords)
    else:
        print('Wrong vectorizer parameter. Using CountVectorizer...')
        vc = CountVectorizer(stop_words=stopwords)

    X = vc.fit_transform(train_data['Content'] + ' ' + train_data['Title'])
    print(X.shape)

    if use_svd:
        # Apply SVD
        print('Applying Singular Value Decomposition ...')
        # using n_components = 100 as recommended by scikit learn
        svd = TruncatedSVD(n_components=100, random_state=0)
        X = svd.fit_transform(X)

    # Normalize data
    normalizer = preprocessing.Normalizer()
    X = normalizer.fit_transform(X)

    # Labels to integers encoding
    le = preprocessing.LabelEncoder()
    y = le.fit_transform(train_data["Label"])

    return X, y


# Extend CountVectorizer to support stemming
class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        stemmer = SnowballStemmer("english", ignore_stopwords=True)
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])


def preprocess_data_i(train_data, vectorizer, use_svd):
    # improved prepossessing
    stopwords = ENGLISH_STOP_WORDS
    # Vectorization
    if vectorizer == 1:
        vc = CountVectorizer(stop_words=stopwords, max_df=0.7)
        # vc = StemmedCountVectorizer(stop_words='english') vectorizer that performs stemming
    elif vectorizer == 2:
        vc = HashingVectorizer(n_features=200_000, stop_words=stopwords, ngram_range=(1, 2))
    else:
        print('Wrong vectorizer parameter. Using CountVectorizer...')
        vc = CountVectorizer(stop_words=stopwords)

    # Apply TF IDF
    X = vc.fit_transform(train_data['Content'] + ' ' + train_data['Title'] + ' ')
    transformer = TfidfTransformer()
    X = transformer.fit_transform(X)
    print(X.shape)

    if use_svd:
        # Apply SVD
        print('Applying Singular Value Decomposition ...')
        # using n_components = 100 as recommended by scikit learn
        svd = TruncatedSVD(n_components=100, random_state=0)
        X = svd.fit_transform(X)

    # Labels to integers encoding
    le = preprocessing.LabelEncoder()
    y = le.fit_transform(train_data["Label"])

    return X, y, vc, transformer


def preprocess_test_data(test_data, vc, transformer):
    # Use the fitted vectorizer (vc) and the fitted transformer from the training set
    X = vc.transform(test_data['Content'] + ' ' + test_data['Title'] + ' ')
    X = transformer.transform(X)
    print(X.shape)
    return X
