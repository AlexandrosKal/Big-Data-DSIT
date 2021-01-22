import LSHrandom_projection as lsh_rp
import pandas as pd
from datasketch import MinHash, MinHashLSH
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
# from stop_words import get_stop_words
import timeit
from scipy.sparse import coo_matrix, vstack
import collections


def make_words(doc):
    wordList = doc.split()
    return set(wordList)


def store_words(data):
    # dataframe containing all the shingles
    temp_dict = {'Words': []}  # for faster implementation

    # words for our train data
    for index, row in data.iterrows():
        new_word = make_words(row['Content'])
        temp_dict['Words'].append(new_word)

    return pd.DataFrame(temp_dict)


def make_shingles(doc, k=5):
    """
    function that performs
    the shingling of a document creating k-shingles
    """
    shingles = []
    for i in range(0, len(doc) - k):
        shingles.append(doc[i:i + k])
    return set(shingles)


def store_shingles(data, k=5):
    # dataframe containing all the shingles
    temp_dict = {'Shingles': []}  # for faster implementation

    # shingles for our train data
    for index, row in data.iterrows():
        new_shingle = make_shingles(row['Content'], k)
        temp_dict['Shingles'].append(new_shingle)

    return pd.DataFrame(temp_dict)


def Intersect_list(L1, L2):
    """Efficient intersection implementation"""
    # instead of turning all lists into sets and using & operator
    # we turn the largest list to a set and then search for duplicates
    # in the other one

    if len(L1) > len(L2):
        L = L2
        temp_set = set(L1)
    else:
        L = L1
        temp_set = set(L2)

    return [val for val in L if val in temp_set]


def exact_Jaccard(train, test, threshold):
    """
    :return: returns Jaccard similarity duplicates
    """
    starttime = timeit.default_timer()
    print("\tlsh exact Jaccard build time \n", file=out, flush=True)
    duplicates = 0

    for st1 in test['Words']:

        for st2 in train['Words']:

            inter = len(st1 & st2)
            union = len(st1) + len(st2) - inter

            if union > 0:
                jaccard = inter / union
            else:
                continue
            if jaccard >= threshold:
                duplicates += 1

    querytime = timeit.default_timer()
    print("\n\texact jaccard query time is:", querytime - starttime, "\n", file=out, flush=True)
    print("\n\texact jaccard query time is:", querytime - starttime, "\n", flush=True)

    print("\n\texact jaccard duplicates", duplicates, "\n", file=out, flush=True)
    print("\nlove", duplicates, flush=True)
    return duplicates


def LSH_minhash(t, permutations, shingles_train, out):
    """building the lsh minhash structure"""
    starttime = timeit.default_timer()
    print("lsh minhash permutations \n", permutations, file=out, flush=True)

    print("\tlsh minhash build time \n", file=out, flush=True)

    # bands aka LSH hash functions
    b = 8
    # rows -> rows * bands  constant
    r = int(permutations / b)

    lsh = MinHashLSH(threshold=t, num_perm=permutations, params=(b, r))

    # creating our signature matrix for the train data
    for index, row in shingles_train.iterrows():
        Signature_matrix = MinHash(num_perm=permutations)

        for shingle in set(shingles_train['Shingles'][index]):
            Signature_matrix.update(shingle.encode('utf8'))

        lsh.insert("signature" + str(index), Signature_matrix)

    buildtime = timeit.default_timer()

    print("\tlsh minhash build time is:", buildtime - starttime, file=out, flush=True)
    print("build", buildtime - starttime, flush=True)

    return lsh


def query(shingles_test, permutations, lsh, out):
    """query for the lsh minhash case"""
    starttime = timeit.default_timer()

    print("\tlsh minhash query time \n", file=out, flush=True)
    duplicates = 0
    # creating our signature matrix for the test data
    for index, row in shingles_test.iterrows():
        Signature_matrix_test = MinHash(num_perm=permutations)
        result = []

        for shingle in set(shingles_test['Shingles'][index]):
            Signature_matrix_test.update(shingle.encode('utf8'))

        result = lsh.query(Signature_matrix_test)
        if result:
            duplicates += 1

    endtime = timeit.default_timer()
    print("\tlsh minhash query execution time", endtime - starttime, file=out, flush=True)
    print("\n\tlsh minhash duplicates", duplicates, "\n", file=out)
    return duplicates


def exact_cosine(train_data, test_data, threshold, out):
    similarity_matrix = cosine_similarity(test_data, train_data, dense_output=False)
    row_data, col_data = similarity_matrix.get_shape()

    print("exact cosine\n", file=out)
    starttime = timeit.default_timer()

    duplicates = 0
    for i in range(row_data):
        row = similarity_matrix.getrow(i)
        row = row.toarray()

        if np.any(row >= threshold):
            duplicates += 1
    endtime = timeit.default_timer()

    print("exact cosine execution time", endtime - starttime, file=out)
    print("\nexact cosine duplicates", duplicates, "\n", file=out)

    print(duplicates)


def lsh_cosine(train_v, test_v, k, threshold, out):
    l = 1
    starttime = timeit.default_timer()
    print("\tlsh random projection build time \n", file=out, flush=True)

    # build
    row_data, dimensions = train_v.get_shape()
    lsh = []
    for i in range(l):
        lsh.append(lsh_rp.Hashtable(k, dimensions))

    hashvectorsall = collections.defaultdict()  # sparse arrays of vectors in the bucket

    for row in train_v:
        for i in range(l):
            # row = train_v.getrow(t)
            lsh[i][row] = row

    for i in range(l):
        for hash in lsh[i].hashtable:
            hashvectorsall[hash] = vstack(lsh[i].hashtable[hash])

    buildtime = timeit.default_timer()
    print("\tlsh random projection build time is:", buildtime - starttime, file=out, flush=True)
    print("time", buildtime - starttime, flush=True)
    print("\n\tlsh random projection query", file=out, flush=True)

    print("build finished\n", flush=True)

    starttime = timeit.default_timer()

    # query
    duplicates = 0
    r, d = test_v.get_shape()
    keep_track = np.zeros(r)

    c = 0
    for row in test_v:
        c += 1
        for j in range(l):

            # taking straight from each hashtable the vectors belonging to the same bucket
            row_hash = lsh[j].hashing(row)

            if row_hash not in hashvectorsall:
                continue

            same_bin_data = hashvectorsall[row_hash]

            # check if buckets are empty
            similarity = cosine_similarity(row, same_bin_data, dense_output=False)
            if similarity.max() >= threshold:
                duplicates += 1
                break

    querytime = timeit.default_timer()
    print("time query finished\n", querytime - starttime, flush=True)

    print("\n\tlsh random projection query time is:", querytime - starttime, "\n", file=out, flush=True)
    print("\n\tlsh cosine duplicates", duplicates, "\n", file=out, flush=True)
    print("\nlove", duplicates, flush=True)


if __name__ == '__main__':
    # loading the training set data
    with open('output.txt', 'w') as out:
        train_data = pd.read_csv("./corpusTrain.csv", sep=',')
        test_data = pd.read_csv("./corpusTest.csv", sep=',')

        # We now have to compute the shingles and the words of each of our documents
        words_train = store_words(train_data)
        words_test = store_words(test_data)
        shingles_train = store_shingles(train_data, 4)
        shingles_test = store_shingles(test_data, 4)


        #preprocessing for the cosine cases
        # stop_words = get_stop_words('english')
        stop_words = stopwords.words('english')
        tfidf = TfidfVectorizer(stop_words=stop_words)

        # vectors used for cosine cases
        train_v = tfidf.fit_transform(train_data["Content"])
        test_v = tfidf.transform(test_data["Content"])
        """
        for i in range(len(shingles_test)):
            if shingles_test["Shingles"][i]:
                np.char.lower(shingles_test["Shingles"][i])

        for i in range(len(shingles_train)):
           if shingles_train["Shingles"][i]:
                np.char.lower(shingles_train["Shingles"][i])
        """

        # Jaccard similarity duplicates train test exact case next
        #result = exact_Jaccard(words_train, words_test, 0.80)
        #print("Jaccard duplicates:", result)

        # LSH family minhash case
        permutations = [16, 32, 64]
        for p in permutations:
            lsh = LSH_minhash(0.80, p, shingles_train, out)
            query(shingles_test, p, lsh, out)

        # exact cosine
        exact_cosine(train_v, test_v, 0.8, out)

        # LSH random proection family cosine case
        k = list(range(1, 11))
        for i in k:
            print("lsh random projection k = ", i, file=out,flush=True)
            lsh_cosine(train_v, test_v, i, 0.80, out)
