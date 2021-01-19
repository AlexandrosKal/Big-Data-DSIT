import LSHrandom_projection as lsh_rp
import pandas as pd
from datasketch import MinHash, MinHashLSH
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
#from stop_words import get_stop_words
import timeit
from scipy.sparse import csr_matrix

def make_shingles(doc, k=5):
    """
    function that performs
    the shingling of a document creating k-shingles
    """
    shingles = []
    for i in range(0, len(doc) - k):
        shingles.append(doc[i:i + k])
    return shingles


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


def exact_Jaccard(shingles_train, shingles_test, threshold):
    """
    :return: returns Jaccard similarity
    """

    duplicates = 0

    for index, row in shingles_train.iterrows():

        for i_test, row_test in shingles_test.iterrows():
            Intersection = Intersect_list(shingles_train['Shingles'][index], shingles_test['Shingles'][i_test])
            Union = shingles_train['Shingles'][index] + shingles_test['Shingles'][i_test]
            Jac_sim = len(Intersection) / len(Union)

            if Jac_sim >= threshold:
                duplicates += 1

    return duplicates


def LSH_minhash(t, permutations, shingles_train):
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

    return lsh


def query(shingles_test, lsh):
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
    return duplicates


def exact_cosine(train_data, test_data, threshold, out):

    similarity_matrix = cosine_similarity( test_data, train_data, dense_output=False)
    row_data, col_data = similarity_matrix.get_shape()

    print("exact cosine\n",file=out)
    starttime = timeit.default_timer()

    duplicates = 0
    for i in range(row_data):
        row = similarity_matrix.getrow(i)
        row = row.toarray()

        if np.any(row >= threshold):
            duplicates += 1
    endtime = timeit.default_timer()

    print("exact cosine execution time",endtime-starttime,file=out)
    print("\nexact cosine duplicates",duplicates,"\n",file=out)

    print(duplicates)


def lsh_cosine(train_v,test_v,k,threshold,out):

    l=2
    starttime = timeit.default_timer()
    print("\tlsh random projection build time \n",file=out)
    #build
    row_data, dimensions = train_v.get_shape()
    lsh = lsh_rp.LSH_RP(l, k, dimensions)
    for i in range(3):
        row = train_v.getrow(i)
        row = np.float32(row.toarray())
        lsh[row[0]] = row[0]

    buildtime = timeit.default_timer()
    print("\tlsh random projection build time is:", buildtime-starttime,file=out)

    print("\n\tlsh random projection query",file=out)

    print("build finished")

    starttime = timeit.default_timer()

    #query
    row_data, dimensions = test_v.get_shape()
    duplicates=0
    for i in range(row_data):
        row = train_v.getrow(i)
        row = np.float32(row.toarray())
        same_bin_data = lsh[row[0]]

        similarity = cosine_similarity(row, same_bin_data, dense_output=False)
        if similarity.max() >= threshold:
            duplicates += 1

    querytime = timeit.default_timer()
    print("\n\tlsh random projection query time is:", querytime-starttime,"\n",file=out)
    print("\n\tlsh cosine duplicates",duplicates,"\n",file=out)

    print(duplicates)




if __name__ == '__main__':
    # loading the training set data
    with open('output.txt','w') as out:

        train_data = pd.read_csv("./corpusTrain.csv", sep=',')
        test_data = pd.read_csv("./corpusTest.csv", sep=',')

        # 1st step:
        # We now have to compute the shingles of each of our documents

        #shingles_train = store_shingles(train_data, 4)
        #shingles_test = store_shingles(test_data, 4)


        #stop_words = get_stop_words('english')
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
        #result = exact_Jaccard(shingles_train,shingles_test,0.80)
        #print(result)

        # LSH family minhash case

        # Signature matrix initialization
        """permutations = 16
        lsh = LSH_minhash(0.80, permutations, shingles_train)
        query(shingles_test,lsh)"""

        # exact cosine
        #exact_cosine(train_v, test_v, 0.8, out)

        #lsh cosine
        k=list(range(1,11))
        for i in k:
            print("lsh random projection k = ",i,file=out)
            lsh_cosine(train_v,test_v,i,0.8,out)
