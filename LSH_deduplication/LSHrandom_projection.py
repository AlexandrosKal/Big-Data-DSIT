import numpy as np
import collections


class Hashtable:
    """
    This class implements bitwise hashing by spliting
    vectorspace in k hyperplanes. Each random vector has
    dimension equal to d which is equal to the input vector
    dimension
    """

    def __init__(self, k, d):
        """
        constructor
        :param k: number of random vectors -> number of hyperplanes
        :param d: dimensions of rand vectors equal to input ones
        """
        self.k = k
        self.dimensions = d
        self.hashtable = collections.defaultdict(list)  # the actual hashtable
        self.randVectors = np.random.randn(self.k, self.dimensions)  # k hyperplanes - d dimensions

    def hashing(self, vector):
        """
        creating the bitwise hash according to the pattern position
        :param vector: input vector
        :return: return the bitwise hash
        """
        # if above hyperplane then 1 else 0
        return np.array_str(((vector @ self.randVectors.T) > 0).astype('int'))  # returning the bitwise hash

    def __setitem__(self, vector, datalabel):
        """
        :param vector: input vector
        :param datalabel: label of vector in dataset
        """

        self.hashtable[self.hashing(vector)].append(vector)  # appending vector label to bin

    def __getitem__(self, vector):
        """
        :param vector: input vector
        :return: returns all items hashed in the vectors' bin
        """
        hash = self.hashing(vector)
        return self.hashtable.get(hash, [])


#we didnt use this class directly for the assignment due to the time penalty instead a more
#direct implementation using just the hashtable class was used and some inline coding
#however this class is usable and correct for another application
class LSH_RP:
    """
    This class is used to perform LSH random projections using vector inputs
    """

    def __init__(self, l, k, d):
        """
        :param l: number of hashtables to be created
        :param k: number of hyperplanes
        :param d: input vector dimensions
        """
        self.l = l
        self.k = k
        self.d = d
        self.hashtables = list()
        for i in range(self.l):
            self.hashtables.append(Hashtable(self.k, self.d))

    def __setitem__(self, vector, datalabel):
        """
        hashes our input data into the l hashtables
        :param vector: input vector
        :param datalabel: vector datalabel
        """

        for ht in self.hashtables:
            ht[vector] = vector

    def __getitem__(self, vector):
        """
        :param vector: input vector
        :return: returns list containing all the items hashed in the same bin
        """
        result = []
        for ht in self.hashtables:
            result.append(ht[vector])

        return result
