import pandas as pd
from sklearn import preprocessing
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.svm import LinearSVC
from scipy.sparse import hstack
import metrics
from tqdm import tqdm


def generate_feats(df, save_as):

    print('Generating features')
    for index, row in tqdm(df.iterrows()):
        df.at[index, 'Fuzzy'] = metrics.fuzzy_score(row['Question1'], row['Question2'])
        df.at[index, 'J_sim'] = metrics.jaccard_sim(row['Question1'], row['Question2'])
    df.to_csv(save_as, index=False)



if __name__ == "__main__":

    # read train and test sets
    path = './train.csv'
    path_test = './test_without_labels.csv'
    train_set_df = pd.read_csv(path, sep=',')
    test_set_df = pd.read_csv(path_test, sep=',')

    # generate features for train set
    train_set_df['J_sim'] = 0
    train_set_df['Fuzzy'] = 0
    print(train_set_df.head())
    generate_feats(train_set_df, save_as='./train_features.csv')

    # generate features for test set
    test_set_df['J_sim'] = 0
    test_set_df['Fuzzy'] = 0
    print(test_set_df.head())
    generate_feats(test_set_df, save_as='./test_features.csv')


