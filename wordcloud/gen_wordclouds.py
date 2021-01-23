import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from wordcloud import WordCloud, STOPWORDS

def generate_wc(input_df, label, curr_dir, stopwords, mask):
    # Input df is a pandas dataframe
    data = input_df.loc[input_df['Label'] == label]
    content = (data['Title'] + ' ') + data['Content']
    # generating wordcloud
    wc = WordCloud(background_color='black', mask=mask, stopwords=stopwords, max_words=250)
    wc.generate("".join(content))
    wc.to_file(os.path.join(curr_dir, label + "_wc.png"))
    print("Wordcloud generated.")


if __name__ == "__main__":

    # read train_set.csv
    path = '../train.csv'
    train_set_df = pd.read_csv(path, sep=',')

    # current dir
    curr_dir = os.path.dirname(os.path.realpath(__file__))

    # set up the stopwords
    # use the the union of sklearn and wordcloud stopwords
    stopwords = set(STOPWORDS)
    stopwords.update(set(ENGLISH_STOP_WORDS))
    stopwords.update(['says'], ['say'], ['said'],
                     ['know'], ['new'], ['make'],
                     ['now'], ['year'], ['one'],
                     ['old'], ['u'], ['s'])

    mask = np.array(Image.open(os.path.join(curr_dir, "cloud_mask.png")))

    # creating wordcloud for each category
    generate_wc(train_set_df, "Business", curr_dir,stopwords, mask)
    generate_wc(train_set_df, "Entertainment", curr_dir, stopwords, mask)
    generate_wc(train_set_df, "Health", curr_dir, stopwords, mask)
    generate_wc(train_set_df, "Technology", curr_dir, stopwords, mask)