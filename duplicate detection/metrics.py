import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from fuzzywuzzy import fuzz

# uncomment to download resources used for the word tokenizer
#nltk.download('punkt')


def tokenize_text(text):
    text = str(text)
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text.lower())
    filtered_tokens = [w for w in word_tokens if not w in stop_words]
    return filtered_tokens


def jaccard_sim(text1, text2):
    tk_text1 = set(tokenize_text(text1))
    tk_text2 = set(tokenize_text(text2))
    j_sim = float(len(tk_text1.intersection(tk_text2)) / len(tk_text1.union(tk_text2)))
    return int(j_sim * 100)


def fuzzy_score(text1, text2):
    f_score = fuzz.token_set_ratio(text1, text2)
    return f_score


