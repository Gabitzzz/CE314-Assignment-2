# NAME: Litcan Nicolae-Gabriel
# REG. NO. 1903165

import nltk
import pandas as pd
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.tree import DecisionTreeClassifier
import re
from sklearn.linear_model import LogisticRegression
stemmer = PorterStemmer()
tokenizer = RegexpTokenizer(r'[0-9a-zA-Z]+')
# r'[0-9a-zA-Z]+'

df = pd.read_csv('IMDB Dataset.csv')

# stop_words_pattern = re.compile('[^0-9a-zA-Z](' + '|'.join(stopwords.words('english')) + ')[^0-9a-zA-Z]')
# print(stop_words_pattern)

# geeks for geeks solution
stop_words = set(stopwords.words('english'))

def stem_sentence(tokens):
    return [stemmer.stem(t) for t in tokens]

df['review'] = df['review'].apply(str.lower)
df['review'] = df['review'].apply(tokenizer.tokenize)

def remove_stop_words(sentence):
    return [w for w in sentence if w not in stop_words]
    # return re.sub(stop_words_pattern, ' ', sentence, re.IGNORECASE)

print(df['review'].iloc[0])

df['review'] = df['review'].apply(remove_stop_words)
print(df['review'].iloc[0])

# df['review'] = df['review'].apply(lambda tokens: [stemmer.stem(t) for t in tokens])

df_train = df.iloc[:40000 ,:]
df_test = df.iloc[10000:,:]

print(df_test)

# classifier.train(train_set)
# return classifier.classify_many(test_features)

# print(df_train['sentiment'].value_counts())
# print(df_test['sentiment'].value_counts())

# https://stackoverflow.com/questions/952914/how-to-make-a-flat-list-out-of-a-list-of-lists

def flatten(t):
    return [item for sublist in t for item in sublist]
all_words = nltk.FreqDist( flatten(df['review'].tolist()) )
# all_words = nltk.FreqDist(w.lower() for w in df['review'].words())
word_features = list(all_words)[:100]

print(all_words.most_common(20))


def document_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
    return features

train_set =  df_train.apply(lambda x: (document_features(x['review']), x['sentiment']), axis=1)
test_set =  df_test.apply(lambda x: (document_features(x['review']), x['sentiment']), axis=1)
classifier = nltk.DecisionTreeClassifier.train(train_set)

print( nltk.classify.accuracy(classifier, test_set) )

# def document_features(document): [2]
#     document_words = set(document) [3]
#     features = {}
#     for word in word_features:
#         features['contains({})'.format(word)] = (word in document_words)
#     return features