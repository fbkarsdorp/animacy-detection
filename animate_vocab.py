import codecs
import sys

from collections import defaultdict, Counter

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report

from gensim.models import Word2Vec


def load_data():
    X, y = [], []
    with codecs.open("data/sinninghe-tagged.tsv", encoding="utf-8") as infile:
        for i, line in enumerate(infile):
            if not line.startswith("<FB/>"):
                fields = line.strip().split('\t')
                X.append([field if field else None for field in fields[:-2]])
                y.append(1 if fields[-2] == "yes" else 0)
    vocabulary = defaultdict(Counter)
    for i, word in enumerate(X):
        vocabulary[word[0].lower()][y[i]] += 1
    # simple majority vote on the classification...
    vocabulary = {word: max(labels, key=labels.get)
                  for word, labels in vocabulary.iteritems()}
    return zip(*vocabulary.items())

def filter_words(X, y, model):
    """Return a list of tuples of (word, label) that contains only words
    that are in the vocabulary of the model."""
    return [(word, label) for word, label in zip(X, y) if word in model]

X, y = load_data()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1)
model = Word2Vec.load(sys.argv[1])

X_train, y_train = zip(*filter_words(X_train, y_train, model))
X_test, y_test = zip(*filter_words(X_test, y_test, model))
X_train, X_test = np.vstack(X_train), np.vstack(X_test)

clf = LogisticRegression()
clf.fit(X_train, y_train)
preds = clf.predict(X_test)
print classification_report(y_test, preds)
