import codecs
from collections import defaultdict, Counter
from gensim.models.word2vec import Word2Vec
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_validation import train_test_split

import numpy as np

model = Word2Vec.load_word2vec_format("data/vvb.tokenized.bin", binary=True)

def load_data(limit=None):
    X, y = [[]], [[]]
    with codecs.open("data/sinninghe-tagged.tsv", encoding="utf-8") as infile:
        for i, line in enumerate(infile):
            if limit is not None and i >= limit:
                break
            if line.startswith("<FB/>"):
                X.append([])
                y.append([])
            else:
                fields = line.strip().split('\t')
                X[-1].append([field if field else None for field in fields[:-2]])
                assert X[-1]
                y[-1].append(fields[-2])
    return X, y


vocabulary = defaultdict(Counter)
X, y = load_data()
for i, x in enumerate(X):
    for j, word in enumerate(x):
        label, word = y[i][j], word[0]
        vocabulary[word][label] += 1
vocabulary = {word: max(labels, key=labels.get)
              for word, labels in vocabulary.iteritems()}
X, y = [], []
for word, label in vocabulary.iteritems():
    if word.lower() in model:
        X.append(model[word.lower()])
        y.append(1 if label == 'yes' else 0)
X, y = np.vstack(X), np.array(y)
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
print 'n_samples: %s, n_features: %s' % X.shape
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
clf = SGDClassifier(shuffle=True, n_iter=50)
clf.fit(X_train, y_train)

def predict(word):
    vector = scaler.transform([model[word]])
    return 'inanimate' if clf.predict(vector) == 0 else "animate"
