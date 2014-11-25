import codecs
from sklearn.base import BaseEstimator
from collections import defaultdict, Counter
from gensim.models.word2vec import Word2Vec
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
import numpy as np

#model = Word2Vec.load_word2vec_format("data/vvb.tokenized-w10-d300.bin", binary=True)
#model = Word2Vec.load_word2vec_format("/Users/folgert/data/twnc/w2v/twnc.bin", binary=True)
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

class W2VNN(BaseEstimator):
    def __init__(self, model, k=1):
        self.k = k
        self.model = model

    def fit(self, X, y):
        self.y = {x: c for x, c in zip(X, y)}
        return self

    def predict(self, X):
        y = self.y
        return [y[self.model.most_similar(x)[0][0]] if x in y else None for x in X]

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
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1)
#clf = SGDClassifier(shuffle=True, loss="hinge", n_iter=100)
clf = LinearSVC()
clf.fit(X_train, y_train)
preds = clf.predict(X_test)
print classification_report(y_test, preds)

def predict(word):
    vector = scaler.transform([model[word]])
    return 'inanimate' if clf.predict(vector) == 0 else "animate"
