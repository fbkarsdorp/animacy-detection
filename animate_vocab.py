import codecs
from collections import defaultdict, Counter

from sklearn.svm import LinearSVC
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report

from gensim.models import Word2Vec, Doc2Vec
from gensim.models.word2vec import LineSentence
from gensim.models.doc2vec import LabeledLineSentence, LabeledSentence
from gensim.utils import tokenize

class NearestNeighborClassifier(object):
    def __init__(self, distance_fn, k=1):
        self.k = k
        self.distance_fn = distance_fn

    def fit(self, X, y):
        self.examples, self.labels = X, y
        return self

    def predict(self, X):
        indexes = range(len(self.labels))
        def _dist(x, i):
            return self.distance_fn(x, self.examples[i])
        return [self.labels[min(indexes, key=lambda i: _dist(x, i))] for x in X]

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
    vocabulary = {word: max(labels, key=labels.get)
                  for word, labels in vocabulary.iteritems()}
    return zip(*vocabulary.items())

X, y = load_data()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1)
data = [list(tokenize(s, lowercase=True)) for s in codecs.open("data/vvb.sentences.txt")]
data = [LabeledSentence(sent, "SENT_%s" % i) for i, sent in enumerate(data)]
#model = Word2Vec(data, sample=0, window=5, min_count=1, size=100, workers=4)
#model = Word2Vec.load_word2vec_format("/Users/folgert/data/twnc/w2v/twnc.bin", binary=True)
model = Doc2Vec(data, sample=0, window=5, min_count=1, size=100, workers=4)
knn = NearestNeighborClassifier(
    lambda x, y: 1 - (model.similarity(x, y) if x in model and y in model else 0))
knn.fit(X_train, y_train)
preds = knn.predict(X_test)
print classification_report(y_test, preds)
for i, (pred, true) in enumerate(zip(preds, y_test)):
    if pred != true:
        print pred, true, X_test[i]
