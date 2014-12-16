import codecs
import os
import sys
import glob
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from gensim.models.word2vec import Word2Vec
from experiment import Windower, WordEmbeddings, FeatureStacker

from frog import Frog

def load_data(filename, limit=None):
    X, y = [[]], [[]]
    with codecs.open(filename, encoding="utf-8") as infile:
        for i, line in enumerate(infile):
            if limit is not None and i >= limit:
                break
            if line.startswith("<FB/>"):
                X.append([])
                y.append([])
            else:
                fields = line.strip().split('\t')
                X[-1].append([field if field else None for field in fields[:-1]])
                assert X[-1]
                y[-1].append('yes' if fields[-1] == 'animate' else 'no')
    return X, y

X, y = load_data(sys.argv[1], limit=None)
model = Word2Vec.load_word2vec_format(sys.argv[2], binary=True)
feature_vectorizer = FeatureStacker(('windower', Windower(window_size=3)),
                                    ('embeddings', WordEmbeddings(model)))

X = feature_vectorizer.fit_transform([[word for word in doc] for doc in X])
y = LabelEncoder().fit_transform([l for labels in y for l in labels])

clf = LogisticRegression().fit(X, y)
frogger = Frog(int(sys.argv[3]))

for filename in glob.glob(os.path.join(sys.argv[4], "*")):
    print filename
    characters = Counter()
    with codecs.open(filename, encoding='utf-8') as infile:
        doc = infile.read()
        document = frogger.tag(doc)
        document = [[f.decode('utf-8') for f in w[:-1]]
                    for sentence in document for w in sentence]
    X_test = feature_vectorizer.transform([document])
    preds = clf.predict(X_test)
    for i, pred in enumerate(preds):
        if pred == 1 and document[i][2] in ('N', 'SPEC'):
            characters[document[i][0]] += 1
    print ', '.join(sorted(characters, key=characters.__getitem__, reverse=True))
