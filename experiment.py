import codecs
import sys

import numpy as np
import scipy.sparse as sp
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.cross_validation import KFold
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, average_precision_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder

from gensim.models.word2vec import Word2Vec


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
                X[-1].append([field if field else None for field in fields[:-2]])
                assert X[-1]
                y[-1].append(fields[-2])
    return X, y

def find_quotes(document, max_quote_length=50):
    "Extract the quote ranges from a document."
    in_quote = False
    quotes = []
    for i, token in enumerate(document):
        if token[0] == '"':
            if in_quote:
                quotes[-1] = (quotes[-1], i)
                in_quote = False
            elif in_quote and abs(i - quotes[-1]) > max_quote_length:
                in_quote = False
                quotes = quotes[:-1]
            else:
                quotes.append(i)
                in_quote = True
    return [quote for quote in quotes if isinstance(quote, tuple)]

def add_speakers(document, labels):
    speakers = []
    for start, end in find_quotes(document):
        left_indices = range(start-1, -1, -1)
        right_indices = range(end+1, len(document))
        found_speaker = False
        for indices in (left_indices, right_indices):
            for i in indices:
                if document[i][0] in '?!."':
                    break
                if document[i][4] == 'su':
                    speakers.append(i)
                    found_speaker = True
                    break
            if found_speaker:
                break
        if found_speaker:
            print document[speakers[-1]][0]
        else:
            print 'No speaker found...'

    return [word + [0 if i not in speakers else 1] for i, word in enumerate(document)]


class Windower(BaseEstimator):

    def __init__(self, window_size=5):
        self.window_size = window_size
        self.fitted = False
        self.vectorizer = DictVectorizer(sparse=True)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_ = []
        n_fields = len(X[0][0])
        for d, doc in enumerate(X):
            for i, word in enumerate(doc):
                features = []
                for j in range(i - self.window_size, i):
                    features.extend([None] * n_fields if j < 0 else doc[j])
                features.extend(word)
                for j in range(i + 1, i + self.window_size):
                    features.extend([None] * n_fields if j >= len(doc) else doc[j])
                X_.append({str(k): f for k, f in enumerate(features) if f != None})
        transform = (self.vectorizer.fit_transform if not self.fitted else
                     self.vectorizer.transform)
        self.fitted = True
        return transform(X_)

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)


class WordEmbeddings(BaseEstimator):
    def __init__(self, model):
        self.model = model

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # x is a document, word[0] is the word token
        return np.vstack([self.model[word[0].lower()] if word[0].lower() in self.model else
                          np.zeros(self.model.layer1_size) for x in X for word in x])

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)


class FeatureStacker(BaseEstimator):
    """Stacks several transformer objects to yield concatenated features.
    Similar to pipeline, a list of tuples ``(name, estimator)`` is passed
    to the constructor.
    """
    def __init__(self, *transformer_list):
        self.transformer_list = transformer_list

    def get_feature_names(self):
        pass

    def fit(self, X, y=None):
        for name, trans in self.transformer_list:
            trans.fit(X, y)
        return self

    def transform(self, X):
        features = []
        for name, trans in self.transformer_list:
            features.append(trans.transform(X))
        issparse = [sp.issparse(f) for f in features]
        if np.any(issparse):
            features = sp.hstack(features).tocsr()
        else:
            features = np.hstack(features)
        return features

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

    def get_params(self, deep=True):
        if not deep:
            return super(FeatureStacker, self).get_params(deep=False)
        else:
            out = dict(self.transformer_list)
            for name, trans in self.transformer_list:
                for key, value in trans.get_params(deep=True).iteritems():
                    out['%s__%s' % (name, key)] = value
            return out

FIELDNAMES = ['word', 'root', 'lcat', 'pos','rel', 'sense', 'frame',
              'special','noun_det', 'noun_countable', 'noun_number',
              'verb_auxiliary', 'verb_tense', 'verb_complements','animate',
              'reference']

def include_features(X, features):
    header = {w: i for i, w in enumerate(FIELDNAMES)}
    excluded = map(header.get, features)
    return [[[field for i, field in enumerate(word) if i in excluded]
             for word in doc] for doc in X]

if __name__ == '__main__':

    # read the data and extract all features
    X, y = load_data(sys.argv[2], limit=None)
    scores = pd.DataFrame(
        columns=['experiment', 'fold', 'class', 'precision', 'recall', 'Fscore', 'AUC'])
    noun_scores = pd.DataFrame(
        columns=['experiment', 'fold', 'class', 'precision', 'recall', 'Fscore', 'AUC'])
    ambiguous_scores = pd.DataFrame(
        columns=['experiment', 'fold', 'class', 'precision', 'recall', 'Fscore', 'AUC'])
    model = Word2Vec.load(sys.argv[1])
    # set up a number of experimental settings
    experiments = [('word',), ('word', 'pos'), ('word', 'pos', 'root'),
                   ('word', 'pos', 'root', 'rel'), tuple(FIELDNAMES)]
    experiments = experiments + [experiment + ('embeddings', )
                                 for experiment in experiments]
    experiments += [('embeddings', )]

    classifiers = {
        'lr': LogisticRegression(C=1.0),
        'sgd': SGDClassifier(n_iter=100, shuffle=True),
        'svm': LinearSVC(),
        'knn': KNeighborsClassifier(weights='distance')
    }

    ambiguous_words = set(line.strip() for line in codecs.open(
        'data/ambiguous.txt', encoding='utf-8'))

    n_experiments = 0
    for k, (train_index, test_index) in enumerate(KFold(
            len(X), n_folds=10, shuffle=True, random_state=1)):
        # get the actual data by flattening the documents
        X_train_docs = [X[i] for i in train_index]
        y_train_docs = [label for i in train_index for label in y[i]]
        X_test_docs = [X[i] for i in test_index]
        y_test_docs = [label for i in test_index for label in y[i]]

        for experiment in experiments:
            print "Features: %s" % ', '.join(experiment)
            exp_name = '_'.join(experiment)
            if 'embeddings' in experiment and len(experiment) > 1:
                features = FeatureStacker(('windower', Windower(window_size=3)),
                                          ('embeddings', WordEmbeddings(model)))
            elif 'embeddings' in experiment:
                features = WordEmbeddings(model)
                experiment = ('word', ) + experiment # needed to extract the vectors
            else:
                features = Windower(window_size=3)
            X_train = include_features(X_train_docs, experiment)
            X_test = include_features(X_test_docs, experiment)
            X_train = features.fit_transform(X_train)
            X_test = features.transform(X_test)
            le = LabelEncoder()
            y_train = le.fit_transform(y_train_docs)
            y_test = le.transform(y_test_docs)
            # initialize a classifier
            clf = classifiers[sys.argv[3]]
            print clf.__class__.__name__
            clf.fit(X_train, y_train)
            preds = clf.predict(X_test)
            p, r, f, _ = precision_recall_fscore_support(y_test, preds)
            pred_probs = clf.predict_proba(X_test)
            ap = average_precision_score(y_test, pred_probs[:,1], average="micro")
            scores.loc[n_experiments] = np.array([exp_name, k, 0, p[0], r[0], f[0], ap])
            scores.loc[n_experiments+1] = np.array([exp_name, k, 1, p[1], r[1], f[1], ap])
            print classification_report(y_test, preds)
            print "Classification report on nouns:"
            noun_preds = []
            i = 0
            for idx in test_index:
                for j, w in enumerate(X[idx]):
                    if w[3] in ('noun', 'name'):
                        noun_preds.append(i + j)
                i += len(X[idx])
            print classification_report(y_test[noun_preds], preds[noun_preds])
            p, r, f, _ = precision_recall_fscore_support(
                y_test[noun_preds], preds[noun_preds])
            ap = average_precision_score(y_test[noun_preds], pred_probs[noun_preds][:,1])
            noun_scores.loc[n_experiments] = np.array([exp_name, k, 0, p[0], r[0], f[0], ap])
            noun_scores.loc[n_experiments+1] = np.array([exp_name, k, 1, p[1], r[1], f[1], ap])

            ambiguous_preds = []
            i = 0
            for idx in test_index:
                for j, w in enumerate(X[idx]):
                    if w[0] in ambiguous_words:
                        ambiguous_preds.append(i + j)
                i += len(X[idx])
            print "Classification report on ambigous words:"
            print classification_report(y_test[ambiguous_preds], preds[ambiguous_preds])
            p, r, f, _ = precision_recall_fscore_support(
                y_test[ambiguous_preds], preds[ambiguous_preds])
            ap = average_precision_score(y_test[ambiguous_preds], pred_probs[ambiguous_preds][:,1])
            ambiguous_scores.loc[n_experiments] = np.array([exp_name, k, 0, p[0], r[0], f[0], ap])
            ambiguous_scores.loc[n_experiments+1] = np.array([exp_name, k, 1, p[1], r[1], f[1], ap])

            n_experiments += 2

        # print "Fitting a majority vote DummyClassifier"
        # dummy_clf = DummyClassifier(strategy='constant', constant=1)
        # dummy_clf.fit(X_train, y_train)
        # preds = dummy_clf.predict(X_test)
        # print "Classification report for Dummy Classifier:"
        # print classification_report(y_test, preds)
