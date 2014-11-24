import codecs
from functools import partial
from pprint import pprint

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.cross_validation import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_regression, chi2, f_classif
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder


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

    def fit(self, documents, y=None):
        return self

    def transform(self, documents, labels):
        X, y = [], []
        n_fields = len(documents[0][0])
        for d, document in enumerate(documents):
            for i, word in enumerate(document):
                features = []
                for j in range(i - self.window_size, i):
                    features.extend([None] * n_fields if j < 0 else document[j])
                features.extend(word)
                for j in range(i + 1, i + self.window_size):
                    features.extend([None] * n_fields if j >= len(document) else document[j])
                X.append({str(k): f for k, f in enumerate(features) if f != None})
                y.append(labels[d][i])
        transform = (self.vectorizer.fit_transform if not self.fitted else
                     self.vectorizer.transform)
        self.fitted = True
        return transform(X), y


class FeatureStacker(BaseEstimator):
    """Stacks several transformer objects to yield concatenated features.
    Similar to pipeline, a list of tuples ``(name, estimator)`` is passed
    to the constructor.
    """
    def __init__(self, transformer_list):
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
        issparse = [sparse.issparse(f) for f in features]
        if np.any(issparse):
            features = sparse.hstack(features).tocsr()
        else:
            features = np.hstack(features)
        return features

    def get_params(self, deep=True):
        if not deep:
            return super(FeatureStacker, self).get_params(deep=False)
        else:
            out = dict(self.transformer_list)
            for name, trans in self.transformer_list:
                for key, value in trans.get_params(deep=True).iteritems():
                    out['%s__%s' % (name, key)] = value
            return out


# read the data and extract all features
X, y = load_data(limit=None)
# X = [add_speakers(x, y_) for x, y_ in zip(X, y)]
# split the data into a train and test set

for selector_name, selector in (
        ("chi2", chi2),
        ("f1-anova", f_classif),
        ("regression-anova", partial(f_regression, center=False))):
    for window in (1, 2, 3, 4, 5, 10):
        windower = Windower(window)
        X_train_idx, X_test_idx, y_train_idx, y_test_idx = train_test_split(
            range(len(X)), range(len(X)), test_size=0.2, random_state=1)
        X_train, y_train = [X[i] for i in X_train_idx], [y[i] for i in y_train_idx]
        X_test, y_test = [X[i] for i in X_test_idx], [y[i] for i in y_test_idx]
        X_train, y_train = windower.transform(X_train, y_train)
        X_test, y_test = windower.transform(X_test, y_test)
        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
        y_test = le.transform(y_test)
        # initialize a classifier
        clf = SGDClassifier(shuffle=True)
        # experiment with a feature_selection filter
        anova_filter = SelectPercentile(selector)
        percentiles = (1, 3, 6, 10, 15, 20, 30, 40, 60, 80, 100)
        # construct the pipeline
        pipeline = Pipeline([('anova', anova_filter), ('clf', clf)])
        # these are the parameters we're gonna test for in the grid search
        parameters = {
            'clf__class_weight': (None, 'auto'),
            'clf__alpha': 10.0**-np.arange(1,7),
            'clf__n_iter': (20, 50, 100, 200, np.ceil(10**6. / len(X_train))),
            'clf__penalty': ('l2', 'elasticnet'),
            'anova__percentile': percentiles}
        grid_search = GridSearchCV(
            pipeline, param_grid=parameters, n_jobs=12, scoring='f1', verbose=1)
        print "Performing grid search..."
        print "pipeline:", [name for name, _ in pipeline.steps]
        print "Window:", window
        print "Feauture Selection", selector_name
        print "parameters:"
        pprint(parameters)
        grid_search.fit(X_train, y_train)

        print "Best score: %0.3f" % grid_search.best_score_
        print "Best parameters set:"
        best_parameters = grid_search.best_estimator_.get_params()
        for param_name in sorted(parameters.keys()):
            print "\t%s: %r" % (param_name, best_parameters[param_name])

        print
        preds = grid_search.predict(X_test)
        print "Classification report after grid search:"
        print classification_report(y_test, preds)
        print

        print "Classification report on nouns after grid search:"
        noun_preds = []
        i = 0
        for idx in X_test_idx:
            for j, w in enumerate(X[idx]):
                if w[3] in ('noun', 'name'):
                    noun_preds.append(i + j)
            i += len(X[idx])
        print classification_report(preds[noun_preds], y_test[noun_preds])

        print "Fitting a majority vote DummyClassifier"
        dummy_clf = DummyClassifier(strategy='constant', constant=1)
        dummy_clf.fit(X_train, y_train)
        preds = dummy_clf.predict(X_test)
        print "Classification report for Dummy Classifier:"
        print classification_report(y_test, preds)

        print 'Fitting `subject=animate` classifier:'
        preds = [1 if w[4].startswith('su') else 0 for i in X_test_idx for w in X[i]]
        print "Classification report for `subject=animate` classifier:"
        print classification_report(y_test, preds)

        print 'Fitting `subject/object=animate` classifier:'
        preds = [1 if w[4].startswith(('su', 'obj')) else 0 for i in X_test_idx for w in X[i]]
        print "Classification report for `subject=animate` classifier:"
        print classification_report(y_test, preds)
