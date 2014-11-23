import codecs
from functools import partial
from pprint import pprint

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.cross_validation import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import SelectPercentile, f_regression, chi2
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

def load_data(limit=None):
    X, y = [[]], []
    with codecs.open("data/sinninghe-tagged.tsv", encoding="utf-8") as infile:
        for i, line in enumerate(infile):
            if limit is not None and i >= limit:
                break
            if line.startswith("<FB/>"):
                X.append([])
            else:
                fields = line.strip().split('\t')
                X[-1].append([field if field else None for field in fields[:-2]])
                y.append(fields[-2])
    return X, y

class Windower(BaseEstimator):

    def __init__(self, window_size=5):
        self.window_size = window_size

    def fit(self, documents, y=None):
        return self

    def transform(self, documents):
        X = []
        n_fields = len(documents[0][0])
        for document in documents:
            for i, word in enumerate(document):
                features = []
                for j in range(i - self.window_size, i):
                    features.extend([None] * n_fields if j < 0 else document[j])
                features.extend(word)
                for j in range(i + 1, i + self.window_size):
                    features.extend([None] * n_fields if j >= len(document) else document[j])
                X.append({str(k): f for k, f in enumerate(features) if f != None})
        return DictVectorizer(sparse=True).fit_transform(X)


# read the data and extract all features
X, y = load_data(limit=None)
y = np.array(y)
le = LabelEncoder()
y_bak, y = y, le.fit_transform(y)
# split the data into a train and test set
for window in (1, 2, 5, 10):
    windower = Windower(window)
    X_ = windower.transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_, y, test_size=0.2, random_state=1)
    # initialize a classifier
    clf = SGDClassifier()
    # experiment with a feature_selection filter
    anova_filter = SelectPercentile(partial(f_regression, center=False))
    percentiles = (1, 3, 6, 10, 15, 20, 30, 40, 60, 80, 100)
    # construct the pipeline
    pipeline = Pipeline([('anova', anova_filter), ('clf', clf)])
    # these are the parameters we're gonna test for in the grid search
    parameters = {'clf__class_weight': (None, 'auto'),
                  'clf__alpha': (0.01, 0.001, 0.0001, 0.00001),
                  'clf__n_iter': (20, 50, 100),
                  'clf__penalty': ('l2', 'elasticnet'),
                  'anova__percentile': percentiles}

    grid_search = GridSearchCV(pipeline, param_grid=parameters, n_jobs=8, scoring='f1', verbose=1)
    print "Performing grid search..."
    print "pipeline:", [name for name, _ in pipeline.steps]
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

    print "Fitting a majority vote DummyClassifier"
    dummy_clf = DummyClassifier(strategy='constant', constant=1)
    dummy_clf.fit(X_train, y_train)
    preds = dummy_clf.predict(X_test)
    print "Classification report for Dummy Classifier:"
    print classification_report(y_test, preds)
