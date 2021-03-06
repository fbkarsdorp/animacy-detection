import codecs
import ConfigParser
import numpy as np
import scipy.sparse as sp
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.cross_validation import KFold
from sklearn.decomposition import KernelPCA
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, average_precision_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler


from gensim.models.word2vec import Word2Vec
from gensim.matutils import unitvec

def load_data(filename, limit=None, binary=True, lowercase=False):
    X, y = [[]], [[]]
    animacy_specification = dict(
        [line.strip().split('\t') for line in codecs.open("animate-specification.txt", encoding='utf-8')])
    with codecs.open(filename, encoding="utf-8") as infile:
        for i, line in enumerate(infile):
            if limit is not None and i >= limit:
                break
            if line.startswith("<FB/>"):
                X.append([])
                y.append([])
            else:
                fields = line.strip().split('\t')
                lower = lambda w: w.lower() if lowercase else w
                X[-1].append([lower(field) if field else None for field in fields[:-2]])
                assert X[-1]
                if not binary:
                    if fields[-2] == 'yes':
                        if fields[3] in ('noun', 'name'):
                            y[-1].append(animacy_specification[fields[0]])
                        else:
                            y[-1].append("animate")
                    else:
                        y[-1].append("inanimate")
                else:
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
    def __init__(self, model, scale=False):
        self.model = model
        self.scale = scale

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # x is a document, word[0] is the word token
        return np.vstack([self.model[word[0].lower()] if word[0].lower() in self.model else
                          np.zeros(self.model.layer1_size) for x in X for word in x])

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        if self.scale:
            return MinMaxScaler().fit_transform(self.transform(X))
        return self.transform(X)


class WordContextEmbeddings(BaseEstimator):
    def __init__(self, model, window_size=3, scale=False):
        self.model = model
        self.window_size = window_size
        self.scale = scale

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_ = []
        for d, doc in enumerate(X):
            for i, word in enumerate(doc):
                X_.append([doc[j][0].lower() for j in range(
                    i - self.window_size, i + self.window_size) if j >= 0 and j < len(doc)])
        return np.vstack([
            unitvec(np.array([self.model[w] if w in self.model else np.zeros(self.model.layer1_size)
                              for w in window]).mean(axis=0))
            for window in X_])

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        if self.scale:
            return MinMaxScaler().fit_transform(self.transform(X))
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

FIELDNAMES = ['word', 'root', 'lcat', 'pos', 'rel', 'sense', 'frame',
              'special', 'noun_det', 'noun_countable', 'noun_number',
              'verb_auxiliary', 'verb_tense', 'verb_complements', 'cluster',
              'animate', 'reference']

def include_features(X, features):
    header = {w: i for i, w in enumerate(FIELDNAMES)}
    excluded = map(header.get, features)
    return [[[field for i, field in enumerate(word) if i in excluded]
             for word in doc] for doc in X]

if __name__ == '__main__':

    config = ConfigParser.ConfigParser()
    config.read("config.cfg")
    # read the data and extract all features
    X, y = load_data(config.get("data", "annotation"), limit=None,
                     binary=config.getboolean("features", "binary-labels"),
                     lowercase=config.getboolean("features", "lowercase"))
    scores = pd.DataFrame(
        columns=['experiment', 'fold', 'class', 'precision', 'recall', 'Fscore', 'AUC'])
    noun_scores = pd.DataFrame(
        columns=['experiment', 'fold', 'class', 'precision', 'recall', 'Fscore', 'AUC'])
    ambiguous_scores = pd.DataFrame(
        columns=['experiment', 'fold', 'class', 'precision', 'recall', 'Fscore', 'AUC'])
    if config.get("data", "model-type") == 'word2vec':
        model = Word2Vec.load_word2vec_format(config.get("data", "embeddings"), binary=True)
    elif config.get("data", "model-type") == "gensim":
        model = Word2Vec.load(config.get("data", "embeddings"))
    model.init_sims(replace=True)
    # set up a number of experimental settings
    experiments = [('word',), ('word', 'pos'), ('word', 'pos', 'root'),
                   ('word', 'pos', 'root', 'rel')]
    #experiments = experiments + [experiment + ('cluster', )
    #                             for experiment in experiments]
    experiments = experiments + [experiment + ('embeddings', )
                                 for experiment in experiments]
    experiments += [('embeddings', )]
    #experiments += [('cluster', )]
    #experiments = experiments + [experiment + ('context-embeddings',)
    #                             for experiment in experiments]
    #experiments += [('context-embeddings', )]

    if config.get("features", "feature-model") != "all":
        experiments = [experiments[config.getint("features", "feature-model")]]

    classifiers = {
        'logistic-regression': LogisticRegression,
        'sgd': SGDClassifier,
        'svm': SVC,
    }

    classweight = config.get("classifier", "class-weight")
    random_state = config.getint("classifier", "random-state")
    parameters = {

        'logistic-regression': {'C': config.getfloat("classifier", "C"),
               'random_state': random_state,
               'dual': config.getboolean("classifier", "dual"),
               'class_weight': None if classweight != 'auto' else classweight,
               'penalty': config.get('classifier', 'penalty')},

        'sgd': {'loss': config.get("classifier", "loss"),
                'shuffle': config.getboolean("classifier", "shuffle"),
                'n_iter': config.getint("classifier", "sgd-iterations"),
                'penalty': config.get("classifier", "penalty"),
                'random_state': random_state,
                'class_weight': None if classweight != 'auto' else classweight},

        'svm': {'C': config.getfloat("classifier", "C"),
                'kernel': config.get("classifier", "kernel"),
                'class_weight': None if classweight != 'auto' else classweight,
                'random_state': random_state
                }
    }

    #ambiguous_words = set(line.strip() for line in codecs.open(
    #    'data/ambiguous.txt', encoding='utf-8'))
    ambiguous_words = set()
    for line in codecs.open("animate-specification.txt", encoding='utf-8'):
        word, label = line.strip().split('\t')
        if label == "inan-anim":
            ambiguous_words.add(word)

    n_experiments = 0
    scale = config.getboolean("features", "scale")
    for k, (train_index, test_index) in enumerate(KFold(
            len(X), n_folds=config.getint("evaluation", "n-folds"),
            shuffle=True, random_state=random_state)):
        # get the actual data by flattening the documents
        X_train_docs = [X[i] for i in train_index]
        y_train_docs = [label for i in train_index for label in y[i]]
        X_test_docs = [X[i] for i in test_index]
        test_words = [word[0] for i in test_index for word in X[i]]
        y_test_docs = [label for i in test_index for label in y[i]]

        for experiment in experiments:
            backoff = False
            print "Features: %s" % ', '.join(experiment)
            exp_name = '_'.join(experiment)
            window_size = config.getint("features", "window-size")
            if ('embeddings' in experiment and len(experiment) > 1
                and not 'context-embeddings' in experiment):
                features = FeatureStacker(
                    ('windower', Windower(window_size=window_size)),
                    ('embeddings', WordEmbeddings(model, scale=scale)))
                backoff_features = Windower(window_size=window_size)
                backoff = True
                if 'word' not in experiment:
                    experiment = ('word', ) + experiment # needed to extract the vectors
            elif ('embeddings' in experiment and len(experiment) > 2
                and 'context-embeddings' in experiment):
                features = FeatureStacker(
                    ('windower', Windower(window_size=window_size)),
                    ('embeddings', WordEmbeddings(model, scale=scale)),
                    ('context-embeddings', WordContextEmbeddings(model, window_size, scale=scale)))
                backoff_features = Windower(window_size=window_size)
                backoff = True
                if "word" not in experiment:
                    experiment = ('word', ) + experiment # needed to extract the vectors
            elif ('embeddings' in experiment and 'context-embeddings' in experiment):
                features = FeatureStacker(
                    ('embeddings', WordEmbeddings(model, scale=scale)),
                    ('context-embeddings', WordContextEmbeddings(model, window_size, scale=scale)))
                experiment = ('word', ) + experiment # needed to extract the vectors
            elif experiment == ('embeddings', ):
                features = WordEmbeddings(model, scale=scale)
                experiment = ('word', ) + experiment # needed to extract the vectors
            elif ('context-embeddings' in experiment and len(experiment) > 1
                  and not 'embeddings' in experiment):
                features = FeatureStacker(
                    ('windower', Windower(window_size=window_size)),
                    ('context-embeddings', WordContextEmbeddings(model, window_size, scale=scale)))
                backoff_features = Windower(window_size=window_size)
                backoff = True
                if "word" not in experiment:
                    experiment = ('word', ) + experiment # needed to extract the vectors
            elif experiment == ('context-embeddings', ):
                features = WordContextEmbeddings(model, window_size, scale=scale)
                experiment = ('word', ) + experiment
            else:
                features = Windower(window_size=window_size)

            X_train = include_features(X_train_docs, experiment)
            X_test = include_features(X_test_docs, experiment)

            X_train = features.fit_transform(X_train)
            X_test = features.transform(X_test)

            if backoff:
                X_train_backoff = include_features(
                    X_train_docs, [f for f in experiment if f != 'embeddings'])
                X_train_backoff = backoff_features.fit_transform(X_train_backoff)
                X_test_backoff = include_features(
                    X_test_docs, [f for f in experiment if f != 'embeddings'])
                X_test_backoff = backoff_features.transform(X_test_backoff)

            le = LabelEncoder()
            y_train = le.fit_transform(y_train_docs)
            y_test = le.transform(y_test_docs)
            # initialize a classifier
            classifier = config.get("classifier", "classifier")
            clf = classifiers[classifier](**parameters[classifier])
            backoff_clf = classifiers[classifier](**parameters[classifier])
            if (classifier != 'logistic-regression' and config.get("classifier", "loss") != 'log'):
                predict_proba = clf.decision_function
                backoff_predict_proba = backoff_clf.decision_function
            else:
                predict_proba = clf.predict_proba
                backoff_predict_proba = backoff_clf.predict_proba
            print clf.__class__.__name__
            clf.fit(X_train, y_train)
            if backoff:
                backoff_clf.fit(X_train_backoff, y_train)

            if backoff:
                preds, pred_probs = [], []
                for i, word in enumerate(X_test):
                    if test_words[i].lower() not in model:
                        print "Backoff prediction for", test_words[i]
                        preds.append(backoff_clf.predict(X_test_backoff[i])[0])
                        pred_probs.append(backoff_predict_proba(X_test_backoff[i]))
                    else:
                        preds.append(clf.predict(X_test[i])[0])
                        pred_probs.append(predict_proba(X_test[i])[0])
                preds = np.array(preds)
                pred_probs = np.vstack(pred_probs)

            elif exp_name in ("embeddings", "context-embeddings", "embeddings_context-embeddings"):
                preds, pred_probs = [], []
                for i, word in enumerate(X_test):
                    if test_words[i].lower() not in model:
                        print "Default prediction for", test_words[i]
                        if not config.getboolean("features", "binary-labels"):
                            preds.append(2)
                            pred_probs.append(np.array([0.0, 0.0, 1.0]))
                        else:
                            preds.append(0)
                            pred_probs.append(np.array([1.0, 0.0]))
                    else:
                        preds.append(clf.predict(word)[0])
                        pred_probs.append(predict_proba(word)[0])
                preds = np.array(preds)
                pred_probs = np.vstack(pred_probs)
            else:
                preds = clf.predict(X_test)
                pred_probs = predict_proba(X_test)

            p, r, f, s = precision_recall_fscore_support(y_test, preds)
            if not config.getboolean("features", "binary-labels"):
                ap = average_precision_score(y_test, pred_probs[:,1], average="micro")
            else:
                ap = 0
            for label_i, label in enumerate(sorted(set(y_test))):
                if s[label_i] == 0:
                    scores.loc[scores.shape[0]] = np.array(
                        [exp_name, k, label, 1.0, 1.0, 1.0, ap])
                else:
                    scores.loc[scores.shape[0]] = np.array(
                        [exp_name, k, label, p[label_i], r[label_i], f[label_i], ap])
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
            p, r, f, s = precision_recall_fscore_support(
                y_test[noun_preds], preds[noun_preds])
            if not config.getboolean("features", "binary-labels"):
                ap = average_precision_score(y_test[noun_preds], pred_probs[noun_preds][:,1])
            else:
                ap = 0
            for label_i, label in enumerate(sorted(set(y_test[noun_preds]))):
                if s[label_i] == 0:
                    noun_scores.loc[noun_scores.shape[0]] = np.array(
                        [exp_name, k, label, 1.0, 1.0, 1.0, ap])
                else:
                    noun_scores.loc[noun_scores.shape[0]] = np.array(
                        [exp_name, k, label, p[label_i], r[label_i], f[label_i], ap])

            ambiguous_preds = []
            i = 0
            for idx in test_index:
                for j, w in enumerate(X[idx]):
                    if w[0] in ambiguous_words:
                        ambiguous_preds.append(i + j)
                i += len(X[idx])
            print "Classification report on ambigous words:"
            print classification_report(y_test[ambiguous_preds], preds[ambiguous_preds])
            p, r, f, s = precision_recall_fscore_support(
                y_test[ambiguous_preds], preds[ambiguous_preds])
            ap = 0
            for label_i, label in enumerate(sorted(set(y_test[ambiguous_preds]))):
                if s[label_i] == 0:
                    ambiguous_scores.loc[ambiguous_scores.shape[0]] = np.array(
                        [exp_name, k, label, 1.0, 1.0, 1.0, ap])
                else:
                    ambiguous_scores.loc[ambiguous_scores.shape[0]] = np.array(
                        [exp_name, k, label, p[label_i], r[label_i], f[label_i], ap])
