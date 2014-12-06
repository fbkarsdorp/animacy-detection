import codecs
import sys
from sklearn.cross_validation import train_test_split
from gensim.models.word2vec import Word2Vec
from sklearn.metrics import classification_report
import pycrfsuite

def load_data(filename):
    X, y = [[[]]], [[[]]]
    with codecs.open(filename, encoding='utf-8') as infile:
        for i, line in enumerate(infile):
            if line.startswith('<FB/>'):
                X.append([[]])
                y.append([[]])
            else:
                fields = line.strip().split('\t')
                X[-1][-1].append([field if field else None for field in fields[:-2]])
                y[-1][-1].append(fields[-2])
                if fields[0] in '?.!': # sentence breaks TODO: make this more robust
                    X[-1].append([])
                    y[-1].append([])
    return X, y

model = Word2Vec.load(sys.argv[1])
X, y = load_data(sys.argv[2])

X_train_idx, X_test_idx, y_train_idx, y_test_idx = train_test_split(
    range(len(X)), range(len(X)), test_size=0.2, random_state=1)

X_train = [sent for i in X_train_idx for sent in X[i]]
y_train = [label for i in y_train_idx for label in y[i]]
X_test = [sent for i in X_test_idx for sent in X[i]]
y_test = [label for i in y_test_idx for label in y[i]]

def word2features(sent, i, features):
    FIELDNAMES = ['word', 'root', 'lcat', 'pos','rel', 'sense', 'frame',
                  'special','noun_det', 'noun_countable', 'noun_number',
                  'verb_auxiliary', 'verb_tense', 'verb_complements',
                  'animate', 'reference']
    header = {w: k for k, w in enumerate(FIELDNAMES)}
    included = map(header.get, features)
    features = {}
    for j in range(i-2, i+3):
        if j >=0 and j < len(sent):
            for k, field in enumerate(sent[j]):
                if k in included:
                    features['%s%s-%s' % ('-' if j < i else '+' if j > i else '',
                                             abs(i-j), k)] = field.lower()
    if 'embeddings' in features:
        word = sent[i][0].lower()
        if word in model:
            features.update({'w2v:%s' % i: v for i, v in enumerate(model[word])})
    return features

def sent2features(sent, features):
    return pycrfsuite.ItemSequence(
        [word2features(sent, i, features) for i in range(len(sent))])

features = ('word', 'pos', 'embeddings')
X_train = [sent2features(sent, features) for sent in X_train]
X_test = [sent2features(sent, features) for sent in X_test]

trainer = pycrfsuite.Trainer(verbose=False)
for xseq, yseq in zip(X_train, y_train):
    trainer.append(xseq, yseq)

trainer.set_params({
    'c1': 2.0,   # coefficient for L1 penalty
    'c2': 1e-3,  # coefficient for L2 penalty
    'max_iterations': 100,  # stop earlier

    # include transitions that are possible, but not observed
    'feature.possible_transitions': True
})

trainer.train("animacy.crf")
tagger = pycrfsuite.Tagger()
tagger.open('animacy.crf')
preds = [tagger.tag(seq) for seq in X_test]
print classification_report(sum(y_test, []), sum(preds, []))

from collections import Counter
info = tagger.info()

def print_state_features(state_features):
    for (attr, label), weight in state_features:
        print("%0.6f %-6s %s" % (weight, label, attr))

print("Top positive:")
print_state_features(Counter(info.state_features).most_common(20))

print("\nTop negative:")
print_state_features(Counter(info.state_features).most_common()[-20:])
