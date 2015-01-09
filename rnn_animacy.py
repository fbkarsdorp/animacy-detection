import codecs
import os
import time
import random
import sys

from collections import defaultdict, OrderedDict

import numpy as np 

from sklearn.cross_validation import train_test_split
from sklearn.metrics import f1_score
from sklearn.utils import shuffle

from gensim.models import Word2Vec

import theano
from theano import tensor as T

def minibatch(l, bs):
    '''
    l :: list of word idxs
    return a list of minibatches of indexes
    which size is equal to bs
    border cases are treated as follow:
    eg: [0,1,2,3] and bs = 3
    will output:
    [[0],[0,1],[0,1,2],[1,2,3]]
    '''
    out  = [l[:i] for i in xrange(1, min(bs,len(l)+1) )]
    out += [l[i-bs:i] for i in xrange(bs,len(l)+1) ]
    assert len(l) == len(out)
    return out

def contextwin(l, win):
    '''
    win :: int corresponding to the size of the window
    given a list of indexes composing a sentence
    it will return a list of list of indexes corresponding
    to context windows surrounding each word in the sentence
    '''
    assert (win % 2) == 1
    assert win >=1
    l = list(l)

    lpadded = win/2 * [-1] + l + win/2 * [-1]
    out = [ lpadded[i:i+win] for i in range(len(l)) ]

    assert len(out) == len(l)
    return out


class RNN(object):
    
    def __init__(self, emb, nh, nc, ne, de, cs):
        '''
        nh :: dimension of the hidden layer
        nc :: number of classes
        ne :: number of word embeddings in the vocabulary
        de :: dimension of the word embeddings
        cs :: word window context size 
        '''
        # parameters of the model
        if emb is not None:
            emb = np.vstack((emb, np.random.uniform(-1.0, 1.0, emb.shape[1]) * 0.2))
            self.emb = theano.shared(emb.astype(theano.config.floatX))
        else:
            self.emb = theano.shared(0.2 * np.random.uniform(-1.0, 1.0,\
                       (ne+1, de)).astype(theano.config.floatX)) # add one for PADDING at the end
        self.Wx  = theano.shared(0.2 * np.random.uniform(-1.0, 1.0,\
                   (de * cs, nh)).astype(theano.config.floatX))
        self.Wh  = theano.shared(0.2 * np.random.uniform(-1.0, 1.0,\
                   (nh, nh)).astype(theano.config.floatX))
        self.W   = theano.shared(0.2 * np.random.uniform(-1.0, 1.0,\
                   (nh, nc)).astype(theano.config.floatX))
        self.bh  = theano.shared(np.zeros(nh, dtype=theano.config.floatX))
        self.b   = theano.shared(np.zeros(nc, dtype=theano.config.floatX))
        self.h0  = theano.shared(np.zeros(nh, dtype=theano.config.floatX))

        # bundle
        self.params = [ self.emb, self.Wx, self.Wh, self.W, self.bh, self.b, self.h0 ]
        self.names  = ['embeddings', 'Wx', 'Wh', 'W', 'bh', 'b', 'h0']
        idxs = T.imatrix() # as many columns as context window size/lines as words in the sentence
        x = self.emb[idxs].reshape((idxs.shape[0], de*cs))
        y    = T.iscalar('y') # label

        def recurrence(x_t, h_tm1):
            h_t = T.nnet.sigmoid(T.dot(x_t, self.Wx) + T.dot(h_tm1, self.Wh) + self.bh)
            s_t = T.nnet.softmax(T.dot(h_t, self.W) + self.b)
            return [h_t, s_t]

        [h, s], _ = theano.scan(fn=recurrence, \
            sequences=x, outputs_info=[self.h0, None], \
            n_steps=x.shape[0])

        p_y_given_x_lastword = s[-1,0,:]
        p_y_given_x_sentence = s[:,0,:]
        y_pred = T.argmax(p_y_given_x_sentence, axis=1)

        # cost and gradients and learning rate
        lr = T.scalar('lr')
        nll = -T.mean(T.log(p_y_given_x_lastword)[y])
        gradients = T.grad( nll, self.params )
        updates = OrderedDict(( p, p-lr*g ) for p, g in zip( self.params , gradients))
        
        # theano functions
        self.classify = theano.function(inputs=[idxs], outputs=y_pred)

        self.train = theano.function( inputs  = [idxs, y, lr],
                                      outputs = nll,
                                      updates = updates )

        self.normalize = theano.function( inputs = [],
                         updates = {self.emb:\
                         self.emb/T.sqrt((self.emb**2).sum(axis=1)).dimshuffle(0,'x')})

    def save(self, folder):   
        for param, name in zip(self.params, self.names):
            np.save(os.path.join(folder, name + '.npy'), param.get_value())


def Vocab():
    d = defaultdict()
    d.default_factory = lambda: len(d)
    return d

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

def to_sentences(indices, X, y, vocab, v):
    sentences, labels = [], []
    for i in indices:
        doc, doc_labels = X[i], y[i]
        sentences.append([])
        labels.append([])
        for word, label in zip(doc, doc_labels):
            if word[0].lower() not in vocab:
                if word[0] in '.!?':
                    sentences.append([])
                    labels.append([])
                continue
            sentences[-1].append(v[word[0].lower()])
            labels[-1].append(1 if label == 'yes' else 0)
        if not sentences[-1]:
            sentences = sentences[:-1]
            labels = labels[:-1]
    for sent, label in zip(sentences, labels):
        if sent:
            yield sent, label

if __name__ == '__main__':
    pretrained_embeddings = Word2Vec.load(sys.argv[1])
    pretrained_embeddings.init_sims(replace=True)
    embeddings = pretrained_embeddings.syn0
    de = embeddings.shape[0]
    vocab = pretrained_embeddings.vocab
    learning_rate = 0.0627142536696559
    verbose = 1 if sys.argv[5] == 'verbose' else 0
    bs = 9
    n_hidden = int(sys.argv[6])
    window = int(sys.argv[3])
    n_epochs = int(sys.argv[4])
    X, y = load_data(sys.argv[2])
    v = Vocab()

    indices = range(len(X))
    train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=1)
    train_sents, train_labels = zip(*to_sentences(train_indices, X, y, vocab, v))
    test_sents, test_labels = zip(*to_sentences(test_indices, X, y, vocab, v))

    emb = []
    for w in sorted(v, key=v.__getitem__):
        emb.append(embeddings[vocab[w].index])
    emb = np.array(emb)

    vocsize = len(v)
    nclasses = 2
    np.random.seed(1983)
    random.seed(1983)
    rnn = RNN(emb=emb, nh=n_hidden, nc=nclasses, ne=vocsize, de=emb.shape[1], cs=window)
    best_f1 = -np.inf
    best_epoch = 0
    current_learning_rate = learning_rate

    for epoch in range(50):
        train_sents, train_labels = shuffle(train_sents, train_labels, random_state=1)
        current_epoch = epoch
        tic = time.time()
        for i in range(len(train_sents)):
            cwords = contextwin(train_sents[i], window)
            words = [np.asarray(x).astype('int32') for x in minibatch(cwords, 9)]
            labels = train_labels[i]
            for word_batch, label_last_word in zip(words, labels):
                rnn.train(word_batch, label_last_word, current_learning_rate)
                rnn.normalize()
            if verbose:
                print '[learning] epoch %i >> %2.2f%% completed in %.2f (sec) <<\r' % (
                    epoch, (i + 1) * 100. / len(train_sents), (time.time() - tic)),
                sys.stdout.flush()
        # evaluation
        preds_test = [rnn.classify(np.asarray(contextwin(x, 3)).astype('int32')) for x in test_sents]
        preds_test = sum(map(list, preds_test), [])
        y_test = sum(test_labels, [])
        test_f1 = f1_score(y_test, preds_test)
        if test_f1 > best_f1:
            best_f1 = test_f1
            rnn.save(".")
            if verbose:
                print 'NEW BEST: epoch', epoch, 'valid F1', test_f1, 'best test F1', test_f1, ' '*20
            best_epoch = epoch
        else:
            print ''

        if abs(best_epoch - epoch) >= 10:
            current_learning_rate *= 0.5
        if current_learning_rate < 1e-5:
            break
