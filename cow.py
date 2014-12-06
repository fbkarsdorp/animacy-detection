import codecs
import logging
import os

from gensim.models.word2vec import Word2Vec
from gensim.utils import tokenize

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)

class CowReader(object):
    root = '/vol/bigdata/corpora/COW/NLCOW'
    dirs = ['nlcow14ax01', 'nlcow14ax02', 'nlcow14ax03', 'nlcow14ax04',
            'nlcow14ax05', 'nlcow14ax06', 'nlcow14ax07']
    vvb = '/vol/tensusers/fkarsdorp/vvb.tokenized.txt'

    def __iter__(self):
        for directory in CowReader.dirs:
            with codecs.open(
                    os.path.join(CowReader.root, directory, directory + ".xml"),
                    encoding='utf-8') as infile:
                sentence = []
                for line in infile:
                    if line.startswith('<s'):
                        continue
                    elif line.startswith('</s>'):
                        yield sentence
                        sentence = []
                    else:
                        word, pos, lemma = line.strip().split('\t')
                        if pos not in ('$.', 'punc'):
                            sentence.append(word.lower())
        with codecs.open(CowReader.vvb, encoding='utf-8') as vvb:
            for sentence in vvb:
                yield list(tokenize(sentence, lowercase=True))

sentences = CowReader()
model = Word2Vec(sentences, size=300, window=10, min_count=10, workers=20)
model.save("/vol/tensusers/fkarsdorp/cow-vvb.w2v")
