import codecs
import logging
import os

from gensim.models.doc2vec import Doc2Vec, LabeledSentence

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)

class CowReader(object):
    root = '/vol/bigdata/corpora'
    dirs = ['nlcow14ax01', 'nlcow14ax02', 'nlcow14ax03', 'nlcow14ax04',
            'nlcow14ax05', 'nlcow14ax06', 'nlcow14ax07']

    def __iter__(self):
        for directory in CowReader.dirs:
            with codecs.open(
                    os.path.join(CowReader.root, directory, directory + ".xml"),
                    encoding='utf-8') as infile:
                sentence, sentence_id = [], None
                for line in infile:
                    if line.startswith('<s'):
                        sentence_id =re.search('docid="(.*?)"', line).group(0)
                    elif line.startswith('</s>'):
                        yield LabeledSentence(sentence, sentence_id)
                        sentence = []
                    else:
                        word, pos, lemma = line.strip().split('\t')
                        if pos not in ('$.', 'punc'):
                            sentence.append(word.lower())
sentences = CowReader()
model = Doc2Vec(sentences, workers=15)
model.save("/vol/tensusers/fkarsdorp/cow.w2v")
