import codecs
import sys
import glob
import os
sys.path.append("/Users/folgert/local/brat/server/src")

from tokenise import gtb_token_boundary_gen
import numpy as np

from sklearn.metrics import confusion_matrix


annotators = folgert, marten = 0, 1

def read_annotation(filename):
    for line in codecs.open(filename, encoding='utf-8'):
        _, entity, _ = line.strip().split('\t')
        _, start, end = entity.split()
        yield filename.split('/')[-1][1:], int(start), int(end)

def word_positions(filename):
    with codecs.open(filename) as infile:
        text = infile.read()
    return [o for o in gtb_token_boundary_gen(text)]

annotations = {}
root = "/Users/folgert/Dropbox/stories/data/sinninghe-kappa"
for filename in glob.glob(os.path.join(root, '*.ann')):
    filename_str = filename.split('/')[-1]
    for start, end in word_positions(filename.replace(".ann", ".txt")):
        found = False
        annotations[filename_str[1:], start, end] = [0, 0]

for filename in glob.glob(os.path.join(root, '*.ann')):
    filename_str = filename.split('/')[-1]
    for annotation, start, end in read_annotation(filename):
        if (annotation, start, end) in annotations:
            annotations[annotation, start, end][
                folgert if filename_str.startswith('F') else marten] = 1
        else:
            for (f, s, e) in annotations:
                if f == annotation and (s >= start and e <= end):
                    annotations[annotation, s, e][
                        folgert if filename_str.startswith('F') else marten] = 1

folgert_ann, marten_ann = zip(*annotations.values())
cm = confusion_matrix(folgert_ann, marten_ann)

def kappa(table):
    table = np.array(table, dtype=np.float)
    n_annotations = float(table.sum())
    p_a = (table[1, 1] + table[0, 0]) / n_annotations
    p_1 = table[1,:].sum() / n_annotations * table[:,1].sum() / n_annotations
    p_0 = table[0,:].sum() / n_annotations * table[:,0].sum() / n_annotations
    p_e = p_1 + p_0
    return (p_a - p_e) / (1 - p_e)


def test():
    cm = np.array([[20., 5.], [10., 15.]])
    assert abs(kappa(cm) - 0.4) < 1e-5
    print 'test passed'

test()
