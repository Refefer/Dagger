import random
import sys

import numpy as np

import scipy.sparse as sp

from sklearn.linear_model import SGDClassifier

from sklearn.metrics import classification_report
from sklearn.cross_validation import KFold

from utils import *

def build(Xs, ys, idxs):
    random.shuffle(idxs)
    X, y = [], []
    for idx in idxs:
        X.extend(Xs[idx])
        y.extend(ys[idx])

    return sp.vstack(X), np.vstack(y)

def train(X, y):
    clf = SGDClassifier(loss="hinge", penalty="l2", n_iter=5)
    clf.fit(X, y.ravel())
    return clf

def main(fn, sp):
    print "Reading in dataset"
    data, classes = readDataset(fn)
    proc = Processor(classes, 2, 2, ohe=False)

    print "Converting to features"
    Xs, ys = [], []
    for d in data:
        X, y = [], []
        for i in xrange(len(d)):
            X.append(proc.transform(d, i))
            y.append(proc.encode_target(d, i))

        Xs.append(X)
        ys.append(y)

    print "Starting KFolding"
    y_trues, y_preds = [], []
    for train_idx, test_idx in KFold(len(data), 10):
        tr_X, tr_y = build(Xs, ys, train_idx)
        print "Training"
        clf = train(tr_X, tr_y)

        seq = Sequencer(proc, clf)

        print "Testing"
        y_true, y_pred = [], []
        for idx in test_idx:
            y_true.extend(y for yi in ys[idx] for y in yi)
            preds = seq.classify(data[idx])
            y_pred.extend(preds)

        y_trues.extend(y_true)
        y_preds.extend(y_pred)
    
    print classification_report(y_trues, y_preds, target_names=proc.labels)

    save(sp, seq)

if __name__ == '__main__':
    random.seed(0)
    main(sys.argv[1], sys.argv[2])
