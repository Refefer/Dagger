import time
import random
import sys

import numpy as np

import scipy.sparse as spa

from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC

from sklearn.metrics import classification_report
from sklearn.model_selection import KFold

from utils import readDataset, Processor, Sequencer, test, save


def build(Xs, ys, idxs, rs):
    if type(idxs) != list:
        idxs = list(idxs)
    rs.shuffle(idxs)

    print('Train IDXS', idxs[:10], '...')

    X, y = [], []
    for idx in idxs:
        X.extend(Xs[idx])
        y.extend(ys[idx])

    return spa.vstack(X), np.vstack(y)

def train(X, y):
    clf = SGDClassifier(loss="hinge", penalty="l2", max_iter=50, tol=1e-3)
    #clf = LinearSVC(penalty="l2", class_weight='auto')
    clf.fit(X, y.ravel())
    return clf

def main(fn, sp):
    print("Reading in dataset")
    data, classes = readDataset(fn)
    print(len(data), " sequences found")
    print("Found classes:", sorted(classes))
    proc = Processor(classes, 2, 2, prefix=(1,3), affix=(2,1), hashes=2,
            features=100000, stem=False, ohe=False)

    print("Converting to features")
    Xs, ys = [], []
    sTime = time.time()
    for i, d in enumerate(data):
        if i % 100 == 0 and i:
            print("Converted %s of %s: %s DPS" % (i, len(data), i / (time.time() - sTime)))

        X, y = [], []
        trad = [x['output'] for x in d]
        for i in range(len(d)):
            X.append(proc.transform(d, trad, i))
            y.append(proc.encode_target(trad, i))

        Xs.append(X)
        ys.append(y)

    print("Starting KFolding")
    rs = np.random.RandomState(seed=2016)
    y_trues, y_preds = [], []
    fold_object = KFold(5, random_state=1)
    for train_idx, test_idx in fold_object.split(data):
        tr_X, tr_y = build(Xs, ys, train_idx, rs)

        print("Training")
        clf = train(tr_X, tr_y)

        seq = Sequencer(proc, clf)

        print("Testing")
        y_true, y_pred = test(data, ys, test_idx, seq)
        print( classification_report(y_true, y_pred))

        y_trues.extend(y_true)
        y_preds.extend(y_pred)
    
    print("Total Report")
    print( classification_report(y_trues, y_preds))

    print("Training all")
    idxs = range(len(Xs))
    tr_X, tr_y = build(Xs, ys, idxs, rs)
    clf = train(tr_X, tr_y)
    seq = Sequencer(proc, clf)

    save(sp, seq)

if __name__ == '__main__':
    random.seed(0)
    main(sys.argv[1], sys.argv[2])
