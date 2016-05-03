from itertools import izip
import random
import sys

import numpy as np

import scipy.sparse as sp

from sklearn.linear_model import SGDClassifier

from sklearn.metrics import classification_report
from sklearn.cross_validation import KFold

from utils import *

def generate(seq, y_set, exp, sequencer):
    """
    Generates a new tradjectory.
    """
    # copy orig seq
    trad = []

    for i in xrange(len(seq)):
        if random.random() < exp:
            # Oracle
            output = y_set[i]
        else:
            # Policy
            output = sequencer._partial_pred(seq, trad, i)

        trad.append(output)

    return trad

import pprint
def gen_dataset(Xss, yss, proc, clf, epoch):
    sequencer = Sequencer(proc, clf)

    # Generate new dataset
    nX, nY, trads = [], [], []
    for Xs, ys in izip(Xss, yss):
        # build tradjectory
        #trad = generate(Xs, ys, .95 ** epoch, sequencer)
        trad = generate(Xs, ys, 1 if epoch == 0 else 0, sequencer)
        # If different than the input, add it
        if any(y != t for y, t in izip(ys, trad)):
            nX.append(Xs)
            nY.append(ys)
            trads.append(trad)
            #pprint.pprint(zip([X['feature'] for X in Xs], ys, trad))

    return nX, nY, trads

def key(X, y):
    idxs = tuple(X.nonzero()[1].tolist())
    vs = tuple(X[0, i] for i in idxs)
    return (idxs, vs, y)

def train(Xss, yss, proc, epochs=30):
    cache = {}

    # Make copies for convenience
    nXss = Xss[:]
    nyss = yss[:]
    ntradss = yss[:]

    # Initial policy just mimics the expert
    clf = train_model(Xss, yss, yss, proc, cache)

    for i in xrange(1, epochs):
        sequencer = Sequencer(proc, clf)

        # Generate new dataset
        print "Generating new dataset"
        uniques = len(cache) 
        nX, nY, nT = gen_dataset(Xss, yss, proc, clf, i)
        nXss.extend(nX)
        nyss.extend(nY)
        ntradss.extend(nT)

        # Retrain
        print "Retraining"
        clf = train_model(nXss, nyss, ntradss, proc, cache)

        # No new data - exit
        if len(cache) == uniques:
            break

    return clf

def cache_key(Xs, ys, trads):
    f = tuple(x['feature'] for x in Xs)
    return (f, tuple(ys), tuple(trads))

def build_training_data(Xss, yss, tradss, proc, cache):
    seen = set()
    n = c = 0
    print "Cache size:", len(cache), len(Xss)
    for Xs, ys, trads in izip(Xss, yss, tradss):
        n += 1
        ci = cache_key(Xs, ys, trads)
        if ci not in cache:
            c += 1
            feats, targets = [], []
            for i in xrange(len(Xs)):
                X = proc.transform(Xs, trads, i)
                feats.append(X)
                targets.append(proc.encode_target(ys, i)[0])
            
            cache[ci] = (feats, targets)

        for X, y in izip(*cache[ci]):
            k = key(X, y)
            if k not in seen:
                seen.add(k)
                yield X, y

    print "Cache Hit:", (n - c) / float(n)

def train_model(Xss, yss, trads, proc, cache):
    print "Featurizing..."
    tX, tY = [], []
    for X, y in build_training_data(Xss, yss, trads, proc, cache):
        tX.append(X)
        tY.append(y)

    tX, tY = sp.vstack(tX), np.vstack(tY)

    print "Running learner..."
    clf = SGDClassifier(loss="hinge", penalty="l2", n_iter=5)
    print tX.shape, tY.shape
    clf.fit(tX, tY.ravel())
    return clf

def subset(Xss, yss, idxs):
    random.shuffle(idxs)
    tXss = [Xss[i] for i in idxs]
    tyss = [yss[i] for i in idxs]
    return tXss, tyss

def main(fn, outf):
    print "Reading in dataset"
    data, classes = readDataset(fn)
    print len(data), " sequences found"
    print "Found classes:", sorted(classes)
    proc = Processor(classes, 2, 2, ohe=False)

    yss = []
    ryss = []
    for Xs in data:
        ys = [x['output'] for x in Xs]
        yss.append(ys)
        ryss.append([proc.encode_target(ys, i) for i in xrange(len(ys))])

    print "Starting KFolding"
    y_trues, y_preds = [], []
    for train_idx, test_idx in KFold(len(data), 3, random_state=1):
        tr_X, tr_y = subset(data, yss, train_idx)
        print "Training"
        clf = train(tr_X, tr_y, proc)

        seq = Sequencer(proc, clf)

        print "Testing"
        y_true, y_pred = test(data, ryss, test_idx, seq)
        print classification_report(y_true, y_pred, target_names=proc.labels)

        y_trues.extend(y_true)
        y_preds.extend(y_pred)

    print "Total Report"
    print classification_report(y_trues, y_preds, target_names=proc.labels)

    print "Training all"
    idxs = range(len(data))
    tr_X, tr_y = subset(data, yss, idxs)
    clf = train(tr_X, tr_y, proc)
    seq = Sequencer(proc, clf)

    print "Testing"
    y_true, y_pred = test(data, ryss, idxs, seq)
    print classification_report(y_true, y_pred, target_names=proc.labels)

    save(outf, seq)

if __name__ == '__main__':
    random.seed(0)
    main(sys.argv[1], sys.argv[2])
