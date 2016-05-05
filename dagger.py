from itertools import izip
import random
import sys

import numpy as np

import scipy.sparse as sp

from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC, SVC

from sklearn.metrics import classification_report
from sklearn.cross_validation import KFold, ShuffleSplit

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
        trad = generate(Xs, ys, 1 if epoch == 0 else 0, sequencer)
        # If different than the input, add it
        if any(y != t for y, t in izip(ys, trad)):
            nX.append(Xs)
            nY.append(ys)
            trads.append(trad)
            pprint.pprint(zip([X['feature'] for X in Xs], ys, trad))

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
    clf, uniques = train_model(Xss, yss, yss, proc, cache)

    for i in xrange(1, epochs):
        sequencer = Sequencer(proc, clf)

        # Generate new dataset
        print "Generating new dataset"
        nX, nY, nT = gen_dataset(Xss, yss, proc, clf, i)
        nXss.extend(nX)
        nyss.extend(nY)
        ntradss.extend(nT)

        # Retrain
        print "Retraining"
        oUnique = uniques
        clf, uniques = train_model(nXss, nyss, ntradss, proc, cache)

        # No new data - exit
        if oUnique == uniques:
            break

    return clf

def cache_key(Xs, ys, trads):
    f = tuple(x['feature'] for x in Xs)
    return (f, tuple(ys), tuple(trads))

def build_training_data(Xss, yss, tradss, proc, cache):
    seen = set()
    n = c = 0
    print "Cache size:", len(cache), len(Xss)
    for cidx, (Xs, ys, trads) in enumerate(izip(Xss, yss, tradss)):
        n += 1

        # Don't judge me
        if cidx in cache:
            v = cache[cidx]
        else:
            ci = cache_key(Xs, ys, trads)
            err_idxs = {i for i, y, t in izip(xrange(len(ys)), ys, trads) if y != t}
            if ci not in cache:
                c += 1
                feats, targets = [], []
                for i in xrange(len(Xs)):
                    verbose = bool(err_idxs)
                    X = proc.transform(Xs, trads, i, verbose=verbose)
                    if verbose:
                        print ys[i]

                    feats.append(X)
                    targets.append(proc.encode_target(ys, i)[0])
                
                cache[ci] = (feats, targets)
                cache[cidx] = (feats, targets)

            v = cache[ci]


        # Positive means unique to this point
        for si, (X, y) in enumerate(izip(*v)):
            skey = (cidx, si)
            if skey in cache:
                unique, hkey = cache[skey]
                if unique:
                    yield X, y
                    seen.add(hkey)
            else:
                k = key(X, y)
                unique = k not in seen
                if unique:
                    seen.add(k)
                    yield X, y

                cache[skey] = (unique, k)

    print "Cache Hit:", (n - c) / float(n)

def train_model(Xss, yss, trads, proc, cache):
    print "Featurizing..."
    tX, tY = [], []
    for X, y in build_training_data(Xss, yss, trads, proc, cache):
        tX.append(X)
        tY.append(y)

    tX, tY = sp.vstack(tX), np.vstack(tY)

    print "Running learner..."
    #clf = SGDClassifier(loss="hinge", penalty="l2", n_iter=30)
    clf = LinearSVC()
    print tX.shape, tY.shape
    clf.fit(tX, tY.ravel())
    return clf, tX.shape[0]

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
    proc = Processor(classes, 3, 1, features=10000, stem=False, ohe=False)

    yss = []
    ryss = []
    for Xs in data:
        ys = [x['output'] for x in Xs]
        yss.append(ys)
        ryss.append([proc.encode_target(ys, i) for i in xrange(len(ys))])

    print "Starting KFolding"
    y_trues, y_preds = [], []
    for train_idx, test_idx in KFold(len(data), 5, random_state=1):
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
    #print classification_report(y_true, y_pred, target_names=proc.labels, digits=4)

    save(outf, seq)

if __name__ == '__main__':
    random.seed(0)
    main(sys.argv[1], sys.argv[2])
