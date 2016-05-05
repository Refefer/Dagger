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

class Dagger(object):
    def __init__(self, Xss, yss, proc):
        self.Xss = Xss
        self.yss = yss
        self.seen_states = set()
        self.state_set = []
        self.proc = proc

    def generate(self, Xs, ys, exp, sequencer):
        """
        Generates a new tradjectory.
        """
        # copy orig seq
        trad = []

        for i in xrange(len(Xs)):
            if random.random() < exp:
                # Oracle
                output = ys[i]
            else:
                # Policy
                output = sequencer._partial_pred(Xs, trad, i)

            trad.append(output)

        return trad

    def gen_dataset(self, clf, epoch):
        sequencer = Sequencer(self.proc, clf)

        # Generate new dataset
        for Xs, ys in izip(self.Xss, self.yss):
            # build tradjectory
            trad = self.generate(Xs, ys, 1 if epoch == 0 else 0, sequencer)
            # If different than the input, add it
            if any(y != t for y, t in izip(ys, trad)):
                self.add_sequence(Xs, ys, trad)
                #pprint.pprint(zip([X['feature'] for X in Xs], ys, trad))

    def add_sequence(self, Xs, ys, trads, force=False):
        for i in xrange(len(Xs)):
            state = self.get_state(Xs, trads, i)
            if state not in self.seen_states or force:
                X = self.proc.transform(Xs, trads, i)
                y = self.proc.encode_target(ys, i)[0]
                self.state_set.append((X, y))
                self.seen_states.add(state)

    def get_state(self, Xs, trad, idx):
        return ' '.join(self.proc.state(Xs, trad, idx))

    def train(self, epochs=30):

        # Add to the initial set
        states = 0
        for Xs, ys in izip(self.Xss, self.yss):
            states += len(ys)
            self.add_sequence(Xs, ys, ys, force=True)

        print len(self.state_set), "unique,", states, "total"
        
        # Initial policy just mimics the expert
        clf = self.train_model()

        for e in xrange(1, epochs):

            # Generate new dataset
            print "Generating new dataset"
            dataSize = len(self.state_set)
            self.gen_dataset(clf, e)

            if dataSize == len(self.state_set):
                break

            # Retrain
            print "Training"
            clf = self.train_model()

        return clf

    def train_model(self):
        print "Featurizing..."
        tX, tY = [], []
        for X, y in self.state_set:
            tX.append(X)
            tY.append(y)

        tX, tY = sp.vstack(tX), np.vstack(tY)

        print "Running learner..."
        #clf = SGDClassifier(loss="hinge", penalty="l2", n_iter=30)
        clf = LinearSVC(penalty='l2')
        print "Samples:", tX.shape[0]
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
    proc = Processor(classes, 3, 1, features=100000, stem=False, ohe=False)

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
        d = Dagger(tr_X, tr_y, proc)
        clf = d.train(10)

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
    d = Dagger(tr_X, tr_y, proc)
    clf = d.train()
    seq = Sequencer(proc, clf)

    print "Testing"
    y_true, y_pred = test(data, ryss, idxs, seq)
    #print classification_report(y_true, y_pred, target_names=proc.labels, digits=4)

    save(outf, seq)

if __name__ == '__main__':
    random.seed(0)
    main(sys.argv[1], sys.argv[2])
