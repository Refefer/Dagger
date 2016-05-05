from itertools import izip
import random
import sys

import numpy as np

import scipy.sparse as sp

from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import DistanceMetric

from sklearn.metrics import classification_report
from sklearn.cross_validation import KFold, ShuffleSplit

from utils import *

class Dagger(object):
    def __init__(self, proc, Xss, yss, valid_set=0.1, validation_set=None):
        self.seen_states = set()
        self.state_set = []
        self.proc = proc
        self.valid_set = 0.1
        self.surr_loss = DistanceMetric.get_metric('hamming')

        if validation_set is None:
            self._split(Xss, yss)
        else:
            self.Xss = Xss
            self.yss = yss
            self.valid_Xss, self.valid_yss = validation_set

    def _split(self, Xss, yss):
        train_Xss, valid_Xss = [], []
        train_yss, valid_yss = [], []
        for Xs, ys in izip(Xss, yss):
            if np.random.binomial(1, 1 - self.valid_set) == 1:
                train_Xss.append(Xs)
                train_yss.append(ys)
            else:
                valid_Xss.append(Xs)
                valid_yss.append(ys)

        self.Xss = train_Xss
        self.yss = train_yss
        self.valid_Xss = valid_Xss
        self.valid_yss = valid_yss

    def generate(self, Xs, ys, exp, sequencer):
        """
        Generates a new tradjectory.
        """
        # copy orig seq
        trad = []

        for i in xrange(len(Xs)):
            if np.random.random() < exp:
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

    def score_policy(self, clf):
        sequencer = Sequencer(self.proc, clf)
        scores = []
        for Xs, ys in izip(self.valid_Xss, self.valid_yss):
            trads = [i for i in sequencer.classify(Xs, raw=True)]
            expected = [self.proc.encode_target(ys, i)[0] for i in xrange(len(ys))]
            scores.append(self.surr_loss.pairwise([trads], [expected]))

        return np.mean(scores)

    def train(self, epochs=30):

        # Add to the initial set
        states = 0
        for Xs, ys in izip(self.Xss, self.yss):
            states += len(ys)
            self.add_sequence(Xs, ys, ys, force=True)

        print len(self.seen_states), "unique,", states, "total"
        
        # Initial policy just mimics the expert
        clf = self.train_model()

        # Get best policy found so far
        bscore, bclf= self.score_policy(clf), clf

        print "Best score seen:",  bscore

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

            print "Scoring"
            score = self.score_policy(clf)
            print "New Policy Score:",  bscore

            if score < bscore:
                bscore = score
                bclf = clf

        return bclf

    def train_model(self):
        print "Featurizing..."
        tX, tY = [], []
        for X, y in self.state_set:
            tX.append(X)
            tY.append(y)

        tX, tY = sp.vstack(tX), np.vstack(tY)

        print "Running learner..."
        clf = SGDClassifier(loss="hinge", penalty="l2", n_iter=10)
        #clf = LinearSVC(penalty="l2", class_weight='auto')
        print "Samples:", tX.shape[0]
        clf.fit(tX, tY.ravel())
        return clf

def subset(Xss, yss, idxs, rs, shuffle=True):
    if shuffle:
        rs.shuffle(idxs)
        print 'Train IDXS', idxs[:10], '...'

    tXss = [Xss[i] for i in idxs]
    tyss = [yss[i] for i in idxs]
    return tXss, tyss

def main(fn, outf):
    print "Reading in dataset"
    data, classes = readDataset(fn)
    print len(data), " sequences found"
    print "Found classes:", sorted(classes)
    proc = Processor(classes, 2, 2, prefix=(1,3), affix=(2,1), hashes=2,
            features=100000, stem=False, ohe=False)

    yss = []
    ryss = []
    for Xs in data:
        ys = [x['output'] for x in Xs]
        yss.append(ys)
        ryss.append([proc.encode_target(ys, i) for i in xrange(len(ys))])

    rs = np.random.RandomState(seed=2016)
    print "Starting KFolding"
    y_trues, y_preds = [], []
    for train_idx, test_idx in KFold(len(data), 8, random_state=1):
        tr_X, tr_y = subset(data, yss, train_idx, rs)
        test_data = subset(data, yss, test_idx, rs, False)

        print "Training"
        d = Dagger(proc, tr_X, tr_y, validation_set=test_data)
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
    tr_X, tr_y = subset(data, yss, idxs, rs)
    d = Dagger(proc, tr_X, tr_y)
    clf = d.train()
    seq = Sequencer(proc, clf)

    save(outf, seq)

if __name__ == '__main__':
    random.seed(0)
    np.random.seed(0)
    main(sys.argv[1], sys.argv[2])
