import sys
import numpy as np
import cPickle

import nltk

from sklearn.feature_extraction import FeatureHasher

class Policy(object):
    def predict(self, data, sequence, i):
        raise NotImplementedError()

class ExpPolicy(Policy):

    def __init__(self, oracle, ppolicy, blend):
        self.oracle = oracle
        self.ppolicy = ppolicy
        self.blend = blend

    def predict(self, feats, y):
        if random.random() < self.blend:
            return y

        return self.ppolicy.predict(feats)

class Processor(object):

    def __init__(self, classes, previous=1, following=1, features=15000, stem=True, ohe=True):
        self.previous = previous
        self.following = following 
        self.stem = stem
        self.es = nltk.stem.snowball.EnglishStemmer()
        self.fh = FeatureHasher(features, input_type='string', dtype='float32')
        self.labels = list(classes)
        self.classes = {c: i for i, c in enumerate(self.labels)}
        self.tlabels = {i: c for c, i in self.classes.iteritems()}
        self.n_classes = len(classes)
        self._nident = np.identity(self.n_classes, 'float32')
        self.to_ohe = ohe

    def ohe(self, y):
        return self._nident[self.classes[y]]

    def _get_feat(self, prefix, f):
        ret = [prefix % f['feature']]
        if self.stem:
            try:
                stem = self.es.stem(feat)
            except Exception:
                stem = ''

            ret.append('s' + prefix % stem)

        return ret

    def transform(self, sequence, trad, idx, verbose=False):
        features = []

        # Print previous featues
        start = {'feature': '_START_'}
        for i, pidx in enumerate(xrange(idx - self.previous, idx)):
            if pidx < 0:
                f, o = start, 'None'
            else:
                f, o = sequence[pidx], trad[pidx]

            features.extend(self._get_feat('p_feat_%s:%%s' % i,  f))
            features.append('p_pred_%s:%s' % (i, o))

        # Print following featues
        until = idx + self.following + 1
        for i, fidx in enumerate(xrange(idx + 1, until)):
            f = {'feature': "_END_"} if fidx >= len(sequence) else sequence[fidx]
            features.extend(self._get_feat('f_feat_%s:%%s' % i,  f))

        # Print current features
        features.extend(self._get_feat('feat:%s', sequence[idx]))

        if verbose:
            print features

        return self.fh.transform([features])

    def encode_target(self, ys, idx):
        y = ys[idx]
        if self.to_ohe:
            return self.ohe(y)

        return [self.classes[y]]

class Sequencer(object):
    def __init__(self, processor, policy):
        self.processor = processor
        self.policy = policy

    def classify(self, sequence, raw=False):
        outputs = []
        for i in xrange(len(sequence)):
            outputs.append(self._partial_pred(sequence, outputs, i))

        if raw:
            return [self.processor.classes[o] for o in outputs]

        return outputs

    def _partial_pred(self, features, trad, i):
        feats = self.processor.transform(features, trad, i)
        output = self.policy.predict(feats)
        return self.processor.tlabels[output[0]]

def readDataset(fn):
    if fn == '-':
        f = sys.stdin
    else:
        f = file(fn)

    def run(f):
        sequences = []
        classes = set()
        sequence = []
        for line in f:
            line = line.strip()
            if line:
                pieces = line.split()
                feature, output = '/'.join(pieces[:-1]), pieces[-1]
                classes.add(output)
                sequence.append(dict(feature=feature, output=output))

            elif sequence:
                sequences.append(sequence)
                sequence = []

        if sequence:
            sequences.append(sequence)

        return sequences, classes

    try:
        return run(f)
    finally:
        if f != '-':
            f.close()

def save(path, cls):
    with file(path, 'w') as out:
        cPickle.dump(cls, out)

def load(path):
    with file(path) as f:
        return cPickle.load(f)

def test(Xss, yss, test_idx, seq):
    y_true, y_pred = [], []
    for idx in test_idx:
        y_true.extend(y for ys in yss[idx] for y in ys)
        preds = seq.classify(Xss[idx], raw=True)
        y_pred.extend(preds)

    return y_true, y_pred

