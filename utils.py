import sys
import numpy as np
import random
import pickle

import nltk
import scipy.sparse as spa

from sklearn.feature_extraction import FeatureHasher
import sklearn.feature_extraction._hashing as hasher 

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

    def __init__(self, classes, previous=1, following=1, 
            features=15000, prefix=(), affix=(), 
            hashes=1, stem=True, ohe=True):
        self.hashes = hashes
        self.previous = previous
        self.following = following 
        self.prefix = prefix
        self.affix = affix
        self.stem = stem
        self.es = nltk.stem.snowball.EnglishStemmer()
        self.features = features
        self.fh = FeatureHasher(features, input_type='string', dtype='float32')
        self.labels = list(classes)
        self.classes = {c: i for i, c in enumerate(self.labels)}
        self.tlabels = {i: c for c, i in self.classes.items()}
        self.n_classes = len(classes)
        self._nident = np.identity(self.n_classes, 'float32')
        self.to_ohe = ohe

    def ohe(self, y):
        return self._nident[self.classes[y]]

    def _get_feat(self, prefix, f):
        feat = f['feature']
        ret = [prefix % feat]
        if self.stem:
            stem = self.es.stem(feat)

            ret.append('s' + prefix % stem)

        pprefix = 'p_' + prefix
        for s in self.prefix:
            ret.append(pprefix % feat[:s])

        affix = 'a_' + prefix
        for s in self.prefix:
            ret.append(affix % feat[-s:])

        return ret

    def transform(self, sequence, trad, idx, verbose=False):
        features = self.state(sequence, trad, idx)

        if verbose:
            print(features)

        #return self.fh.transform(features)
        return self._hash(features)

    def _hash(self, features):
        indices, indptr, values = \
            hasher.transform([[(x,1) for x in features]], self.features, 'float32')

        X = spa.csr_matrix((values, indices, indptr), dtype='float32',
                          shape=(1, self.features))
        X.sum_duplicates()
        return X

    def state(self, Xs, trad, idx):
        features = []

        # Print previous featues
        start = {'feature': '_START_'}
        for i, pidx in enumerate(range(idx - self.previous, idx)):
            if pidx < 0:
                f, o = start, 'None'
            else:
                f, o = Xs[pidx], trad[pidx]

            features.extend(self._get_feat('p_feat_%s:%%s' % i,  f))
            features.append('p_pred_%s:%s' % (i, o))

        # Print following featues
        until = idx + self.following + 1
        for i, fidx in enumerate(range(idx + 1, until)):
            f = {'feature': "_END_"} if fidx >= len(Xs) else Xs[fidx]
            features.extend(self._get_feat('f_feat_%s:%%s' % i,  f))

        # Print current features
        features.extend(self._get_feat('feat:%s', Xs[idx]))

        if self.hashes > 1:
            nhashes = features[:]
            for i in range(1, self.hashes):
                p = '!' * i
                nhashes.extend(p+s for s in features)

            hashes = nhashes
        return features

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
        for i in range(len(sequence)):
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
        f = open(fn)

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

def save(path, obj):
    with open(path, 'wb') as outFile:
        pickle.dump(obj, outFile)

def load(path):
    with open(path) as f:
        return pickle.load(f)

def test(Xss, yss, test_idx, seq):
    y_true, y_pred = [], []
    nerrors = 0.0
    perrs = 0.0
    for idx in test_idx:
        y_true.extend(y for ys in yss[idx] for y in ys)
        preds = seq.classify(Xss[idx], raw=True)
        
        # Calculate errors per sequence
        errors = sum(y[0] != t for y, t in zip(yss[idx], preds))
        if errors > 0:
            nerrors += 1
            perrs += errors
            
        y_pred.extend(preds)

    print("Phrases with errors:", nerrors)
    print("Total Errors:", perrs)
    print("Errors per bad seq:", perrs / nerrors if nerrors else 0, nerrors)
    print("Phrase Accuracy:", (len(test_idx) - nerrors) / len(test_idx))
    return y_true, y_pred

