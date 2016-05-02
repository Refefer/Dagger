import sys
import numpy as np
import cPickle

from sklearn.feature_extraction import FeatureHasher

class Processor(object):

    def __init__(self, classes, previous=1, following=1, ohe=True):
        self.previous = previous
        self.following = following 
        self.fh = FeatureHasher(15000, input_type='string', dtype='float32')
        self.labels = list(classes)
        self.classes = {c: i for i, c in enumerate(self.labels)}
        self.tlabels = {i: c for i, c in enumerate(self.labels)}
        self.n_classes = len(classes)
        self._nident = np.identity(self.n_classes, 'float32')
        self.to_ohe = ohe

    def ohe(self, y):
        return self._nident[self.classes[y]]

    def transform(self, sequence, idx):
        features = []
        # Print previous featues
        for i, pidx in enumerate(xrange(max(idx - self.previous, 0), idx)):
            f = sequence[pidx]
            features.append('p_feat_%s:%s' % (i, f['feature']))
            features.append('p_pred_%s:%s' % (i, f['output']))

        # Print following featues
        until = min(idx + self.following + 1, len(sequence))
        for i, fidx in enumerate(xrange(idx + 1, until)):
            f = sequence[fidx]
            features.append('f_feat_%s:%s' % (i, f['feature']))

        # Print current features
        cur_f = sequence[idx]
        features.append('feat:'  + cur_f['feature'])

        return self.fh.transform([features])

    def encode_target(self, sequence, idx):
        cur_f = sequence[idx]
        if self.to_ohe:
            return self.ohe(cur_f['output'])

        return [self.classes[cur_f['output']]]

class Sequencer(object):
    def __init__(self, processor, policy):
        self.processor = processor
        self.policy = policy

    def classify(self, sequence):
        features = [{"feature": s['feature']} for s in sequence]
        outputs = []
        for i in xrange(len(features)):
            feats = self.processor.transform(features, i)
            output = self.policy.predict(feats)
            outputs.append(output[0])
            features[i]['output'] = self.processor.tlabels[output[0]]

        return outputs

    def pclassify(self, sequence):
        return [self.processor.tlabels[i] 
                for i in self.classify(sequence)]



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
                feature, output = line.split()
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

    return sequences, classes

def save(path, cls):
    with file(path, 'w') as out:
        cPickle.dump(cls, out)

def load(path):
    with file(path) as f:
        return cPickle.load(f)
