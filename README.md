Dagger - Dataset Aggregation for Learning to Search
===

A learning implementation of "A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning" (https://arxiv.org/abs/1011.0686).

Requirements
===

numpy
scikit-learn

TODO
===

A lot of this could be refactored to make it easier to plug in both new Oracles and different learning algorithms.  It shouldn't be particularly hard to extend this to other learning tasks either.

Data
===

While Dagger can be used for any type of structured learning task, this implementation was mostly intended for POS.  However, the amount of time to modify it for other tasks should be fairly trivial.

The format is pretty simple. batches of `<feature> <class>`, separated by a carriage return:

    the article
    quick adjective
    brown adjective
    fox noun
    jumped verb
    over preposition
    the the
    lazy adjective
    dog noun

    i pronoun
    love verb
    this determiner
    time noun
    of preposition
    year noun

Running
===

You can train a POS tagger fairly easily:

    python dagger.py <train_file> <model>

HYou can compare it to a baseline easily as well:

    python baseline.py <train_file> <model>

