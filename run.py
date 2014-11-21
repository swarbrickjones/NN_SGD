import sys
from dnn_all import *

def add_fit_and_score(class_to_chg):
    """ Mutates a class to add the fit() and score() functions to a NeuralNet.
    """
    from types import MethodType
    def fit(self, x_train, y_train, x_dev=None, y_dev=None,
            max_epochs=100, early_stopping=True, split_ratio=0.1,
            method='adadelta', verbose=False, plot=False):
        """
        Fits the neural network to `x_train` and `y_train`. 
        If x_dev nor y_dev are not given, it will do a `split_ratio` cross-
        validation split on `x_train` and `y_train` (for early stopping).
        """
        import time, copy
        if x_dev == None or y_dev == None:
            from sklearn.cross_validation import train_test_split
            x_train, x_dev, y_train, y_dev = train_test_split(x_train, y_train,
                    test_size=split_ratio, random_state=42)
        if method == 'sgd':
            train_fn = self.get_SGD_trainer()
        elif method == 'adagrad':
            train_fn = self.get_adagrad_trainer()
        elif method == 'adadelta':
            train_fn = self.get_adadelta_trainer()
        train_set_iterator = DatasetMiniBatchIterator(x_train, y_train)
        dev_set_iterator = DatasetMiniBatchIterator(x_dev, y_dev)
        train_scoref = self.score_classif(train_set_iterator)
        dev_scoref = self.score_classif(dev_set_iterator)
        best_dev_loss = numpy.inf
        epoch = 0
        # TODO early stopping (not just cross val, also stop training)
        if plot:
            verbose = True
            self._costs = []
            self._train_errors = []
            self._dev_errors = []
            self._updates = []

        while epoch < max_epochs:
            if not verbose:
                sys.stdout.write("\r%0.2f%%" % (epoch * 100./ max_epochs))
                sys.stdout.flush()
            avg_costs = []
            timer = time.time()
            for x, y in train_set_iterator:
                if method == 'sgd' or method == 'adagrad':
                    avg_cost = train_fn(x, y, lr=0.00999999977648)  # TODO: you have to
                                                         # play with this
                                                         # learning rate
                                                         # (dataset dependent)
                elif method == 'adadelta':
                    avg_cost = train_fn(x, y)
                if type(avg_cost) == list:
                    avg_costs.append(avg_cost[0])
                else:
                    avg_costs.append(avg_cost)
            if verbose:
                mean_costs = numpy.mean(avg_costs)
                mean_train_errors = numpy.mean(train_scoref())
                print('  epoch %i took %f seconds' %
                      (epoch, time.time() - timer))
                print('  epoch %i, avg costs %f' %
                      (epoch, mean_costs))
                print('  epoch %i, training error %f' %
                      (epoch, mean_train_errors))
                if plot:
                    self._costs.append(mean_costs)
                    self._train_errors.append(mean_train_errors)
            dev_errors = numpy.mean(dev_scoref())
            if plot:
                self._dev_errors.append(dev_errors)
            if dev_errors < best_dev_loss:
                best_dev_loss = dev_errors
                best_params = copy.deepcopy(self.params)
                if verbose:
                    print('!!!  epoch %i, validation error of best model %f' %
                          (epoch, dev_errors))
            epoch += 1
        if not verbose:
            print("")
        for i, param in enumerate(best_params):
            self.params[i] = param

    def score(self, x, y):
        """ error rates """
        iterator = DatasetMiniBatchIterator(x, y)
        scoref = self.score_classif(iterator)
        return numpy.mean(scoref())

    class_to_chg.fit = MethodType(fit, None, class_to_chg)
    class_to_chg.score = MethodType(score, None, class_to_chg)


if __name__ == "__main__":
    add_fit_and_score(DropoutNet)
    add_fit_and_score(RegularizedNet)

    def nudge_dataset(X, Y):
        """
        This produces a dataset 5 times bigger than the original one,
        by moving the 8x8 images in X around by 1px to left, right, down, up
        """
        from scipy.ndimage import convolve
        direction_vectors = [
            [[0, 1, 0],
             [0, 0, 0],
             [0, 0, 0]],
            [[0, 0, 0],
             [1, 0, 0],
             [0, 0, 0]],
            [[0, 0, 0],
             [0, 0, 1],
             [0, 0, 0]],
            [[0, 0, 0],
             [0, 0, 0],
             [0, 1, 0]]]
        shift = lambda x, w: convolve(x.reshape((8, 8)), mode='constant',
                                      weights=w).ravel()
        X = numpy.concatenate([X] +
                              [numpy.apply_along_axis(shift, 1, X, vector)
                                  for vector in direction_vectors])
        Y = numpy.concatenate([Y for _ in range(5)], axis=0)
        return X, Y

    from sklearn import datasets, svm, naive_bayes
    from sklearn import cross_validation, preprocessing
    MNIST = True  # MNIST dataset
    DIGITS = False  # digits dataset
    FACES = True  # faces dataset
    TWENTYNEWSGROUPS = False  # 20 newgroups dataset
    VERBOSE = True  # prints evolution of the loss/accuracy during the fitting
    SCALE = True  # scale the dataset
    PLOT = True  # plot losses and accuracies

    def train_models(x_train, y_train, x_test, y_test, n_features, n_outs,
            use_dropout=True, n_epochs=100, numpy_rng=None,
            svms=False, nb=False, deepnn=True, name=''):
        if svms:
            print("Linear SVM")
            classifier = svm.SVC(gamma=0.001)
            print(classifier)
            classifier.fit(x_train, y_train)
            print("score: %f" % classifier.score(x_test, y_test))

            print("RBF-kernel SVM")
            classifier = svm.SVC(kernel='rbf', class_weight='auto')
            print(classifier)
            classifier.fit(x_train, y_train)
            print("score: %f" % classifier.score(x_test, y_test))

        if nb:
            print("Multinomial Naive Bayes")
            classifier = naive_bayes.MultinomialNB()
            print(classifier)
            classifier.fit(x_train, y_train)
            print("score: %f" % classifier.score(x_test, y_test))

        if deepnn:
            import warnings
            warnings.filterwarnings("ignore")  # TODO remove

            if use_dropout:
                #n_epochs *= 4  TODO
                pass

            def new_dnn(dropout=False):
                if dropout:
                    print("Dropout DNN")
                    return DropoutNet(numpy_rng=numpy_rng, n_ins=n_features,
                        layers_types=[ReLU, ReLU, LogisticRegression],
                        layers_sizes=[200, 200],
                        dropout_rates=[0.2, 0.5, 0.5],
                        # TODO if you have a big enough GPU, use these:
                        #layers_types=[ReLU, ReLU, ReLU, ReLU, LogisticRegression],
                        #layers_sizes=[2000, 2000, 2000, 2000],
                        #dropout_rates=[0.2, 0.5, 0.5, 0.5, 0.5],
                        n_outs=n_outs,
                        max_norm=4.,
                        fast_drop=True,
                        debugprint=0)
                else:
                    print("Simple (regularized) DNN")
                    return RegularizedNet(numpy_rng=numpy_rng, n_ins=n_features,
                        layers_types=[ReLU, ReLU, LogisticRegression],
                        layers_sizes=[200, 200],
                        n_outs=n_outs,
                        #L1_reg=0.001/x_train.shape[0],
                        #L2_reg=0.001/x_train.shape[0],
                        L1_reg=0.,
                        L2_reg=1./x_train.shape[0],
                        debugprint=0)

            import matplotlib.pyplot as plt
            plt.figure()
            ax1 = plt.subplot(221)
            ax2 = plt.subplot(222)
            ax3 = plt.subplot(223)
            ax4 = plt.subplot(224)  # TODO plot the updates of the weights
            methods = ['sgd', 'adagrad', 'adadelta']
            #methods = ['adadelta'] TODO if you want "good" results asap
            for method in methods:
                dnn = new_dnn(use_dropout)
                print dnn, "using", method
                dnn.fit(x_train, y_train, max_epochs=n_epochs, method=method, verbose=VERBOSE, plot=PLOT)
                test_error = dnn.score(x_test, y_test)
                print("score: %f" % (1. - test_error))
                ax1.plot(numpy.log10(dnn._costs), label=method)
                ax2.plot(numpy.log10(dnn._train_errors), label=method)
                ax3.plot(numpy.log10(dnn._dev_errors), label=method)
                #ax2.plot(dnn._train_errors, label=method)
                #ax3.plot(dnn._dev_errors, label=method)
                ax4.plot([test_error for _ in range(10)], label=method)
            ax1.set_xlabel('epoch')
            ax1.set_ylabel('cost (log10)')
            ax2.set_xlabel('epoch')
            ax2.set_ylabel('train error')
            ax3.set_xlabel('epoch')
            ax3.set_ylabel('dev error')
            ax4.set_ylabel('test error')
            plt.legend()
            plt.savefig('training_' + name + '.png')


    if MNIST:
        from sklearn.datasets import fetch_mldata
        mnist = fetch_mldata('MNIST original')
        X = numpy.asarray(mnist.data, dtype='float32')
        if SCALE:
            #X = preprocessing.scale(X)
            X /= 255.
        y = numpy.asarray(mnist.target, dtype='int32')
        print("Total dataset size:")
        print("n samples: %d" % X.shape[0])
        print("n features: %d" % X.shape[1])
        print("n classes: %d" % len(set(y)))
        x_train, x_test, y_train, y_test = cross_validation.train_test_split(
                    X, y, test_size=0.2, random_state=42)

        train_models(x_train, y_train, x_test, y_test, X.shape[1],
                     len(set(y)), numpy_rng=numpy.random.RandomState(123),
                     name='MNIST')


    if DIGITS:
        digits = datasets.load_digits()
        data = numpy.asarray(digits.data, dtype='float32')
        target = numpy.asarray(digits.target, dtype='int32')
        nudged_x, nudged_y = nudge_dataset(data, target)
        if SCALE:
            nudged_x = preprocessing.scale(nudged_x)
        x_train, x_test, y_train, y_test = cross_validation.train_test_split(
                nudged_x, nudged_y, test_size=0.2, random_state=42)
        train_models(x_train, y_train, x_test, y_test, nudged_x.shape[1],
                     len(set(target)), numpy_rng=numpy.random.RandomState(123),
                     name='digits')

    if FACES:
        import logging
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s %(message)s')
        lfw_people = datasets.fetch_lfw_people(min_faces_per_person=70,
                                               resize=0.4)
        X = numpy.asarray(lfw_people.data, dtype='float32')
        if SCALE:
            X = preprocessing.scale(X)
        y = numpy.asarray(lfw_people.target, dtype='int32')
        target_names = lfw_people.target_names
        print("Total dataset size:")
        print("n samples: %d" % X.shape[0])
        print("n features: %d" % X.shape[1])
        print("n classes: %d" % target_names.shape[0])
        x_train, x_test, y_train, y_test = cross_validation.train_test_split(
                    X, y, test_size=0.2, random_state=42)

        train_models(x_train, y_train, x_test, y_test, X.shape[1],
                     len(set(y)), numpy_rng=numpy.random.RandomState(123),
                     name='faces')

    if TWENTYNEWSGROUPS:
        from sklearn.feature_extraction.text import TfidfVectorizer
        newsgroups_train = datasets.fetch_20newsgroups(subset='train')
        vectorizer = TfidfVectorizer(encoding='latin-1', max_features=10000)
        #vectorizer = HashingVectorizer(encoding='latin-1')
        x_train = vectorizer.fit_transform(newsgroups_train.data)
        x_train = numpy.asarray(x_train.todense(), dtype='float32')
        y_train = numpy.asarray(newsgroups_train.target, dtype='int32')
        newsgroups_test = datasets.fetch_20newsgroups(subset='test')
        x_test = vectorizer.transform(newsgroups_test.data)
        x_test = numpy.asarray(x_test.todense(), dtype='float32')
        y_test = numpy.asarray(newsgroups_test.target, dtype='int32')
        train_models(x_train, y_train, x_test, y_test, x_train.shape[1],
                     len(set(y_train)),
                     numpy_rng=numpy.random.RandomState(123),
                     svms=False, nb=True, deepnn=True,
                     name='20newsgroups')