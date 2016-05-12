__author__ = 'ma'
import nltk
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.classify import ClassifierI

from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC, NuSVC

from smm import config

from statistics import mode
import pickle
import logging

class Classifier(object):
    def __init__(self, training_set, testing_set):
        self.logger = logging.getLogger('classifier')
        self.training_set = training_set
        self.testing_set = testing_set

    def run_naivebayes(self, pickled):
        if pickled:
            self.logger.info('Start Naive Bayes Classifier.')
            naivebayes_classifier = nltk.NaiveBayesClassifier.train(self.training_set)
            self.logger.info('Finished Naive Bayes Classifier.')
        else:
            self.logger.info('Start loading of Naive Bayes Classifier from ' + config.basepath + config.pickles_path + '/naivebayes_classifier.pickle.')
            open_file = open(config.basepath + config.pickles_path + "/naivebayes_classifier.pickle", "rb")
            naivebayes_classifier = pickle.load(open_file)
            open_file.close()
            self.logger.info('Finished loading of Naive Bayes Classifier.')

        #naivebayes_accuracy = (nltk.classify.accuracy(naivebayes_classifier, self.testing_set)) * 100
        #self.logger.info("Naive Bayes Classifier accuracy is '{0}'.".format(naivebayes_accuracy))

        return naivebayes_classifier

    def run_multinomialnb(self, pickled):
        if pickled:
            self.logger.info('Start Multinomial NB Classifier.')
            mnb_classifier = SklearnClassifier(MultinomialNB())
            mnb_classifier.train(self.training_set)
            self.logger.info('Finished MultinomialNB Classifier.')
        else:
            self.logger.info('Start loading of MultinomialNB Classifier from ' + config.basepath + config.pickles_path + '/mnb_classifier.pickle.')
            open_file = open(config.basepath + config.pickles_path + "/mnb_classifier.pickle", "rb")
            mnb_classifier = pickle.load(open_file)
            open_file.close()
            self.logger.info('Finished loading of MultinomialNB Classifier.')

        #mnb_accuracy = (nltk.classify.accuracy(mnb_classifier, self.testing_set)) * 100
        #self.logger.info("MultinomialNB Classifier accuracy is '{0}'.".format(mnb_accuracy))

        return mnb_classifier

    def run_bernoullinb(self, pickled):
        if pickled:
            self.logger.info('Start BernoulliNB Classifier.')
            bernoullinb_classifier = SklearnClassifier(BernoulliNB())
            bernoullinb_classifier.train(self.training_set)
            self.logger.info('Finished BernoulliNB Classifier.')
        else:
            self.logger.info('Start loading of BernoulliNB Classifier from ' + config.basepath + config.pickles_path + '/bernoullinb_classifier.pickle.')
            open_file = open(config.basepath + config.pickles_path + "/bernoullinb_classifier.pickle", "rb")
            bernoullinb_classifier = pickle.load(open_file)
            open_file.close()
            self.logger.info('Finished loading of BernoulliNB Classifier.')

        #bernoullinb_accuracy = (nltk.classify.accuracy(bernoullinb_classifier, self.testing_set)) * 100
        #self.logger.info("BernoulliNB Classifier accuracy is '{0}'.".format(bernoullinb_accuracy))

        return bernoullinb_classifier

    def run_logisticregression(self, pickled):
        if pickled:
            self.logger.info('Start LogisticRegression Classifier.')
            logisticregression_classifier = SklearnClassifier(LogisticRegression())
            logisticregression_classifier.train(self.training_set)
            self.logger.info('Finished LogisticRegression Classifier.')
        else:
            self.logger.info('Start loading of LogisticRegression Classifier from ' + config.basepath + config.pickles_path + '/logisticregression_classifier.pickle.')
            open_file = open(config.basepath + config.pickles_path + "/logisticregression_classifier.pickle", "rb")
            logisticregression_classifier = pickle.load(open_file)
            open_file.close()
            self.logger.info('Finished loading of LogisticRegression Classifier.')

        #logisticregression_accuracy = (nltk.classify.accuracy(logisticregression_classifier, self.testing_set)) * 100
        #self.logger.info("LogisticRegression Classifier accuracy is '{0}'.".format(logisticregression_accuracy))

        return logisticregression_classifier

    def run_sgd(self, pickled):
        if pickled:
            self.logger.info('Start SGD Classifier.')
            sgd_classifier = SklearnClassifier(SGDClassifier())
            sgd_classifier.train(self.training_set)
            self.logger.info('Finished SGD Classifier.')
        else:
            self.logger.info('Start loading of SGD Classifier from ' + config.basepath + config.pickles_path + '/sgd_classifier.pickle.')
            open_file = open(config.basepath + config.pickles_path + "/sgd_classifier.pickle", "rb")
            sgd_classifier = pickle.load(open_file)
            open_file.close()
            self.logger.info('Finished loading of SGD Classifier.')

        #sgd_accuracy = (nltk.classify.accuracy(sgd_classifier, self.testing_set)) * 100
        #self.logger.info("SGD Classifier accuracy is '{0}'.".format(sgd_accuracy))

        return sgd_classifier

    def run_linearsvc(self, pickled):
        if pickled:
            self.logger.info('Start LinearSVC Classifier.')
            linearsvc_classifier = SklearnClassifier(LinearSVC())
            linearsvc_classifier.train(self.training_set)
            self.logger.info('Finished LinearSVC Classifier.')
        else:
            self.logger.info('Start loading of LinearSVC Classifier ' + config.basepath + config.pickles_path + '/linearsvc_classifier.pickle.')
            open_file = open(config.basepath + config.pickles_path + "/linearsvc_classifier.pickle", "rb")
            linearsvc_classifier = pickle.load(open_file)
            open_file.close()
            self.logger.info('Finished loading of LinearSVC Classifier.')


        #linearsvc_accuracy = (nltk.classify.accuracy(linearsvc_classifier, self.testing_set)) * 100
        #self.logger.info("LinearSVC Classifier accuracy is '{0}'.".format(linearsvc_accuracy))

        return linearsvc_classifier

    def run_nusvc(self, pickled):
        if pickled:
            self.logger.info('Start NuSVC Classifier.')
            nusvc_classifier = SklearnClassifier(NuSVC())
            nusvc_classifier.train(self.training_set)
            self.logger.info('Finished NuSVC Classifier.')
        else:
            self.logger.info('Start loading of NuSVC Classifier  ' + config.basepath + config.pickles_path + '/nusvc_classifier.pickle.')
            open_file = open(config.basepath + config.pickles_path + "/nusvc_classifier.pickle", "rb")
            nusvc_classifier = pickle.load(open_file)
            open_file.close()
            self.logger.info('Finished loading of NuSVC Classifier.')

        #nusvc_accuracy = (nltk.classify.accuracy(nusvc_classifier, self.testing_set)) * 100
        #self.logger.info("NuSVC Classifier accuracy is '{0}'.".format(nusvc_accuracy))

        return nusvc_classifier



class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self.logger = logging.getLogger('classifier')
        self._classifiers = classifiers

    def classify(self, features):
        self.logger.info('Start Voted Classifier.')
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        self.logger.info('Finished Voted Classifier.')
        return mode(votes)

    def confidence(self, features):
        self.logger.info('Start calculation of Voted Classifier accuracy.')
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        self.logger.info('Finished calculation of Voted Classifier accuracy.')
        return conf