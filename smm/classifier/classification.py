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
    def __init__(self, training_set, testing_set, name):
        self.logger = logging.getLogger('classifier')
        self.training_set = training_set
        self.testing_set = testing_set
        self.name = name
        self.accuracy = None

    def run_naivebayes(self, pickled):
        if pickled:
            self.logger.info('Start Naive Bayes Classifier.')
            naivebayes_classifier = nltk.classify.NaiveBayesClassifier.train(self.training_set)
            self.logger.info('Finished Naive Bayes Classifier.')
        else:
            self.logger.info('Start loading of Naive Bayes Classifier from ' + config.basepath + config.pickles_path + '/naivebayes_classifier.pickle.')
            open_file = open(config.basepath + config.pickles_path + "/naivebayes_classifier.pickle", "rb")
            naivebayes_classifier = pickle.load(open_file)
            open_file.close()
            self.logger.info('Finished loading of Naive Bayes Classifier.')

        self.logger.info('Started accuracy calculation of Naive Bayes Classifier.')
        self.accuracy = (nltk.classify.accuracy(naivebayes_classifier, self.testing_set)) * 100
        self.logger.info('Finished accuracy calculation of Naive Bayes Classifier.')

        return naivebayes_classifier

    def run_multinomialnb(self, pickled):
        if pickled:
            self.logger.info('Start Multinomial Naive Bayes Classifier.')
            mnb_classifier = SklearnClassifier(MultinomialNB())
            mnb_classifier.train(self.training_set)
            self.logger.info('Finished Multinomial Naive Bayes Classifier.')
        else:
            self.logger.info('Start loading of Multinomial Naive Bayes Classifier from ' + config.basepath + config.pickles_path + '/mnb_classifier.pickle.')
            open_file = open(config.basepath + config.pickles_path + "/mnb_classifier.pickle", "rb")
            mnb_classifier = pickle.load(open_file)
            open_file.close()
            self.logger.info('Finished loading of Multinomial Naive Bayes Classifier.')

        self.logger.info('Started accuracy calculation of Multinomial Naive Bayes Classifier.')
        self.accuracy = (nltk.classify.accuracy(mnb_classifier, self.testing_set)) * 100
        self.logger.info('Finished accuracy calculation of Multinomial Naive Bayes Classifier.')

        return mnb_classifier

    def run_bernoullinb(self, pickled):
        if pickled:
            self.logger.info('Start Bernoulli Naive Bayes Classifier.')
            bernoullinb_classifier = SklearnClassifier(BernoulliNB())
            bernoullinb_classifier.train(self.training_set)
            self.logger.info('Finished Bernoulli Naive Bayes Classifier.')
        else:
            self.logger.info('Start loading of Bernoulli Naive Bayes Classifier from ' + config.basepath + config.pickles_path + '/bernoullinb_classifier.pickle.')
            open_file = open(config.basepath + config.pickles_path + "/bernoullinb_classifier.pickle", "rb")
            bernoullinb_classifier = pickle.load(open_file)
            open_file.close()
            self.logger.info('Finished loading of Bernoulli Naive Bayes Classifier.')

        self.logger.info('Started accuracy calculation of Bernoulli Naive Bayes Classifier.')
        self.accuracy = (nltk.classify.accuracy(bernoullinb_classifier, self.testing_set)) * 100
        self.logger.info('Finished accuracy calculation of Bernoulli Naive Bayes Classifier.')

        return bernoullinb_classifier

    def run_logisticregression(self, pickled):
        if pickled:
            self.logger.info('Start Logistic Regression Classifier.')
            logisticregression_classifier = SklearnClassifier(LogisticRegression())
            logisticregression_classifier.train(self.training_set)
            self.logger.info('Finished Logistic Regression Classifier.')
        else:
            self.logger.info('Start loading of Logistic Regression Classifier from ' + config.basepath + config.pickles_path + '/logisticregression_classifier.pickle.')
            open_file = open(config.basepath + config.pickles_path + "/logisticregression_classifier.pickle", "rb")
            logisticregression_classifier = pickle.load(open_file)
            open_file.close()
            self.logger.info('Finished loading of Logistic Regression Classifier.')

        self.logger.info('Started accuracy calculation of Logistic Regression Classifier.')
        self.accuracy = (nltk.classify.accuracy(logisticregression_classifier, self.testing_set)) * 100
        self.logger.info('Finished accuracy calculation of Logistic Regression Classifier.')

        return logisticregression_classifier

    def run_sgd(self, pickled):
        if pickled:
            self.logger.info('Start Stochastic Gradient Descent Classifier.')
            sgd_classifier = SklearnClassifier(SGDClassifier())
            sgd_classifier.train(self.training_set)
            self.logger.info('Finished Stochastic Gradient Descent Classifier.')
        else:
            self.logger.info('Start loading of Stochastic Gradient Descent Classifier from ' + config.basepath + config.pickles_path + '/sgd_classifier.pickle.')
            open_file = open(config.basepath + config.pickles_path + "/sgd_classifier.pickle", "rb")
            sgd_classifier = pickle.load(open_file)
            open_file.close()
            self.logger.info('Finished loading of Stochastic Gradient Descent Classifier.')

        self.logger.info('Started accuracy calculation of Stochastic Gradient Descent Classifier.')
        self.accuracy = (nltk.classify.accuracy(sgd_classifier, self.testing_set)) * 100
        self.logger.info('Finished accuracy calculation of Stochastic Gradient Descent Classifier.')

        return sgd_classifier

    def run_linearsvc(self, pickled):
        if pickled:
            self.logger.info('Start Linear Support Vector Classifier.')
            linearsvc_classifier = SklearnClassifier(LinearSVC())
            linearsvc_classifier.train(self.training_set)
            self.logger.info('Finished Linear Support Vector Classifier.')
        else:
            self.logger.info('Start loading of Linear Support Vector Classifier ' + config.basepath + config.pickles_path + '/linearsvc_classifier.pickle.')
            open_file = open(config.basepath + config.pickles_path + "/linearsvc_classifier.pickle", "rb")
            linearsvc_classifier = pickle.load(open_file)
            open_file.close()
            self.logger.info('Finished loading of Linear Support Vector Classifier.')

        self.logger.info('Started accuracy calculation of Linear Support Vector Classifier.')
        self.accuracy = (nltk.classify.accuracy(linearsvc_classifier, self.testing_set)) * 100
        self.logger.info('Finished accuracy calculation of Linear Support Vector Classifier.')

        return linearsvc_classifier

    def run_nusvc(self, pickled):
        if pickled:
            self.logger.info('Start Nu Support Vector Classifier.')
            nusvc_classifier = SklearnClassifier(NuSVC())
            nusvc_classifier.train(self.training_set)
            self.logger.info('Finished Nu Support Vector Classifier.')
        else:
            self.logger.info('Start loading of Nu Support Vector Classifier ' + config.basepath + config.pickles_path + '/nusvc_classifier.pickle.')
            open_file = open(config.basepath + config.pickles_path + "/nusvc_classifier.pickle", "rb")
            nusvc_classifier = pickle.load(open_file)
            open_file.close()
            self.logger.info('Finished loading of Nu Support Vector Classifier.')

        self.logger.info('Started accuracy calculation of Nu Support Vector Classifier.')
        self.accuracy = (nltk.classify.accuracy(nusvc_classifier, self.testing_set)) * 100
        self.logger.info('Finished accuracy calculation of Nu Support Vector Classifier.')

        return nusvc_classifier



class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        #self.logger = logging.getLogger('classifier')
        self._classifiers = classifiers
        self.accuracy = None

    def classify(self, features):
        #self.logger.info('Start Voted Classifier.')
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        #self.logger.info('Finished Voted Classifier.')
        return mode(votes)

    def confidence(self, features):
        #self.logger.info('Start calculation of Voted Classifier accuracy.')
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        #self.logger.info('Finished calculation of Voted Classifier accuracy.')
        return conf