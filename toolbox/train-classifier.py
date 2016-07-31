#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK

__author__ = 'gx & ma'

import argparse
import argcomplete
import sys
import os
import pickle

from smm import models
from smm.classifier import classification
from smm import config

import random
import logging

import nltk
from nltk.corpus import stopwords
from nltk.corpus import CategorizedPlaintextCorpusReader
from nltk.tokenize import word_tokenize

parser = argparse.ArgumentParser(description='classify collected reviews', usage='python train-classification.py classifier 10000')
parser.add_argument('name', help='classifier name - must be unique')
parser.add_argument('size', type=int, help='corpus size - how much documents to classify')
parser.add_argument('pickling', help='store results not only in database but also as pickle files', choices=['yes', 'no'])
parser.add_argument('-t', '--type', help='classifier type', default='voteclassifier')
args = parser.parse_args()

argcomplete.autocomplete(parser)

# Define logger
logger = logging.getLogger('train-classifier')

# Pickle
if args.pickling == 'yes':
    pickling = True
else:
    pickling = False

# Load the reviews
pos_corpus = CategorizedPlaintextCorpusReader(config.basepath + config.reviews_path + '/pos/', r'(?!\.).*\.txt', cat_pattern=r'(pos)/.*', encoding='ascii')
neg_corpus = CategorizedPlaintextCorpusReader(config.basepath + config.reviews_path + '/neg/', r'(?!\.).*\.txt', cat_pattern=r'(neg)/.*', encoding='ascii')

pos_reviews = pos_corpus.raw()
neg_reviews = neg_corpus.raw()

# Define arrays for words and documents
all_words = []
documents = []

# j is adject, r is adverb, and v is verb
allowed_word_types = ['J', 'R', 'V']

# Get English stopwords
stops = set(stopwords.words('english'))

# Annotate positive documents
logger.info('Start creating documents out of positive reviews.')
for p in pos_reviews.split('\n'):
    documents.append((p, 'pos'))

    # Split up into words
    words = word_tokenize(p)

    # Tag words (e.g. noun, verb etc.)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())
logger.info('Finished creating documents out of positive reviews.')

# Annotate negative documents
logger.info('Start creating documents out of negative reviews.')
for p in neg_reviews.split('\n'):
    documents.append((p, 'neg'))

    # Split up into words
    words = word_tokenize(p)

    # Tag words (e.g. noun, verb etc.)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())
logger.info('Finished creating documents out of negative reviews.')

if pickling:
    # Save the documents
    logger.info('Start packing documents as pickle to ' + os.path.dirname(os.path.abspath(__file__)) + '/data/pickles/documents.pickle.')
    save_documents = open('data/pickles/documents.pickle', 'wb')
    pickle.dump(documents, save_documents)
    save_documents.close()
    logger.info('Finished packing documents as pickle.')

# Calculate the frequency of each word
all_words = nltk.FreqDist(all_words)

# Get the 10,000 most frequent words
logger.info('Create word feature set of 10,000 most frequent words.')
word_features = list(all_words.keys())[:10000]

if pickling:
    # Save the word features
    logger.info('Start packing word features as pickle to ' + os.path.dirname(os.path.abspath(__file__)) + '/data/pickles/word_features.pickle.')
    save_word_features = open('data/pickles/word_features.pickle', 'wb')
    pickle.dump(word_features, save_word_features)
    save_word_features.close()
    logger.info('Finished packing word features as pickle.')

# Search for features in documents
def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features

# Create a feature set
logger.info('Start creating feature set from documents.')
featuresets = [(find_features(rev), category) for (rev, category) in documents]
logger.info('Finished creating feature set from documents.')

if pickling:
    logger.info('Start packing word features as pickle to ' + os.path.dirname(os.path.abspath(__file__)) + '/data/pickles/featuresets.pickle.')
    save_feature_sets = open('data/pickles/featuresets.pickle', 'wb')
    pickle.dump(featuresets, save_feature_sets)
    save_feature_sets.close()
    logger.info('Finished packing feature set from documents.')

# Shuffle features
random.shuffle(featuresets)

# Define how many of the features are training data and how many are test data
number_of_training_documents = int(len(documents) * 95 / 100)
number_of_test_documents = int(len(documents) * 5 / 100)

# Define the training set
logger.info('Create training set of ' + str(number_of_training_documents) + ' documents.')
training_set = featuresets[:number_of_training_documents]

# Define the test set
logger.info('Create testing set of ' + str(number_of_test_documents) + ' documents.')
testing_set = featuresets[number_of_test_documents:]

cls = classification.Classifier(training_set, testing_set, args.type)

models.connect()

if models.TrainedClassifiers.objects(name = args.name).count():
    print 'TrainedClassifier already exists with name %s try to different name' % args.name
    sys.exit()

if args.type == 'naivebayes':
    resultClassifier = cls.run_naivebayes(True, pickling)

elif args.type == 'multinomialnb':
    resultClassifier = cls.run_multinomialnb(True, pickling)

elif args.type == 'bernoullinb':
    resultClassifier = cls.run_bernoullinb(True, pickling)

elif args.type == 'logisticregression':
    resultClassifier = cls.run_logisticregression(True, pickling)

elif args.type == 'sgd':
    resultClassifier = cls.run_sgd(True, pickling)

elif args.type == 'linearsvc':
    resultClassifier = cls.run_linearsvc(True, pickling)

elif args.type == 'nusvc':
    resultClassifier = cls.run_nusvc(True, pickling)

elif args.type == 'voteclassifier':
    naivebayes_classifier = cls.run_naivebayes(True, pickling)
    mnb_classifier = cls.run_multinomialnb(True, pickling)
    bernoullinb_classifier = cls.run_bernoullinb(True, pickling)
    logisticregression_classifier = cls.run_logisticregression(True, pickling)
    sgd_classifier = cls.run_sgd(True, pickling)
    linearsvc_classifier = cls.run_linearsvc(True, pickling)
    nusvc_classifier = cls.run_nusvc(True, pickling)

    resultClassifier = VoteClassifier(
        naivebayes_classifier,
        mnb_classifier,
        bernoullinb_classifier,
        logisticregression_classifier,
        sgd_classifier,
        linearsvc_classifier,
        nusvc_classifier
    )

    logger.info('Started accuracy calculation of Vote Classifier.')
    resultClassifier.accuracy = (nltk.classify.accuracy(cls, testing_set)) * 100
    logger.info('Finished accuracy calculation of Vote Classifier.')

    if pickling:
        logger.info('Start packing Vote Classifier as pickle to ' + os.path.dirname(os.path.abspath(__file__)) + '/data/pickles/vote_classifier.pickle.')
        save_classifier = open('data/pickles/naivebayes_classifier.pickle', 'wb')
        pickle.dump(resultClassifier, save_classifier)
        save_classifier.close()
        logger.info('Finished packing Vote Classifier as pickle.')
else:
    print '%s is not valid classifier type' % args.type
    sys.exit()

# Save
row = models.TrainedClassifiers()
row.name = args.name
row.set_classifier(resultClassifier)
row.stats = dict(
    classifier=cls.name,
    accuracy=cls.accuracy
)

row.save()


print 'TrainedClassifier saved with ID: %s  Name: %s' % (row.id, row.name)


