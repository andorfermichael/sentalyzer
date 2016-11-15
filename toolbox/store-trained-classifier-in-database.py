#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK

__author__ = 'ma'

import argparse
import argcomplete

from smm import models

import pickle
import logging

parser = argparse.ArgumentParser(description='Classify collected reviews', usage='python store-trained-classifier-in-database.py classifier 10000 data/pickles/naivebayes')
parser.add_argument('name', help='Classifier name - must be unique')
parser.add_argument('size', type=int, help='Corpus size - how much documents to classify')
parser.add_argument('path', help='Path to classifier pickle')
args = parser.parse_args()

argcomplete.autocomplete(parser)

# Define logger
logger = logging.getLogger('store-pickled-classifier-in-database')

# Connect to db
models.connect()

# Load classifier from file
logger.info('Start loading of classifier from ' + args.path + '.pickle')
classifier_f = open(args.path + '_classifier.pickle', 'rb')
classifier = pickle.load(classifier_f)
classifier_f.close()
logger.info('Finished loading of classifier.')

# Load accuracy value from file
logger.info('Start loading of classifier accuracy from ' + args.path + '_accuracy.pickle')
accuracy_f = open(args.path + '_classifier_accuracy.pickle', 'rb')
accuracy = pickle.load(accuracy_f)
accuracy_f.close()
logger.info('Finished loading of classifier accuracy.')

# Save
row = models.TrainedClassifiers()
row.name = args.name
row.set_classifier(classifier)
row.stats = dict(
    classifier=args.name,
    accuracy=accuracy
)

row.save()


print 'TrainedClassifier saved with ID: %s  Name: %s' % (row.id, row.name)


