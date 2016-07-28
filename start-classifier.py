__author__ = 'gx'
import time
import logging
import argparse

from smm.classifier.pool import ClassifierWorkerPool
from smm import models

logger = logging.getLogger('start-classifier')

parser = argparse.ArgumentParser(description='classify tweets', usage='python start-classifier classifier')
parser.add_argument('classifier', help='classifier name',
                    choices=['naivebayes', 'multinomialnb', 'bernoullinb', 'logisticregression', 'sgd',
                             'linearsvc', 'nusvc', 'voteclassifier'])
args = parser.parse_args()

# connect to db
models.connect()

# init pool
pool = ClassifierWorkerPool(args.classifier)

try:
    pool.start()
    logger.info('started with size %s', len(pool.workers))
    while True:
        time.sleep(1)

except (KeyboardInterrupt, SystemExit):
    pool.terminate()