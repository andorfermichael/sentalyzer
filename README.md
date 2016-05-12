Sentalyzer
==========

Sentalyzer is a realtime twitter sentiment analyzer


Requirements
------------

- python 2.7
- mongodb server


Checkout
--------
Checkout latest sentalyzer master branch from github


    git clone https://github.com/andorfermichael/sentalyzer.git ./sentalyzer
    cd sentalyzer


Configure
---------
copy smm/config-sample.py to smm/config.py and edit smm/config.py according to your needs

    cp smm/config-sample.py smm/config.py
    nano smm/config.py


Installation & Setup
--------------------
Download and install required libs and data

    python setup.py develop
    python toolbox/setup-app.py


Testing
-------
Run unittests

    python -m unittest discover tests


Providing data
---------------------
Provide data in raw format e.g
>data/hotel_reviews/pos/pos0001.txt
 data/hotel_reviews/neg/neg0001.txt

Provide data in serialized format e.g.
>data/pickles/naivebayes_classifier.pickle
 data/pickles/bernoullinb_classifier.pickle

The raw data is the recommended way since it reduces errors at a minimum.


Available classifiers
----------------
- Naive Bayes (used in command as "naivebayes")
- Multinomial Naive Bayes (used in command as "multinomialnb")
- Bernoulli Naive Bayes (used in command as "bernoullinb")
- Logistic Regression Classifier (used in command as "logisticregression")
- Stochastic Gradient Descent (used in command as "sgd")
- Linear Support Vector Classification (used in command as "linearsvc")
- Nu Support Vector Classifcation (used in command as "nusvc")
- Voted Classifier (used in command as "votedclassifier")

The `Voted Classifier` combines the results of all above mentioned classifiers.


Train classifier
----------------
Create and save new classifier trained from raw data (or pickled classifier)

    python toolbox/train-classifier.py myClassifier numberOfDocuments

Load and save new classifier from already trained raw data (or pickled classifier)

    python toolbox/store-pickle-classifier-in-database.py myClassifier numberOfDocuments

for more options see

    python toolbox/train-classifier.py --help


Start server stack
------------------
open 3 shells and type in each:
    
    python start-collector.py
    python start-classifier.py
    python start-server.py
    

open browser on http://127.0.0.1:5000


Show stats
----------
Show detailed info on saved classifiers

    python toolbox/show-classifiers.py

Its worth mention that `Training data size` is the size of the trained classifier after it has been
serialized (pickled) with highest protocol actual Memory Usage may vary...



Interactive shell
-----------------
You can directly interact with the trained classifier and get verbose output on how the score is calculated
replace `maxEntTestCorpus` with desired classifier name see `Show stats` to display available classifiers

    python toolbox/shell-classifier.py maxEntTestCorpus

You should see:

    exit: ctrl+c

    Loaded maxEntTestCorpus
    Classify:

Type something and hit enter:

    Classify: today is a bad day for this nation

    Classification: negative with 53.29%

    Feature                                          negativ positiv
    ----------------------------------------------------------------
    bad==1 (1)                                         0.074
    today==1 (1)                                       0.027
    day==1 (1)                                         0.008
    bad==1 (1)                                                -0.178
    nation==1 (1)                                              0.139
    today==1 (1)                                              -0.035
    day==1 (1)                                                -0.007
    -----------------------------------------------------------------
    TOTAL:                                             0.109  -0.081
    PROBS:                                             0.533   0.467



for more options see

    python toolbox/shell-classifier.py --help


Production & Deployment
-----------------------
Run everything behind nginx >= 1.3.13, automate processes management with supervisor.

Since nginx 1.3.13 supports websockets, so you should probably use latest stable version.

This is only one way of many to deploy the app.
in folder ex.conf there are sample config files for nginx and supervisor.


Links, Sources etc
------------------

- http://mpqa.cs.pitt.edu/
- http://nlp.stanford.edu/sentiment/index.html
- http://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html#datasets
- http://nginx.org/en/docs/http/websocket.html
- http://supervisord.org/


Credits
-----------------------------

This project is based on the [Streamcrab Project](https://github.com/cyhex/streamcrab) written by [Cyphex](https://github.com/cyhex)