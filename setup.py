from setuptools import setup, find_packages

setup(
    name='smm',
    version='0.1',
    packages=['smm'],
    url='https://github.com/andorfermichael/sentalyzer',
    license='GPLv3',
    author='Michael Andorfer',
    author_email='mandorfer.mmt-b2014@fh-salzburg.ac.at',
    description='Real-Time, multi-lingual Twitter sentiment analyzer engine',
    install_requires=[
        "requests",
        "requests_oauthlib",
        "mongoengine",
        "nltk",
        'numpy',
        'gevent',
        'gevent-socketio',
        'flask',
        'argcomplete',
        'pymongo == 2.8.1',
        'twitter'
    ]
)
