#!/usr/bin/env python
"""Module Docstrings"""
########################################################################
# File: template.py
#  executable: template.py
#
# Author: Andrew Bailey/ Trevor Pesout
# History: Created 10/18/17
########################################################################

from __future__ import print_function
import sys
from timeit import default_timer as timer
import csv
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from nltk.corpus import stopwords
import numpy as np
import math




def read_spam_data(csv_f):
    """Read in csv of ham spam data and return two lists of data"""
    label = []
    message = []
    with open(csv_f) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            message.append(row['sms'])
            label.append(row['label'])
            # print(row['label'], row['sms'])
    return message, label


def create_train_data(corpus, stop_words, bigram=False, lowercase=True):
    """Extract features from corpus and perform the tf-idf term weighting as well as removing stopwords"""
    # option to have 'this word' as a feature along with 'this' and 'word'
    if bigram:
        vectorizer = CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b',
                                     min_df=1, decode_error='ignore', stop_words=stop_words,
                                     lowercase=lowercase)
    else:
        vectorizer = CountVectorizer(decode_error='ignore', stop_words=stop_words, lowercase=lowercase)

    # create feature vector
    X = vectorizer.fit_transform(corpus)
    transformer = TfidfTransformer(smooth_idf=False, norm='l2')
    # normalize and perform tf-idf
    tfidf = transformer.fit_transform(X)
    return tfidf


def create_spam_ham_labels(labels_text, spam=1, ham=0):
    """Convert spam and ham into either 0 or 1 """
    integer_labels = []
    for label in labels_text:
        if label == "spam":
            integer_labels.append(spam)
        else:
            assert label == "ham"
            integer_labels.append(ham)
    return integer_labels


def square_loss(features, labels, weights):
    """Calculates total loss of lists of features and labels given weights"""
    total_loss = 0
    for index, feature in enumerate(features):
        a = sigmoid(np.matmul(weights, feature.toarray().T)[0])
        total_loss += ((a-labels[index])**2) / 2.0
    return total_loss


def sigmoid(x, alpha=1, beta=0):
    """Sigmoid function"""
    return math.exp(alpha*(x+beta)) / (1 + math.exp(alpha*(x+beta)))


def integral_of_sigmoid(x):
    """Returns the value from the integral of the sigmoid function"""
    return np.log((1+np.exp(x)))


def logistic_loss(features, labels, weights):
    """Calculates total loss of lists of features and labels given weights"""
    total_loss = 0
    for index, feature in enumerate(features):
        y = labels[index]
        a = np.matmul(weights, feature.toarray().T)[0]
        y_hat = sigmoid(a)
        if y_hat == 0:
            print("ERROR its 0")
        # need to double check this equation
        total_loss += y*np.log((y/y_hat)) + (1-y)*np.log((1-y)/(1-y_hat))
    return total_loss


def main():
    """Main flow of program yo"""
    # if cant find stopwords you can download using this
    # import nltk
    # nltk.download('stopwords')

    start = timer()
    stop_words = stopwords.words('english')
    # test_data = 'test.csv'
    train_data = 'train.csv'
    messages, text_labels = read_spam_data(train_data)
    X = create_train_data(messages, stop_words)
    int_labels = create_spam_ham_labels(text_labels, spam=1, ham=0)

    # make sure everything is still aligned
    assert X.shape[0] == len(messages)
    assert X.shape[0] == len(int_labels)
    assert X.shape[0] == len(text_labels)

    # expit is sigmoid function

    n_features = X.shape[1]
    weights = np.random.normal(0, 0.2, n_features)
    loss = square_loss(X, int_labels, weights=weights)
    loss_l = logistic_loss(X, int_labels, weights=weights)

    print(loss, loss_l)



    stop = timer()
    print("Running Time = {} seconds".format(stop - start), file=sys.stderr)


if __name__ == "__main__":
    main()
    raise SystemExit
