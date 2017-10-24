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
from scipy.special import expit


def sigmoid(x, alpha=1, beta=0):
    """Sigmoid function"""

    # this can overflow a double, so we use different implementations
    # return math.exp(alpha*(x+beta)) / (1 + math.exp(alpha*(x+beta)))
    if x < 0:
        return 1 - 1 / (1 + math.exp(alpha * (x + beta)))
    else:
        return 1 / (1 + math.exp(-alpha * (x + beta)))


def integral_of_sigmoid(x):
    """Returns the value from the integral of the sigmoid function"""
    return np.log((1 + np.exp(x)))


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


# TODO tpesout: make this persistent, so we can use it with the test data.
# we want to make this use the same features with different data (so we can test)
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
    return tfidf.toarray()


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
        a = sigmoid(np.matmul(weights, feature.T))
        total_loss += ((a - labels[index]) ** 2) / 2.0
    return float(total_loss/index)


def logistic_loss(features, labels, weights):
    """Calculates total loss of lists of features and labels given weights"""
    total_loss = 0
    for index, feature in enumerate(features):
        y = labels[index]
        a = np.matmul(weights, feature.T)
        y_hat = sigmoid(a)
        if y_hat == 0:
            print("ERROR its 0")
        # need to double check this equation
        total_loss += y * np.log(y) + (1 - y) * np.log(1 - y_hat)
    return total_loss


def logistic_evaluation(inputs, weights):
    """Returns predicted outputs from the inputs and weights"""
    prediction = []
    for index, feature in enumerate(inputs):
        a = np.matmul(weights, feature.T)
        y_hat = sigmoid(a)
        prediction.append(y_hat)
    return np.array(prediction)


def logistic_regression(inputs, labels, weights, l, alpha, eta_0, step, bias=False):
    """Implement regularized logistic regression with moving learning weight"""
    eta = eta_0 * (step ** alpha)
    y_hat = logistic_evaluation(inputs, weights)
    if bias:
        regularize = weights*(eta * l)
        regularize[0] = 0
    else:
        regularize = weights*(eta * l)
    new_weights = weights - regularize - (eta * np.matmul(np.array(y_hat-labels), inputs))
    return new_weights



def iterative_reweighted_least_squares(inputs, labels, weights, lamda):
    """Newton-Raphson iterative reweighted least squares"""
    R, y_hat = logistic_hessian(inputs, weights)
    hessian = np.matmul(np.matmul(inputs.T, R), inputs)
    inv_hessian = np.linalg.pinv(hessian)
    new_weights = weights - np.matmul(inv_hessian, np.matmul(inputs.T, (y_hat - labels)) - (weights*lamda))
    return new_weights


def logistic_hessian(inputs, weights):
    """Calculate hessian matrix for logistic regression"""
    y_hat = logistic_evaluation(inputs, weights)
    b = np.array([1 - y for y in y_hat])
    # p(1-p)

    prob = [y_hat * b]
    n = len(prob[0])
    assert n == len(y_hat)
    d = np.repeat(prob, [n, ], axis=0)
    I = np.identity(n)
    R = np.multiply(I, prob)
    # hessian = np.matmul(np.matmul(inputs.T, R), inputs)
    return R, y_hat

########################################################
# prepare and load the training data.  this involves
# reading in the data and finding the best features
########################################################

# if cant find stopwords you can download using this:
# import nltk
# nltk.download('stopwords')

# init
stop_words = stopwords.words('english')
train_data = 'train.csv'
# test_data = 'test.csv'

# interpret data
messages, text_labels = read_spam_data(train_data)
all_train_data = create_train_data(messages, stop_words)
int_labels = create_spam_ham_labels(text_labels, spam=1, ham=0)

new_col = np.ones([len(all_train_data), 1])
all_train_data = np.append(new_col, all_train_data, 1)

# get sizes
n_messages = len(messages)
n_features = all_train_data.shape[1]
# make sure everything is still aligned
print(all_train_data.shape)
assert all_train_data.shape[0] == len(messages)
assert all_train_data.shape[0] == len(int_labels)
assert all_train_data.shape[0] == len(text_labels)

########################################################
# this is the definition of the hyper parameters for
# the regression
########################################################

# lambda
lambda_base = 2  # 8 #np.e
lambda_exp_min = -1  # -5
lambda_exp_max = 3  # 1
list_of_lambdas = [lambda_base ** i for i in range(lambda_exp_min, lambda_exp_max + 1)]
print("LAMBDAS:\n\tbase: {}   log_min: {}   log_max: {}\n\t{}".format(
    lambda_base, lambda_exp_min, lambda_exp_max, list_of_lambdas))

# sigmoid params
eta_0 = 0.1
alpha = 0.9

########################################################
# divide the data for 10-fold cross validation
########################################################

# identifiers
TRAIN_DATA = "t_data"
TRAIN_LABELS = "t_labels"
VALIDATE_DATA = "v_data"
VALIDATE_LABELS = "v_labels"

# prep
number_of_buckets = 10
size_of_bucket = int(n_messages / number_of_buckets)
all_train_buckets = dict()

# divide into buckets
idx = 0
for b in range(number_of_buckets):
    data = all_train_data[idx:idx + size_of_bucket]
    labels = int_labels[idx:idx + size_of_bucket]
    all_train_buckets[b] = [data, labels]
    idx += size_of_bucket


# how to create train and validation data sets
def get_train_data_set(idx):
    t_data, t_labels = list(), list()
    v_data, v_labels = None, None
    for k in all_train_buckets.keys():
        v = all_train_buckets[k]
        if k == idx:
            v_data = v[0]
            v_labels = v[1]
        else:
            t_data.append(v[0])
            t_labels.append(v[1])
    return {
        TRAIN_DATA: np.vstack(t_data),
        TRAIN_LABELS: np.hstack(t_labels),
        VALIDATE_DATA: v_data,
        VALIDATE_LABELS: v_labels
    }


# get data
all_training_datasets = [get_train_data_set(x) for x in list(range(number_of_buckets))]

# validation
assert len(all_training_datasets) == number_of_buckets
for ds in all_training_datasets:
    assert ds[TRAIN_DATA].shape[1] == n_features
    assert ds[TRAIN_DATA].shape[0] == len(ds[TRAIN_LABELS])
    assert ds[VALIDATE_DATA].shape[1] == n_features
    assert ds[VALIDATE_DATA].shape[0] == len(ds[VALIDATE_LABELS])


########################################################
# definition of our regression function
########################################################


def run_regression(lamda, train, train_labels, validate, validate_labels,
                   eta_0=0.1, alpha=0.9, iterations=321, verbose=False, iwlsr=False, bias=True):
    # init
    report_frequency = int(iterations / 10.0)
    t = None
    # run regression
    try:
        weights = np.random.normal(0, 0.2, n_features)
        min_validate_loss = sys.maxsize
        min_train_loss = sys.maxsize
        min_weights_val_loss = []
        for t in range(iterations):
            if iwlsr:
                weights = iterative_reweighted_least_squares(train, train_labels, weights, lamda)
            else:
                weights = logistic_regression(train, train_labels, weights, lamda, alpha, eta_0, t, bias=bias)
            train_loss = square_loss(train, train_labels, weights=weights)
            val_loss = square_loss(validate, validate_labels, weights=weights)
            if verbose and (t % report_frequency) == 0:
                print("{}:\t#{}\ttrain {}  \t\tvalidate {}".format(lamda, t, train_loss, val_loss))
            if val_loss < min_validate_loss: min_weights_val_loss = weights
            if val_loss < min_validate_loss: min_validate_loss = val_loss
            if train_loss < min_train_loss: min_train_loss = train_loss

    except Exception as e:
        print("\nlambda {} #{}: {}".format(lamda, t, e), sys.stderr)
        import traceback
        traceback.print_exc()
        return False

    # return best
    return min_validate_loss, min_train_loss, min_weights_val_loss




dataset = all_training_datasets[0]


# #
# start = timer()
# v_error, t_error, best_weights = run_regression(0.000000, dataset[TRAIN_DATA], dataset[TRAIN_LABELS],
#                                                     dataset[VALIDATE_DATA], dataset[VALIDATE_LABELS], iwlsr=True, iterations=3, verbose=True)
# print(v_error, t_error, best_weights)
# stop = timer()
# print("Linear Regression Running Time = {} seconds".format(stop - start), file=sys.stderr)
#

start = timer()
v_error, t_error, best_weights = run_regression(0.000001, dataset[TRAIN_DATA], dataset[TRAIN_LABELS],
                                                dataset[VALIDATE_DATA], dataset[VALIDATE_LABELS], iwlsr=False, iterations=200, verbose=True,
                                                bias=True)
print(v_error, t_error, best_weights)
stop = timer()
print("Iterative Running Time = {} seconds".format(stop - start), file=sys.stderr)

#
#
# #######################################################
# run on each of our k-folded datasets
# #######################################################
#
#
#
#
# # prep
# lambda_to_validate_errors = dict()
# lambda_to_training_errors = dict()
#
# print(list_of_lambdas)
# # calculate for our lambdas
# for l in list_of_lambdas:
#     start = timer()
#     print("\nLAMBDA: {}\n\t".format(l), end='')
#     v_errors = list()
#     t_errors = list()
#     lambda_to_validate_errors[l] = v_errors
#     lambda_to_training_errors[l] = t_errors
#     for dataset in all_training_datasets:
#         v_error, t_error, best_weights = run_regression(l, dataset[TRAIN_DATA], dataset[TRAIN_LABELS],
#                                           dataset[VALIDATE_DATA], dataset[VALIDATE_LABELS])
#         v_errors.append(v_error)
#         t_errors.append(t_error)
#         print('.', end='')
#     print(" ({}s)".format(int(timer() - start)))
#     print("\tv errors:    {}".format(v_errors))
#     print("\tv error avg: {}".format(np.mean(v_errors)))
#     print("\tv error std: {}".format(np.std(v_errors)))
#
# ########################################################
# # analyze and print plots of the data
# ########################################################
# # %matplotlib inline
# import matplotlib.pyplot as plt
#
#
# def get_plot_data(items):
#     items.sort(key=lambda x: x[0])
#     x, y, var, log_lambda = list(), list(), list(), lambda_exp_min
#     for k, v in items:
#         x.append(log_lambda)
#         y.append(np.mean(v))
#         var.append(np.std(v) ** 2)
#         assert lambda_base ** log_lambda == k
#         log_lambda += 1
#     return x, y, var
#
#
# validate_x, validate_y, validate_v = get_plot_data(list(lambda_to_validate_errors.items()))
# train_x, train_y, train_v = get_plot_data(list(lambda_to_training_errors.items()))
#
# plt.errorbar(train_x, train_y, train_v, None, 'bo-', label="Train")
# plt.errorbar(validate_x, validate_y, validate_v, None, 'ro-', label="Validate")
# plt.axis([lambda_exp_min - .5, lambda_exp_max - .5, -2, 20])
# plt.legend(bbox_to_anchor=(.31, .95))
# plt.title("Logistic Regression Error Rates")
# plt.xlabel("Log of Lambda (Base {})".format(lambda_base))
# plt.ylabel("Error")
# plt.show()
