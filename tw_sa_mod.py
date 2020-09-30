#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 13:08:16 2020

@author: raphaele
"""

import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import roc_curve
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
import json
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay
import plotly.express as px
from sklearn.naive_bayes import MultinomialNB
from pandas import Series, DataFrame
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.decomposition import SparsePCA
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import SGDClassifier




#import os
# set working directory
#os.getcwd()
#os.chdir('/Users/raphaele/Documents/Py/sentiment_a')

###########################
# import data
###########################


# import and cache the data
def load_data():
    # import tweet data as collected in tweet_corpus.py
    # read the tweets file to a list
    with open('model_tweets.txt', 'r') as f:
        model_data = json. loads(f. read())
    return model_data






########################
# pre processing data functions
########################

# clean tweets text

def clean_text(raw_tweets):
    import re
    for tweet in raw_tweets:
        # all text in lower case
        tweet['text'] = tweet['text'].lower()
        # remove urls
        tweet['text'] = re.sub(r"http\S+|www\S+|https\S+", '', tweet['text'], flags=re.MULTILINE)
        # remove user references and '#'
        tweet['text'] = re.sub(r'\@\w+|\#','', tweet['text'])
    #return ' '.join(tweet)
    return(raw_tweets)


# split data into x and y

def split_variables(full_data):
    # object for explantory data ie tweets text
    x_tweet_text = [tweet['text'] for tweet in full_data]
    # object for target variable ie sentiment
    y_tweet_sentiment = [tweet['label'] for tweet in full_data]
    return(y_tweet_sentiment, x_tweet_text)


##########
# feature extraction function
###########
def feature(x_var):
    # tokenisation and vectorization of x data
    # break down x data to word units, add 1-grams and 2-grams
    # and change strings to normalized numeric vectors and count occurrences
    count_vect = CountVectorizer(analyzer='word', ngram_range=(1, 2))
    x_var_counts = count_vect.fit_transform(x_var)

    # transform the count matrix to a tf-idf representation
    tfidf_transformer = TfidfTransformer()
    x_var = tfidf_transformer.fit_transform(x_var_counts)
    return(x_var)




########################
# main function
########################

def prep_data():
    model_data = load_data()
    clean_data = clean_text(model_data)
    y_var, x_var = split_variables(clean_data)
    #x_var = feature(x_var)
    #dims = x_var.shape
    #print('x_var is now a spare numerical matrix of shape: ', dims)
    #st.write(x_var.toarray())
    # split x data into training and test set on a 80/20 basis
    X_train, X_test, y_train, y_test = train_test_split(x_var, y_var, test_size=0.2, random_state=42)
    return (X_train, X_test, y_train, y_test)

X_train, X_test, y_train, y_test = prep_data()

########################
# logistic regression
########################



# pipeline data vectorisation => transformation => classifier
lg_clf = Pipeline([('vect', CountVectorizer()),
                   ('tfidf', TfidfTransformer()),
                   ('pca', TruncatedSVD()),
                   # rescale matrix after pca to remove negative values
                   #('scale', MinMaxScaler(feature_range=(0, 1))),
                   ('clf', LogisticRegression())])


# parameter tuning with grid search

lg_parameters = {'vect__ngram_range': [(1, 1), (1, 2)], 
                                       'tfidf__use_idf': (True, False),
                                       'pca__n_components': [400, 450, 500],
                                       #'logistic__C': np.logspace(-4, 4, 4),
                                       #'logistic__class_weight': ('balanced', None)
                                       }

# set grid search insatnce
gs_lg_clf = GridSearchCV(lg_clf, lg_parameters, cv=5, n_jobs=-1)

# fit the grid search instance

gs_lg_clf = gs_lg_clf.fit(X_train, y_train)

gs_lg_clf.best_score_  # accuracy 0.7945408281286144, it is 0.66 with PCA

for param_name in sorted(lg_parameters.keys()):
    print("%s: %r" % (param_name, gs_lg_clf.best_params_[param_name]))
# best parameters
#pca__n_components: 450
#tfidf__use_idf: True
#vect__ngram_range: (1, 1)
    
# note about PCA: it does not improve performance and increases running time significantly,
    # so no PCA for any of these models. Maybe there is another algo that could
    #do feature reduction on sparse amtrix?

########################
# Naive Bayes
########################


# pipeline data vectorisation => transformation => classifier
nb_clf = Pipeline([('vect', CountVectorizer()),
                   ('tfidf', TfidfTransformer()),
                   #('pca', TruncatedSVD()),
                   # rescale matrix after pca to remove negative values
                   #('scale', MinMaxScaler(feature_range=(0, 1))),
                   ('clf', MultinomialNB())])


# parameter tuning with grid search

nb_parameters = {'vect__ngram_range': [(1, 1), (1, 2)], 
                                       'tfidf__use_idf': (True, False),
                                       #'pca__n_components': [5, 15, 30, 45, 64],
                                       'clf__alpha': (1e-2, 1e-3),}
# set grid search insatnce
gs_nb_clf = GridSearchCV(nb_clf, nb_parameters, cv=5, n_jobs=-1)

# fit the grid search instance

gs_nb_clf = gs_nb_clf.fit(X_train, y_train)

gs_nb_clf.best_score_  # accuracy 0.8113115891741846

for param_name in sorted(nb_parameters.keys()):
    print("%s: %r" % (param_name, gs_nb_clf.best_params_[param_name]))
# best parameters
#clf__alpha: 0.01
#tfidf__use_idf: False
#vect__ngram_range: (1, 1)
    
# update the model with best parameters
nb_clf = Pipeline([('vect', CountVectorizer(ngram_range = (1, 1))),
                   ('tfidf', TfidfTransformer(use_idf = False)),
                   ('clf', MultinomialNB(alpha = 0.01))])
    
# fit model
nb_clf.fit(X_train, y_train)
    
# prediction on test data
nb_y_pred = nb_clf.predict(X_test)
    

conf_nb = metrics.confusion_matrix(y_test, nb_y_pred)

# accuracy
(conf_nb[0,0] + conf_nb[1,1]) / len(nb_y_pred) # 0.7696969696969697


########################
# SVM
########################

# pipeline data vectorisation => transformation => classifier
svm_clf = Pipeline([('vect', CountVectorizer()),
                   ('tfidf', TfidfTransformer()),
                   ('pca', TruncatedSVD(n_components = 450)),
                   # rescale matrix after pca to remove negative values
                   #('scale', MinMaxScaler(feature_range=(0, 1))),
                   ('clf', SGDClassifier(
                                         alpha=1e-3, random_state=42,
                                         max_iter=5, tol=None))])



# parameter tuning with grid search

svm_parameters = {'vect__ngram_range': [(1, 1), (1, 2)], 
                                       'tfidf__use_idf': (True, False),
                                       #'pca__n_components': [440],
                                       'clf__loss': ('hinge','squared_hinge'),
                                       'clf__penalty': ('l2', 'l1', 'elasticnet')
                                       }
# set grid search insatnce
svm_nb_clf = GridSearchCV(svm_clf, svm_parameters, cv=5, n_jobs=-1)

# fit the grid search instance

svm_nb_clf = svm_nb_clf.fit(X_train, y_train)

svm_nb_clf.best_score_  # 0.8173259310663891

for param_name in sorted(svm_parameters.keys()):
    print("%s: %r" % (param_name, svm_nb_clf.best_params_[param_name]))
# best parameters
#clf__loss: 'hinge'
#clf__penalty: 'l2'
#tfidf__use_idf: True
#vect__ngram_range: (1, 1)


# update the model with best parameters
svm_clf = Pipeline([('vect', CountVectorizer(ngram_range = (1, 1))),
                   ('tfidf', TfidfTransformer(use_idf = True)),
                   ('clf', SGDClassifier(loss = 'hinge', penalty = 'l2', alpha=1e-3, random_state=42,
                                         max_iter=5, tol=None))])
    
# fit model
svm_clf.fit(X_train, y_train)
    
# prediction on test data
svm_y_pred = svm_clf.predict(X_test)
    
conf_svm = metrics.confusion_matrix(y_test, svm_y_pred)

# accuracy
(conf_svm[0,0] + conf_svm[1,1]) / len(svm_y_pred) # 0.8303030303030303

# we select because it has the best accuracy


########################
# train model
########################


#
#
## fit model to training set
#
#sklearn.linear_model.LogisticRegression
#
#clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)
#
#

#len(y_var)
#
#

#
#
#log_reg_mod.score(x_var_tfidf, y_var)
#roc_curve(y_var, y_pred)
#auc(y_var, y_pred)
#

#
## for all lables
#precision_recall_fscore_support(y_var, y_pred, average='weighted')
#

########################
# optimize model
########################



