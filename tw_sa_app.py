#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 13:08:16 2020

@author: raphaele
"""

import streamlit as st
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score
from sklearn.decomposition import TruncatedSVD


###########################
# import data
###########################


# import and cache the data
@st.cache(persist=True, suppress_st_warning=True, allow_output_mutation=True)
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
       

########################
# main function
########################

def main():
    def plot_metrics(metrics_list):
        if 'Confusion Matrix' in metrics_list:
            st.subheader("Confusion Matrix")
            plot_confusion_matrix(model, X_test, y_test, display_labels=class_names)
            st.pyplot()

        if 'ROC Curve' in metrics_list:
            st.subheader("ROC Curve")
            plot_roc_curve(model, X_test, y_test)
            st.pyplot()
        
        if 'Precision-Recall Curve' in metrics_list:
            st.subheader('Precision-Recall Curve')
            plot_precision_recall_curve(model, X_test, y_test)
            st.pyplot()
    
    st.header('Tweets sentiment prediction')
    data = load_data()
    clean_data = clean_text(data)  
    class_names = ['positive', 'negative']
    y_var, x_var = split_variables(clean_data)
    # split x data into training and test set on a 80/20 basis
    X_train, X_test, y_train, y_test = train_test_split(x_var, y_var, test_size=0.2, random_state=42)

    st.sidebar.subheader("Choose Classifier")
    classifier = st.sidebar.selectbox("Classifier", ("Support Vector Machine (SVM)", "Logistic Regression", "Naive Bayes"))
    
    if classifier == 'Support Vector Machine (SVM)':
        # choose text encoding
        st.sidebar.subheader("Feature extraction")
        ngram_range = st.sidebar.radio('N-grams', [(1, 1), (1, 2)])
        use_idf = st.sidebar.radio('Use idf', ('True', 'False'))
        # choose number of PCA
        st.sidebar.subheader("Parameters reduction")
        n_components = st.sidebar.radio('Number of PCA components', (600, 700, 800))
        #choose parameters
        st.sidebar.subheader("Model Hyperparameters")
        loss= st.sidebar.radio('Loss', ('hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'))
        penalty = st.sidebar.radio('Penalty', ('l2', 'l1', 'elasticnet'))
        alpha = st.sidebar.radio("Alpha", (0.0001, 0.001, 0.01))


        metrics = st.sidebar.multiselect("What metrics to plot?", ['Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'], 
                                         default=['Confusion Matrix'])
        
        st.subheader("Support Vector Machine (SVM) Results")    
        
        model = Pipeline([('vect', CountVectorizer(analyzer='word', ngram_range=ngram_range)),
                   ('tfidf', TfidfTransformer(use_idf = use_idf)),
                   ('pca', TruncatedSVD(n_components = n_components)),
                   ('clf', SGDClassifier(loss = loss, penalty = penalty, 
                                         alpha=alpha, random_state=42, 
                                         max_iter=5, tol=None))])
        model.fit(X_train, y_train)
        
        accuracy = model.score(X_test, y_test)
        y_pred = model.predict(X_test)
        st.write("Accuracy: ", accuracy.round(2))
        st.write("Precision: ", precision_score(y_test, y_pred, pos_label='positive').round(2))
        st.write("Recall: ", recall_score(y_test, y_pred, pos_label='positive').round(2))
        plot_metrics(metrics)
        
    if classifier == 'Logistic Regression':
        # choose text encoding
        st.sidebar.subheader("Feature extraction")
        ngram_range = st.sidebar.radio('N-grams', [(1, 1), (1, 2)])
        use_idf = st.sidebar.radio('Use idf', ('True', 'False'))
        # choose number of PCA
        st.sidebar.subheader("Parameters reduction")
        n_components = st.sidebar.radio('Number of PCA components', (600, 700, 800))
        st.sidebar.subheader("Model Hyperparameters")
        #choose parameters
        C = st.sidebar.number_input("C (Regularization parameter)", 5.0, 10.0, step=0.01, key='C_LR')
        max_iter = st.sidebar.slider("Maximum number of iterations", 100, 500, key='max_iter')
        penalty = st.sidebar.radio('Penalty', ('l2', 'none'))

        metrics = st.sidebar.multiselect("What metrics to plot?", ['Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'], 
                                         default=['Confusion Matrix'])        
        st.subheader("Logistic Regression Results")
        
        model = Pipeline([('vect', CountVectorizer(analyzer='word', ngram_range=ngram_range)),
                   ('tfidf', TfidfTransformer(use_idf = use_idf)),
                   ('pca', TruncatedSVD(n_components = n_components)),
                   ('clf', LogisticRegression(C=C, max_iter=max_iter, 
                                              penalty = penalty, random_state=42))])
        model.fit(X_train, y_train)
        
        accuracy = model.score(X_test, y_test)
        y_pred = model.predict(X_test)
        st.write("Accuracy: ", accuracy.round(2))
        st.write("Precision: ", precision_score(y_test, y_pred, pos_label='positive').round(2))
        st.write("Recall: ", recall_score(y_test, y_pred, pos_label='positive').round(2))
        plot_metrics(metrics)
    
    if classifier == 'Naive Bayes':
         # choose text encoding
        st.sidebar.subheader("Feature extraction")
        ngram_range = st.sidebar.radio('N-grams', [(1, 1), (1, 2), (1, 3)])
        use_idf = st.sidebar.radio('Use idf', ('True', 'False'))
        st.sidebar.subheader("Model Hyperparameters")
        #choose parameters
        alpha = st.sidebar.radio("Alpha", (0.0001, 0.001, 0.01, 0.5, 0.8, 0.9, 1))
        fit_prior = st.sidebar.radio("Learn class prior probabilities", ('True', 'False'))

        metrics = st.sidebar.multiselect("What metrics to plot?", ['Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'], 
                                         default=['Confusion Matrix'])        
        st.subheader("Naive Bayes results")

        model = Pipeline([('vect', CountVectorizer(analyzer='word', ngram_range=ngram_range)),
                   ('tfidf', TfidfTransformer(use_idf = use_idf)),
                   ('clf', MultinomialNB(alpha = alpha, fit_prior = fit_prior))])
        model.fit(X_train, y_train)
       
        accuracy = model.score(X_test, y_test)
        y_pred = model.predict(X_test)
        st.write("Accuracy: ", accuracy.round(2))
        st.write("Precision: ", precision_score(y_test, y_pred, pos_label='positive').round(2))
        st.write("Recall: ", recall_score(y_test, y_pred, pos_label='positive').round(2))
        plot_metrics(metrics)
    
if __name__ == '__main__':
    main()
   
    
    




