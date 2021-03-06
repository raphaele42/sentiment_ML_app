#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 13:08:16 2020

@author: raphaele
"""

import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve
from sklearn.metrics import precision_score, recall_score
import matplotlib  #needed by heroku for executing plot confusion matrix
from pandas import Series, DataFrame  #needed
import matplotlib.pyplot as plt
import plotly.express as px
import base64
import json

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
    # returns a dataframe
    return model_data

###########################
# best models data
###########################

# caching this object so it is not refreshed each time the script runs
@st.cache(persist=True, suppress_st_warning=True, allow_output_mutation=True)
def load_best_data():
    # initiate data frame to store perf and parameters of best models
    best_models = DataFrame(columns=['Value', 'SVM','LogisticReg','NaiveBaye'])
    best_models['Value'] = ['Accuracy', 'Precision', 'Recall',
               'N-grams', 'Idf', 'Loss', 'Penalty', 'Alpha',
               'C', 'Max iterations', 'Learn prior']
    rows_perf = [0, 1, 2]
    for rows in rows_perf:
        best_models.loc[rows, 1:4] = 0
    return best_models

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
# function to download the new labelled tweets
########################

def get_table_download_link(df):
    # generate a link allowing the labelled data generated bu the model to be downloaded
    #in:  dataframe
    #out: href string
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}">Download csv file</a>'
    return href


########################
# main function
########################

def main():
    
    def plot_metrics():
    # define plots for ROC curve and confusion matrix
        
        st.subheader('ROC Curve')
        st.write('You can also use the ROC curve to adjuste the model parameters: the best model will show a blue line as close as possible to the dotted red line.')
        plot_roc_curve(model, X_test, y_test, lw=1)
        plt.plot([0, 0, 1], [0, 1, 1], color='red', lw=0.5, linestyle='--')
        st.pyplot()
        
        st.subheader("Confusion Matrix")
        st.write('The confusion matrix shows how well the model spots negative and positive sentiments. We want as many entries as possible in the True Positive (top left) and True Negative (bottom right) cells.')
        plot_confusion_matrix(model, X_test, y_test, display_labels=class_names, cmap=plt.cm.BuPu)
        st.pyplot()

        
    # load and apply pre processing to data
    data = load_data()
    clean_data = clean_text(data)  
    class_names = ['positive', 'negative']
    y_var, x_var = split_variables(clean_data)
    # split x data into training and test set on a 80/20 basis
    X_train, X_test, y_train, y_test = train_test_split(x_var, y_var, test_size=0.2, random_state=42)
    
    # initiate df to store best models performance
    best_models = load_best_data()
    
    # initiate df to store current model performance
    curr_perf = DataFrame(columns=['Accuracy', 'Precision','Recall']) 
    
    # copy - explain the functionning of the app
    st.title('Tweets sentiment prediction')
    st.write('To determine positive and negative sentiments in a set of un-labelled tweets, we will:')
    st.write('1 - Train three different algorithms and optimise them to get the best models for each one.')
    st.write('2 - Select the best model to analyse the sentiment in a new set of tweets.')
    st.write('3 - Get a preview of the results and download the csv file with labelled tweets.')
    
    # copy - give insights in the initial data ser
    pos_y_sentiment = [i for i in y_var if i == 'positive']
    pos_perc = (100*len(pos_y_sentiment)/len(y_var))
    st.write('The training data set includes ', len(y_var), ' tweets, including ',round(pos_perc, 2), '% classified as positive.')
    
    ############# 
    # section 1 #
    #############
    
    st.header('1 - Find the best model')
    
    st.write('Choose a classifier in the left hand side. Adjust the parameters \
             in the left column and see how it impacts the model performance. \
             Higher values in the table below indicate higher performance of \
             the model. For each algorithm tested, the best model and its \
             parameters are saved in the table in section 2.')
    
    ################### 
    # sidebar content #
    ###################
    
    st.sidebar.markdown('_[See code and documentation](https://github.com/raphaele42/sentiment_a)_.')
    st.sidebar.subheader("Choose a classifier")
    classifier = st.sidebar.selectbox('Choose an algorithm:', ("Support Vector Machine (SVM)", "Logistic Regression", "Naive Bayes"))
    
    ################### 
    # sidebar for SVM #
    ###################
    
    if classifier == 'Support Vector Machine (SVM)':
        # choose text encoding
        st.sidebar.subheader('Feature extraction')
        ngram_range_svm = st.sidebar.radio('N-grams (length of word groups to extract)', [(1, 1), (1, 2)])
        use_idf_svm = st.sidebar.radio('Use idf (method to compute words frequency)', ('True', 'False'))
        #choose parameters
        st.sidebar.subheader('Model Hyperparameters')
        loss= st.sidebar.radio('Loss (function to measure model fit)', ('hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'))
        penalty_svm = st.sidebar.radio('Penalty (to penalise model complexity)', ('l2', 'l1', 'elasticnet'))
        alpha_svm = st.sidebar.radio('Alpha (to multiply the penalty)', (0.0001, 0.001, 0.01))

        
        st.subheader('Current Support Vector Machine (SVM) results:')    
        
        model = Pipeline([('vect', CountVectorizer(analyzer='word', ngram_range=ngram_range_svm)),
                   ('tfidf', TfidfTransformer(use_idf = use_idf_svm)),
                   ('clf', SGDClassifier(loss = loss, penalty = penalty_svm, 
                                         alpha=alpha_svm, random_state=42, 
                                         max_iter=5, tol=None))])
        # fit the model
        model.fit(X_train, y_train)
        # prediction on test data
        y_pred = model.predict(X_test)
       
        # perf scores for svm
        accuracy_svm = model.score(X_test, y_test).round(2)        
        precision_svm = precision_score(y_test, y_pred, pos_label='positive').round(2)
        recall_svm = recall_score(y_test, y_pred, pos_label='positive').round(2)
        
        # populate the best perf table
        curr_perf_new = {'Accuracy': accuracy_svm, 'Precision' : precision_svm, 'Recall' : recall_svm}
        curr_perf = curr_perf.append(curr_perf_new, ignore_index=True) 
        curr_perf = curr_perf.rename(index={0: 'Score'})
        st.table(curr_perf)
        
        # update the best perf table if new model is better than current best one
        if (accuracy_svm > best_models.iloc[0, 1]):
            best_models.iloc[0, 1] = accuracy_svm
            best_models.iloc[1, 1] = precision_svm
            best_models.iloc[2, 1] = recall_svm
            best_models.iloc[3, 1] = ngram_range_svm
            best_models.iloc[4, 1] = use_idf_svm
            best_models.iloc[5, 1] = loss
            best_models.iloc[7, 1] = alpha_svm
            best_models.iloc[6, 1] = penalty_svm
        
            
        plot_metrics()
        
    ################### 
    # sidebar for LR  #
    ###################
    
    if classifier == 'Logistic Regression':
        # choose text encoding
        st.sidebar.subheader("Feature extraction")
        ngram_range_lr = st.sidebar.radio('N-grams (length of word groups to extract)', [(1, 1), (1, 2)], key = 'ngram_range_lr')
        use_idf_lr = st.sidebar.radio('Use idf', ('True', 'False'), key = 'use_idf_lr')
        st.sidebar.subheader("Model Hyperparameters")
        #choose parameters
        C = st.sidebar.number_input("C (Regularization parameter)", 5.0, 10.0, step=0.01, key='C_LR')
        max_iter = st.sidebar.slider("Maximum number of iterations", 100, 500, key='max_iter')
        #penalty = st.sidebar.radio('Penalty', ('l2', 'none'))
        penalty_lr = st.sidebar.selectbox('Penalty (to penalise model complexity)', ('l2', 'none'))
     
        st.subheader("Current Logistic Regression results:")
        
        model = Pipeline([('vect', CountVectorizer(analyzer='word', ngram_range=ngram_range_lr)),
                   ('tfidf', TfidfTransformer(use_idf = use_idf_lr)),
                   ('clf', LogisticRegression(C=C, max_iter=max_iter, 
                                              penalty = penalty_lr, random_state=42))])
        # fir the model
        model.fit(X_train, y_train)
        # predict the sentiment on test data
        y_pred = model.predict(X_test)
        
        # perf scores for lr
        accuracy_lr = model.score(X_test, y_test).round(2)
        precision_lr = precision_score(y_test, y_pred, pos_label='positive').round(2)
        recall_lr = recall_score(y_test, y_pred, pos_label='positive').round(2)
        
        # populate the best perf table
        curr_perf_new = {'Accuracy': accuracy_lr, 'Precision' : precision_lr, 'Recall' : recall_lr}
        curr_perf = curr_perf.append(curr_perf_new, ignore_index=True) 
        curr_perf = curr_perf.rename(index={0: 'Score'})
        st.table(curr_perf)
        
        # update the best perf table if new model is better than current best
        if (accuracy_lr > best_models.iloc[0, 2]):
            best_models.iloc[0, 2] = accuracy_lr
            best_models.iloc[1, 2] = precision_lr
            best_models.iloc[2, 2] = recall_lr
            best_models.iloc[3, 2] = ngram_range_lr
            best_models.iloc[4, 2] = use_idf_lr
            best_models.iloc[8, 2] = C
            best_models.iloc[9, 2] = max_iter
            best_models.iloc[6, 2] = penalty_lr
            
        # plot ROC and confusion matrix
        plot_metrics()
    
    ################### 
    # sidebar for NB  #
    ###################
    
    if classifier == 'Naive Bayes':
         # choose text encoding
        st.sidebar.subheader('Feature extraction')
        ngram_range_nb = st.sidebar.radio('N-grams (length of word groups to extract)', [(1, 1), (1, 2), (1, 3)])
        use_idf_nb = st.sidebar.radio('Use idf', ('True', 'False'))
        st.sidebar.subheader('Model Hyperparameters')
        #choose parameters
        alpha_nb = st.sidebar.radio('Alpha (to multiply the penalty)', (0.0001, 0.001, 0.01, 0.1))
        fit_prior = st.sidebar.radio('Learn class prior probabilities', ('True', 'False'))
    
        st.subheader("Current Naive Bayes results:")

        model = Pipeline([('vect', CountVectorizer(analyzer='word', ngram_range=ngram_range_nb)),
                   ('tfidf', TfidfTransformer(use_idf = use_idf_nb)),
                   ('clf', MultinomialNB(alpha = alpha_nb, fit_prior = fit_prior))])
        # fit model
        model.fit(X_train, y_train)
        # prediction on test data
        y_pred = model.predict(X_test)
        
        # performance scores for naive bayes
        accuracy_nb = model.score(X_test, y_test).round(2)
        precision_nb = precision_score(y_test, y_pred, pos_label='positive').round(2)
        recall_nb = recall_score(y_test, y_pred, pos_label='positive').round(2)
        
        # populate the best performance table
        curr_perf_new = {'Accuracy': accuracy_nb, 'Precision' : precision_nb, 'Recall' : recall_nb}
        curr_perf = curr_perf.append(curr_perf_new, ignore_index=True) 
        curr_perf = curr_perf.rename(index={0: 'Score'})
        st.table(curr_perf)
        
        # update the best performance table if the new model is better than current best
        if (accuracy_nb > best_models.iloc[0, 3]):
            best_models.iloc[0, 3] = accuracy_nb
            best_models.iloc[1, 3] = precision_nb
            best_models.iloc[2, 3] = recall_nb
            best_models.iloc[3, 3] = ngram_range_nb
            best_models.iloc[4, 3] = use_idf_nb
            best_models.iloc[8, 3] = alpha_nb
            best_models.iloc[9, 3] = fit_prior
        # plot ROC curbe and confusion matrix
        plot_metrics()
        
    
    ############# 
    # section 2 #
    #############
    
    st.header('2 - Select your model')

    st.write('For each algorithm tested, the best model and its parameters are saved in the table below.')
    
    # display best model and its parameters for each algorithm
    st.dataframe(best_models)
    
    # the use can now select the model to predict sentiment on new set of tweets
    sel_model = st.radio('Select the model you want to use to analyse the new tweets:', ('SVM', 'Logistic Regression', 'Naive Bayes'))
    
    ############# 
    # section 3 #
    #############
    
    st.header('3 - View results of prediction with the selected model')
    
    st.write('The optimal model you have selected above was used to analyse a new set of tweets about Schitt\'s Creek. See below how positive and negative sentiments are distributed and some examples of tweets.')
    
     # import new data: unlabelled tweets
    with open('new_tweets.txt', 'r') as f:
        new_data = json. loads(f. read())
    
    # pre process new tweets
    clean_new_tweets = clean_text(new_data)
    new_tweet_text = [tweet['text'] for tweet in clean_new_tweets]
    
    # apply SVM to new tweets
    if sel_model == 'SVM':
        # pick the parameter values from the best model matrix
        best_ngram_range = best_models.iloc[3, 1]
        best_use_idf = best_models.iloc[4, 1]
        best_loss = best_models.iloc[5, 1]
        best_alpha = best_models.iloc[7, 1]
        best_penalty = best_models.iloc[6, 1]
            
        # fit an train the model with best performing paraneters
        model = Pipeline([('vect', CountVectorizer(analyzer='word', ngram_range=best_ngram_range)),
                   ('tfidf', TfidfTransformer(use_idf = best_use_idf)),
                   ('clf', SGDClassifier(loss = best_loss, penalty = best_penalty, 
                                         alpha=best_alpha, random_state=42, 
                                         max_iter=5, tol=None))])
        model.fit(X_train, y_train)
        
            # use newly trained model to predict new tweets labels
        new_pred = model.predict(new_tweet_text)
     
        
    # apply LR to new tweets
    if sel_model == 'Logistic Regression':
        # pick the parameter values from the best model matrix
        best_ngram_range = best_models.iloc[3, 2]
        best_use_idf = best_models.iloc[4, 2]
        best_C = best_models.iloc[8, 2]
        best_max_iter = best_models.iloc[9, 2]
        best_penalty_lr = best_models.iloc[6, 2]
            
            # fit an train the model with best performing paraneters
        model = Pipeline([('vect', CountVectorizer(analyzer='word', ngram_range=best_ngram_range)),
                   ('tfidf', TfidfTransformer(use_idf = best_use_idf)),
                   ('clf', LogisticRegression(C=best_C, max_iter=best_max_iter, 
                                              penalty = best_penalty_lr, random_state=42))])
        model.fit(X_train, y_train)
        
        # use newly trained model to predict new tweets labels
        new_pred = model.predict(new_tweet_text)
        
        
    # apply NB to new tweets
    if sel_model == 'Naive Bayes':
        # pick the parameter values from the best model matrix
        best_ngram_range = best_models.iloc[3, 3]
        best_use_idf = best_models.iloc[4, 3]
        best_alpha_nb = best_models.iloc[8, 3]
        best_fit_prior = best_models.iloc[9, 3]
            
            # fit an train the model with best performing paraneters
        model = Pipeline([('vect', CountVectorizer(analyzer='word', ngram_range=best_ngram_range)),
                   ('tfidf', TfidfTransformer(use_idf = best_use_idf)),
                   ('clf', MultinomialNB(alpha = best_alpha_nb, fit_prior = best_fit_prior))])
        model.fit(X_train, y_train)
        
        # use newly trained model to predict new tweets labels
        new_pred = model.predict(new_tweet_text)

    
    # build a data set with text + new sentiment labels
    
    # import unlabelled and raw tweets
    with open('new_tweets.txt', 'r') as f:
        new_data = json. loads(f. read())
    
    # extract tweet text only
    new_tweet_og_text = [tweet['text'] for tweet in new_data]
    # bring together raw text and sentiment label
    new_labelled_tweets = zip(new_tweet_og_text,list(new_pred))
    zipped_list = list(new_labelled_tweets)
    
    # change to df format for easier data exploratio n
    df_new_tweets = DataFrame (zipped_list,columns=['Text', 'Sentiment'])  
    
    # separate positive and negative tweets 
    ptweets = df_new_tweets[df_new_tweets['Sentiment'] == 'positive']
    ntweets = df_new_tweets[df_new_tweets['Sentiment'] == 'negative']
    
    st.write('Total number of tweets analysed: ', len(df_new_tweets))
    
    # plot percentage of positive and negative tweets
    fig = px.histogram(df_new_tweets, x = 'Sentiment')
    st.plotly_chart(fig)
    
    # show first 5 positive tweets 
    st.write('\n\nPositive tweets:') 
    st.table(ptweets[0:5])

    # show first 5 negative tweets 
    st.write('\n\nNegative tweets:') 
    st.table(ntweets[0:5])
    
    # show the download link for new labelled tweets (result of prediction)
    st.markdown(get_table_download_link(df_new_tweets), unsafe_allow_html=True)
    
if __name__ == '__main__':
    main()
   
