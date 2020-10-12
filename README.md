![SA banner](https://github.com/raphaele42/sentiment_a/blob/master/Sentiment.png "sentiment analysis") ![python logo](https://github.com/raphaele42/sentiment_ML_app/blob/master/python.png "python logo") ![streamlit logo](https://github.com/raphaele42/sentiment_ML_app/blob/master/streamlit.png "streamlit logo")
# Sentiment Analysis of a corpus of tweets
 
**Summary:** The goal of this project is to predict the sentiment on a corpus of tweets as positive or negative. An app was developed for users to select the best model and apply it to a new set of tweets. The steps of the project are: 
- building a corpus of labelled tweets,
- pre-selecting models to compare,
- developing an app for users to select the most performant model.

**Results:** The goal for this project was to reach the industry benchmark for sentiment analysis accuracy: 70-80%. The goal was exceeded as the accuracy of each model was equal or above 80%.

**Try the app:** [https://twsent.herokuapp.com/](https://twsent.herokuapp.com/)

![Preview of section 2](https://github.com/raphaele42/sentiment_a/blob/master/preview.png "Preview")

[Preview video (1.23 minutes)](http://www.youtube.com/watch?v=p4adZ2ZYfAo)

**Technologies:**
* Python
* Scikit Learn
* Streamlit
* Heroku

**Table of content**
* [Corpus](#corpus)
* [Model selection](#model)
* [Application](#application)


<a name="corpus"/>

## 1 - Corpus

The model will be trained on a set of 1220 tweets labelled as positive or negative. The corpus is built by collating a list of labelled IDs with the corresponding tweet text, via the Twitter API. Output: a text file to train the model and a text file with new unclassified tweets.

Full code available [here](https://github.com/raphaele42/sentiment_a/blob/master/tweet_corpus.py).


<a name="model"/>

## 2 - Model selection

**Goal:** explore three algortithms to determine if they are relevant to perform this task and what range of hyper parameters can ve provided to users for optimisation.

**Algorithms:** Logistic Regression is a quick and robust algorithm for binary classification problems like this one. Support Vector Machine (SVM) and Naive Bayes have also shown good results for text classification.

**Methodology:** for each algorithm, a pipeline  is built for vectorisation, transformation, feature reduction (if applicable) and classification. Then a grid search is applied to the pipeline with a range of parameter values. This way we can determine which combination of parameters yields the most performant model.

### Import modules
```python
import json
import numpy as np
from pandas import Series, DataFrame
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import TruncatedSVD
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
```

### Import data

Importing the data set that was built [here](https://github.com/raphaele42/sentiment_a/blob/master/tweet_corpus.py). 

```python
def load_data():
    # import tweet data as collected in tweet_corpus.py
    # read the tweets file to a list
    with open('model_tweets.txt', 'r') as f:
        model_data = json. loads(f. read())
    return model_data
```
### Pre-processing data

Cleaning the tweets invlove turning all text into lower case standard characters, to prepare it for vectorisation. Then the data is split into prediction (y) and explanatory (x) variables.

```python
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
```

### Feature extraction

To make the x variables compatible with the selected algorithm, the simple text must be broken down into word units (tokenisation) of single words (1-grams) or 2 words (2-grams). The data is then turned into numbers (vectorisation) and the frequency of each word units is computed (tf-idf).

```python
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
```

### Call all data prep functions

```python
def prep_data():
    # get data
    model_data = load_data()
    # clean data
    clean_data = clean_text(model_data)
    # split into prediction variable (y) and explanatory variables (x)
    y_var, x_var = split_variables(clean_data)
    # split x data into training and test set on a 80/20 basis
    X_train, X_test, y_train, y_test = train_test_split(x_var, y_var, test_size=0.2, random_state=42)
    return (X_train, X_test, y_train, y_test)

X_train, X_test, y_train, y_test = prep_data()
```
The output of this step are four sets of data:
* X_train: transformed text that will be used to train the model
* y_train: labels matching X_train, used as reference to train the model 
* X_test: transformed text that will be used to make label prediction with the trained model
* y_test: transformed text that will be compared to classes predicted by the model for X_test data

The algorithms (Logistic regression, SVM and Naive Bayes) have been tested with a range of parameters and values to determine which ones could asignificantly impact the model's performance. They don't all appear in the code below as this was an incremental process of trial and error.

Note about PCA: it does not improve the performance of the Logistic Regression modela and increases the running time significantly. For this reason, no feature reduction will be applied to these models.

### Example of pipeline with grid search with Naive Bayes

```python
# pipeline data vectorisation => transformation => classifier
nb_clf = Pipeline([('vect', CountVectorizer()),
                   ('tfidf', TfidfTransformer()),
                   #('pca', TruncatedSVD()),
                   # rescale matrix after pca to remove negative values
                   #('scale', MinMaxScaler(feature_range=(0, 1))),
                   ('clf', MultinomialNB())])

# parameter tuning with grid search
# train and test model with all combinations of parameters
# to identify most performant
nb_parameters = {'vect__ngram_range': [(1, 1), (1, 2)], 
                                       'tfidf__use_idf': (True, False),
                                       #'pca__n_components': [5, 15, 30, 45, 64],
                                       'clf__alpha': (1e-2, 1e-3),}
# set grid search instance
gs_nb_clf = GridSearchCV(nb_clf, nb_parameters, cv=5, n_jobs=-1)

# fit the grid search instance
gs_nb_clf = gs_nb_clf.fit(X_train, y_train)

# which performed best?
gs_nb_clf.best_score_  # accuracy 0.8113115891741846
for param_name in sorted(nb_parameters.keys()):
    print("%s: %r" % (param_name, gs_nb_clf.best_params_[param_name]))
# best parameters
#clf__alpha: 0.01
#tfidf__use_idf: False
#vect__ngram_range: (1, 1)
```

### Conclusions

After running these 3 algorithms with a range of parameters and values, decisions are made for the app offering:
* Feature extraction options: n-gram size and use of idf.
* Parameters for Logistic Regression: `C`, `max_iter` and `penalty`.
* Parameters for SVM: `loss`, `penalt`y and `alpha`.
* Parameters for Naive Bayes: `alpha` and `fit_prior`.
* No feature reduction will be offered.

Full code fir this step is here: https://github.com/raphaele42/sentiment_a/blob/master/tw_sa_mod.py


<a name="application"/>

## 3 - Application

Full coode is [here](https://github.com/raphaele42/sentiment_a/blob/master/tw_sa_app.py).

The [app](https://twsent.herokuapp.com/) works as follows:
 
1. Train three different algorithms and optimise them to get the best models for each one. Store the hyper parameters of the best model. 

2. Allow user to select the best model to analyse the sentiment in a new set of tweets.

3. Give a preview of the results and download the csv file with labelled tweets.


### Object for best models

This dataframe will store the scores and parameter values for the best performing scores. As the streamlit script will re run each time a new parameter is selected in the widgets, this object is cached so it will only be updated when a new best score is found.

```python
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
```

### Function to download the new labelled tweets

```python
def get_table_download_link(df):
    # generate a link allowing the labelled data generated bu the model to be downloaded
    #in:  dataframe
    #out: href string
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}">Download csv file</a>'
    return href
```

### Example of side bar for the SVM model

```python    
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
```
      
### Example of applying the selected models to new data set with Logistic Regression

```python    
    
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
```

### Build the data set of newly labelled tweets and add the download link

```python
        
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
    
```


