# Sentiment Analysis of a corpus of tweets
Sentiment analysis of social media posts.
 Online app is here: https://twsent.herokuapp.com/
 do you see the change?
 
This project goal is to predict the sentiment on a corpus of tweets as positive or negative, to allow for further analysis. An app was published for users to select the best model and apply it to a new set of tweets. It includes bulding a corpus of labelled tweets, selecting models to compare and allow users to select the most performant model via a web app.

Technologies:
Python
Scikit Learn
Streamlit

Table of content



 
## 1 - Corpus

The model will be trained on a set of 1220 tweets labelled as positive or negative. The corpus is built by collating a list of labelled IDs with the corresponding tweet text, via the Twitter API. Output: a text file to train the model and a text file with new unclassified tweets.

Full code available [here](https://github.com/raphaele42/sentiment_a/blob/master/tweet_corpus.py).




## 2 - Model selection

**Goal:** explore three algortithms to determine if they are relevant to perform this task and what range of hyper parameters can ve provided to users for optimisation.

**Algorithms:** Logistic Regression is a quick and robust algorithm for binary classification problems like this one. Support Vector Machine (SVM) and Naive Bayes have also shown good results for text classification.

**Methodology:** for each algorithm, a pipeline  is built for vectorisation, transformation, feature reduction (if applicable) and classification. Then a grid search is applied to the pipeline with a range of parameter values. This way we can determine which combination of parameters yields the most performant model.

### Import modules
```
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

```
def load_data():
    # import tweet data as collected in tweet_corpus.py
    # read the tweets file to a list
    with open('model_tweets.txt', 'r') as f:
        model_data = json. loads(f. read())
    return model_data
```
### Pre-processing data

Cleaning the tweets invlove turning all text into lower case standard characters, to prepare it for vectorisation. Then the data is split into prediction (y) and explanatory (x) variables.

```

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

```
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

```
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


### Logistic regression

```
# pipeline data vectorisation => transformation => feature reduction => classifier
lg_clf = Pipeline([('vect', CountVectorizer()),
                   ('tfidf', TfidfTransformer()),
                   ('pca', TruncatedSVD()),
                   ('clf', LogisticRegression())])

# parameter tuning with grid search
# train and test model with all combinations of parameters
# to identify most performant
lg_parameters = {'vect__ngram_range': [(1, 1), (1, 2)], 
                                       'tfidf__use_idf': (True, False),
                                       'pca__n_components': [400, 450, 500]
                                       }

# set grid search instance
gs_lg_clf = GridSearchCV(lg_clf, lg_parameters, cv=5, n_jobs=-1)

# fit the grid search instance
gs_lg_clf = gs_lg_clf.fit(X_train, y_train)

# which performed best?
gs_lg_clf.best_score_  # accuracy 0.79 withoput PCA , it is 0.66 with PCA
for param_name in sorted(lg_parameters.keys()):
    print("%s: %r" % (param_name, gs_lg_clf.best_params_[param_name]))
# best parameters:
#pca__n_components: 450
#tfidf__use_idf: True
#vect__ngram_range: (1, 1)
    
```

Note about PCA: it does not improve the performance of the Logistic Regression modela and increases the running time significantly. For this reason, no feature reduction will be applied to these models.

### Naive Bayes

```
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

### SVM

```
# pipeline data vectorisation => transformation => feature reduction => classifier
svm_clf = Pipeline([('vect', CountVectorizer()),
                   ('tfidf', TfidfTransformer()),
                   ('pca', TruncatedSVD(n_components = 450)),
                   ('clf', SGDClassifier(
                                         alpha=1e-3, random_state=42,
                                         max_iter=5, tol=None))])

# parameter tuning with grid search
# train and test model with all combinations of parameters
# to identify most performant
svm_parameters = {'vect__ngram_range': [(1, 1), (1, 2)], 
                                       'tfidf__use_idf': (True, False),
                                       #'pca__n_components': [440],
                                       'clf__loss': ('hinge','squared_hinge'),
                                       'clf__penalty': ('l2', 'l1', 'elasticnet')
                                       }
# set grid search instance
svm_nb_clf = GridSearchCV(svm_clf, svm_parameters, cv=5, n_jobs=-1)

# fit the grid search instance
svm_nb_clf = svm_nb_clf.fit(X_train, y_train)

# which performed best?
svm_nb_clf.best_score_  # 0.8173259310663891
for param_name in sorted(svm_parameters.keys()):
    print("%s: %r" % (param_name, svm_nb_clf.best_params_[param_name]))
# best parameters
#clf__loss: 'hinge'
#clf__penalty: 'l2'
#tfidf__use_idf: True
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

## 3 - Application



The app works as follows:
 
1 - Train three different algorithms and optimise them to get the best models for each one.

2 - Select the best model to analyse the sentiment in a new set of tweets.

3 - Get a preview of the results and download the csv file with labelled tweets.

Screen shots / screencast

Table of content (inline: https://github.com/amitmerchant1990/electron-markdownify#readme)

BAnner

Diagram of the process?


Summary

Technologies
