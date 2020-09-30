#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 11:47:13 2020

@author: raphaele
"""


import twitter
import json

########################
# import corpus of tweets
########################

#pip install twitter



# initialize api instance
twitter_api = twitter.Api(consumer_key = 'viw8hLXDwoDno2i1HPQLTJ3ci',
                          consumer_secret = 'ho5P83EBBiIjfuZ7sFJS35NzBY5ROADXbIqK7qqtLEztpmgUL5',
                          access_token_key = '1025065834455158785-zp6vmG244kGC4fCrvjnV33pBzKeaGT',
                          access_token_secret = '7pL3dClgrF4C5i6fWX2NQVindYDe6mazPxSlfy6I4Btyx',
                          sleep_on_rate_limit=True)

# test authentication
print(twitter_api.VerifyCredentials())


 # define a subset of the full corpus for testing the function


import pandas as pd
#import random

# remove the observations labelled as irrelevant and neutral



#p = 0.04  # 1% of the lines
# keep the header, then take only 1% of lines
# if random from [0,1] interval is greater than 0.01 the row will be skipped
#df = pd.read_csv(
#         'corpus.csv',
#         header=0, 
#         skiprows=lambda i: i>0 and random.random() > p
#)

df = pd.read_csv(
         'corpus.csv',
         header=0)

df = df.drop(df[(df.label == 'irrelevant')].index)
df = df.drop(df[(df.label == 'neutral')].index)


# see sentiment distribution
df.label.value_counts()

# convert to small cvs file
df.to_csv (r'/Users/raphaele/Documents/Py/twitter_sa/corpus.csv', index = True, index_label='index_name', header=True)

df

# get the corpus of labelled tweets. Based on the method described at:
# https://towardsdatascience.com/creating-the-twitter-sentiment-analysis-program-in-python-with-naive-bayes-classification-672e5589a7ed


def build_data_set(labelled_ids, tweets_data_set):
    import csv

   

    corpus=[]
    
    with open(labelled_ids,'r') as csvfile:
        lineReader = csv.reader(csvfile, delimiter=',', quotechar="\"")
        for row in lineReader:
            corpus.append({"tweet_id":row[3], "label":row[2], "topic":row[1]})
    
    #rate_limit=180
    #sleep_time=900/180
    
    full_data_set=[]

    for tweet in corpus:
        try:
            status = twitter_api.GetStatus(tweet["tweet_id"])
            print("Tweet fetched: " + status.text)
            tweet["text"] = status.text
            full_data_set.append(tweet)
        except: 
            continue
    # Now we write them to the empty CSV file
    with open(tweets_data_set,'w') as csvfile:
        linewriter=csv.writer(csvfile,delimiter=',',quotechar="\"")
        for tweet in full_data_set:
            #try:
            linewriter.writerow([tweet["tweet_id"],tweet["text"],tweet["label"],tweet["topic"]])
            #except Exception as e:
                #print(e)
    return full_data_set

# ------------------------------------------------------------------------

labelled_ids = "/Users/raphaele/Documents/Py/twitter_sa/corpus.csv"
tweets_data_set = "/Users/raphaele/Documents/Py/twitter_sa/tweets_data_set.csv"

model_data = build_data_set(labelled_ids, tweets_data_set)

type(model_data)
len(model_data)

# export list of tweets to txt fil

with open('model_tweets.txt', 'w') as f:
    f. write(json. dumps(model_data))
    


########################
# build new corpus woth search term
########################    
    

results1 = twitter_api.GetSearch(raw_query="q=schitt%27s%20creek%20lang%3Aen%20since%3A2020-09-02&src=typed_query&count=100")   
results2 = twitter_api.GetSearch(raw_query="q=schitt%27s%20creek%20lang%3Aen%20since%3A2020-09-11&src=typed_query&count=100")   
results3 = twitter_api.GetSearch(raw_query="q=schitt%27s%20creek%20lang%3Aen%20since%3A2020-09-19&src=typed_query&count=100")   
results4 = twitter_api.GetSearch(raw_query="q=schitt%27s%20creek%20lang%3Aen%20since%3A2020-&src=typed_query&count=100")   

results[0]['Text']
results.text

type(results1)

# bring together raw text and sentiment label
results_zip = zip(results1, results2, results3, results4)
results = list(results_zip)

results = results1 + results2 + results3 + results4

len(results)

# view unique values sorted in ascending order
sorted(results.ID.unique())

tweets = [] 

# parsing tweets one by one 
for tweet in results: 
    # empty dictionary to store required params of a tweet 
    parsed_tweet = {} 

    # saving text of tweet 
    parsed_tweet['text'] = tweet.text
    tweets.append(parsed_tweet) 
    
tweets[2]
# export new tweets
with open('new_tweets.txt', 'w') as f:
    f. write(json. dumps(tweets))

    
