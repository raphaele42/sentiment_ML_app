#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Created on Wed Sep 23 11:47:13 2020

# get the corpus of labelled tweets with text. Based on the method described at:
# https://towardsdatascience.com/creating-the-twitter-sentiment-analysis-program-in-python-with-naive-bayes-classification-672e5589a7ed

import twitter
import json
import pandas as pd



# initialize api instance
twitter_api = twitter.Api(consumer_key = 'xxxxxxxxxxxxxxxx',
                          consumer_secret = 'xxxxxxxxxxxxxxxx',
                          access_token_key = 'xxxxxxxxxxxxxxxx',
                          access_token_secret = 'xxxxxxxxxxxxxxxx',
                          sleep_on_rate_limit=True)

# test authentication
print(twitter_api.VerifyCredentials())

########################
# import and cleand list 
# of tweets ID and sentiment
########################

# import the corpus of tweet IDs and labels
df = pd.read_csv('corpus.csv', header=0)

# remove the observations labelled as irrelevant and neutral
df = df.drop(df[(df.label == 'irrelevant')].index)
df = df.drop(df[(df.label == 'neutral')].index)

# see sentiment distribution
df.label.value_counts()

# convert to cvs file
df.to_csv (r'/Users/raphaele/Documents/Py/twitter_sa/corpus.csv', index = True, index_label='index_name', header=True)


###########################
# build the full corpus 
# by concatenating ids, sentiment and text
###########################

def build_data_set(labelled_ids, tweets_data_set):
    import csv
    
    corpus=[]
    
    with open(labelled_ids,'r') as csvfile:
        lineReader = csv.reader(csvfile, delimiter=',', quotechar="\"")
        for row in lineReader:
            corpus.append({"tweet_id":row[3], "label":row[2], "topic":row[1]})
    
    full_data_set=[]

    for tweet in corpus:
        try:
            status = twitter_api.GetStatus(tweet["tweet_id"])
            print("Tweet fetched: " + status.text)
            tweet["text"] = status.text
            full_data_set.append(tweet)
        except: 
            continue
    # write them to the empty CSV file
    with open(tweets_data_set,'w') as csvfile:
        linewriter=csv.writer(csvfile,delimiter=',',quotechar="\"")
        for tweet in full_data_set:
            linewriter.writerow([tweet["tweet_id"],tweet["text"],tweet["label"],tweet["topic"]])
    return full_data_set

# ------------------------------------------------------------------------

labelled_ids = "/Users/raphaele/Documents/Py/twitter_sa/corpus.csv"
tweets_data_set = "/Users/raphaele/Documents/Py/twitter_sa/tweets_data_set.csv"

# this is the full data set: ID + text + sentiment
model_data = build_data_set(labelled_ids, tweets_data_set)

# export list of tweets to txt file
# for re-use by the app
with open('model_tweets.txt', 'w') as f:
    f. write(json. dumps(model_data))
    


########################
# build new corpus 
# with search term
########################    
    
# here search term is: Schitt's Creek
# searching on 4 different days
results1 = twitter_api.GetSearch(raw_query="q=schitt%27s%20creek%20lang%3Aen%20since%3A2020-09-02&src=typed_query&count=100")   
results2 = twitter_api.GetSearch(raw_query="q=schitt%27s%20creek%20lang%3Aen%20since%3A2020-09-11&src=typed_query&count=100")   
results3 = twitter_api.GetSearch(raw_query="q=schitt%27s%20creek%20lang%3Aen%20since%3A2020-09-19&src=typed_query&count=100")   
results4 = twitter_api.GetSearch(raw_query="q=schitt%27s%20creek%20lang%3Aen%20since%3A2020-&src=typed_query&count=100")   

# bring together the four days of tweets
results_zip = zip(results1, results2, results3, results4)
results = list(results_zip)
results = results1 + results2 + results3 + results4

# initiate an object to store the tweets text
tweets = [] 

# parsing tweets one by one 
for tweet in results: 
    # empty dictionary to store required params of a tweet 
    parsed_tweet = {} 

    # saving text of tweet 
    parsed_tweet['text'] = tweet.text
    tweets.append(parsed_tweet) 
    
# export new tweets
with open('new_tweets.txt', 'w') as f:
    f. write(json. dumps(tweets))

    
