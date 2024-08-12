#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install imblearn')


# In[ ]:


#import libraries
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.pylab as pylab
import numpy as np
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from matplotlib import pyplot
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import plotly.express as px    
get_ipython().system('pip install nltk')
import nltk
nltk.download([
   "names",
   "stopwords",
   "state_union",
   "wordnet"
   "twitter_samples",
   "movie_reviews",
   "averaged_perceptron_tagger",
   "vader_lexicon",
   "punkt",
])
import nltk
nltk.download('stopwords')
get_ipython().system('pip install wordcloud')
 
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from wordcloud import WordCloud
from nltk.sentiment import SentimentIntensityAnalyzer


# In[ ]:


from sklearn.impute import SimpleImputer 
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer


# In[ ]:


import random
import re
import sys
import os


# In[ ]:


#read in data set
Climate = pd.read_csv('/Users/Faderemi/Desktop/Climate_change.csv')


# In[ ]:


Climate.head()


# In[ ]:


Climate.tail()


# In[ ]:


Climate.describe()


# In[ ]:


Climate.info()


# In[ ]:


#check for missing values
Climate.isna().any()

print("Missing values distribution: ")
print(Climate.isnull().mean())
print("")


# In[ ]:


# remove rows with any missing values
Climate = Climate.dropna()

print(Climate)


# In[ ]:


Climate.info()


# In[ ]:


Climate.isna().any()


# In[ ]:


# Sort mentions in a tweet and place in column named mentions
Climate['mentions'] = Climate['Embedded_text'].apply(lambda x: x.count('@'))

# Sort hashtags in a tweet and put it in column named hashtags
Climate['hashtags'] = Climate['Embedded_text'].apply(lambda x: x.count('#'))

# Count URLs - new column with true or false
def contains_url(text):
    url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    return bool(url_pattern.search(text))

Climate['link'] = Climate['Embedded_text'].apply(contains_url)

Climate['link'] = Climate['link'].astype(int)


# In[ ]:


Climate.head(15)


# In[ ]:


#check for images in tweets
def contains_image(row):
  text = row['Embedded_text']
  if 'pic.twitter.com' in text:
    return True
  else:
    return False


# In[ ]:


#create rows with images
Climate["contains_image"] = Climate.apply(count_urls, axis= 1)


# In[ ]:


Climate.columns


# In[ ]:


Climate["contains_image"].value_counts()


# In[ ]:


# Remove commas as number separators
Climate['Likes'] = Climate['Likes'].str.replace(',', '')
Climate['Retweets'] = Climate['Retweets'].str.replace(',', '')

# Express likes and retweets in full numbers by making K = 1000 and M = 1000000
Climate['Likes']=Climate['Likes'].replace({'K': '*1000', 'M': '*100000'}, regex=True).map(pd.eval).astype(int)
Climate['Retweets']=Climate['Retweets'].replace({'K': '*1000', 'M': '*1000000'}, regex=True).map(pd.eval).astype(int)


# In[ ]:


print('Count of rows in the data is:  ', len(Climate))


# In[ ]:


Climate['Timestamp'] = pd.to_datetime(Climate['Timestamp'], infer_datetime_format=True)
Climate.info()


# In[ ]:


import nltk  
from nltk.corpus import stopwords
stop = stopwords.words("english")

#extend stopwords list
my_stopwords = ['https', 'I', 'That','This','There','amp', 'It']
for i in my_stopwords:
    stop.append(i)
print(stop)
#apply
Climate["Embedded_text"] = Climate["Embedded_text"].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))


# In[ ]:


# Define a function to remove short words from a text
def remove_short_words(text):
    words = text.split()
    filtered_words = [word for word in words if len(word) > 4]
    filtered_text = " ".join(filtered_words)
    return filtered_text


# In[ ]:


Climate['Embedded_text'] = Climate['Embedded_text'].apply(remove_short_words)


# In[ ]:


print(Climate)


# In[ ]:


wordcloud = WordCloud(width=600,
                      height=400,
                      random_state=2,
                      max_font_size=100).generate(' '.join(Climate['Embedded_text']))


# In[ ]:


plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off');


# In[ ]:


def preprocess_tweet(tweet):
    # remove URLs, mentions, and hashtags
    tweet = re.sub(r'http\S+', '', tweet)
    tweet = re.sub(r'@\w+', '', tweet)
    tweet = re.sub(r'#\w+', '', tweet)
    # remove special characters and punctuation
    tweet = re.sub(r'[^\w\s]', '', tweet)
    # convert to lowercase
    tweet = tweet.lower()
    return tweet


# In[ ]:


#Apply part of speech tagging:
import nltk
from nltk.tag import pos_tag
import nltk
nltk.download('averaged_perceptron_tagger')
#tokenize tweets
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize


# In[ ]:


def tokenize_tweet(tweet):
    tokens = word_tokenize(tweet)
    return tokens


# In[ ]:


import nltk
nltk.download('punkt')
#tokenize the tweet into words
Climate['Embedded_text'] = Climate['Embedded_text'].apply(lambda x: word_tokenize(x))
Climate.head()


# In[ ]:


Climate['Embedded_text'] = Climate['Embedded_text'].astype(str)


# In[ ]:


from nltk import pos_tag, word_tokenize

Climate['tokens'] = Climate['Embedded_text'].apply(word_tokenize)
Climate['POS_tags'] = Climate['tokens'].apply(pos_tag)


# In[ ]:


Climate['Embedded_text'] = Climate['Embedded_text'].astype(str)


# In[ ]:


from nltk.probability import FreqDist
fdist = FreqDist(Climate["Embedded_text"])
print(fdist)


# In[ ]:


fdist.most_common(5)


# In[ ]:


#DATA VISUALISATION
Climatex = Climate[Climate['Timestamp'].dt.year.isin([2022])]
# Convert the date column to a pandas DatetimeIndex and extract the year and hour c
year = pd.DatetimeIndex(Climatex['Timestamp']).year
hour = pd.DatetimeIndex(Climatex['Timestamp']).hour
month = pd.DatetimeIndex(Climatex['Timestamp']).month
# Categorize the hours into morning, afternoon, or evening bins
bins = [0, 12, 18, 23]
labels = ['morning', 'afternoon', 'evening']
time_of_day = pd.cut(hour, bins=bins, labels=labels)

# Group the tweets into month and time of day, and count the number of tweets in each
grouped = Climatex.groupby([month, time_of_day])['Embedded_text'].count().unstack()


# In[ ]:


# create a new column for the month of each tweet
Climate['month'] = pd.to_datetime(Climate['Timestamp']).dt.month

Climate['day']= pd.to_datetime(Climate['Timestamp']).dt.day

Climate.head(15)


# In[ ]:


sns.catplot(data= Climate, x= 'month', y='Likes', kind='bar')


# In[ ]:


# Create a stacked bar chart to visualize the tweet frequency by time of day and ye
ax = grouped.plot(kind='bar', stacked=True, title='Tweet Frequency by Time of Day')
ax.set_xlabel('Month')
ax.set_ylabel('Number of Tweets')
ax.legend(title='Month')
plt.show()


# In[ ]:


#plot the most common words
# Obtain top 10 words
top_10 = fdist.most_common(10)
# add pandas series to plot
fdist = pd.Series(dict(top_10))
#plot common words
sns.set_theme(style="ticks")
sns.barplot(y=fdist.index, x=fdist.values, color='purple');


# In[ ]:


sns.catplot(data=Climate, x='month', y='Likes', kind='swarm')


# In[ ]:


sns.catplot(data=Climate, x='month', y='Retweets', kind='bar')


# In[ ]:


Climate.info()


# In[ ]:


#Performing Sentiment Analysis
get_ipython().system('pip install textblob')
from textblob import TextBlob

def get_tweet_sentiment(tweet):
    # create a TextBlob object for the tweet
    blob = TextBlob(tweet)
    # calculate the sentiment polarity of the tweet (-1 to 1)
    sentiment_polarity = blob.sentiment.polarity
    #group sentiment into positive, negative, or neutral based on the polarity score
    if sentiment_polarity > 0:
        sentiment = 'positive'
    elif sentiment_polarity < 0:
        sentiment = 'negative'
    else:
        sentiment = 'neutral'
    return sentiment


# In[ ]:


#Aggregate the Sentiment Scores and Calculating Summary Statistics:

def calculate_sentiment_stats(sentiments):
    # count the number of positive, negative, and neutral tweets
    positive_count = sentiments.count('positive')
    negative_count = sentiments.count('negative')
    neutral_count = sentiments.count('neutral')
    # calculate the mean, median, and standard deviation of the sentiment scores
    sentiment_scores = [1 if sentiment == 'positive' else -1 if sentiment == 'negative' else 0 for sentiment in sentiments]
    sentiment_mean = sum(sentiment_scores) / len(sentiment_scores)
    sentiment_median = sorted(sentiment_scores)[len(sentiment_scores) // 2]
    sentiment_stddev = (sum([(score - sentiment_mean) * 2 for score in sentiment_scores]) / len(sentiment_scores)) * 0.5
    return positive_count, negative_count, neutral_count, sentiment_mean, sentiment_median, sentiment_stddev


# In[ ]:


#Evaluate accuracy of Sentiment Analysis Algorithm:

def evaluate_sentiment_accuracy(sentiments, labels):
    # compare the predicted sentiments to the true labels and calculate the accuracy
    correct_count = 0
    total_count = len(sentiments)
    for i in range(total_count):
        if sentiments[i] == labels[i]:
            correct_count += 1
    accuracy = correct_count / total_count
    return accuracy


# In[ ]:


Climate['tweet_length'] = Climate['Embedded_text'].apply(len)


# In[ ]:


# Perform sentiment analysis on the text of each tweet
Climate['polarity'] = Climate['Embedded_text'].apply(lambda x: TextBlob(x).sentiment.polarity)
Climate['sentiment'] = Climate['polarity'].apply(lambda x: 'Positive' if x > 0 else ('Negative' if x < 0 else 'Neutral'))


# In[ ]:


# Create a DataFrame with the counts of positive, negative, and neutral tweets
sentiment_counts = Climate.groupby(['sentiment'])['Embedded_text'].count().reset_index()


# In[ ]:


# Create a stacked bar chart of the sentiment analysis results
sns.set_style('whitegrid')
plt.bar(sentiment_counts['sentiment'], sentiment_counts['Embedded_text'], color=['yellow', 'purple', 'red'])
plt.title('Sentiment Analysis of Tweets')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()


# In[ ]:


Climate.info()


# In[ ]:


#Sort data by number of likes and choose top 5 tweets
top5_tweets = Climate.sort_values("Likes", ascending=False).head(5)


# In[ ]:


for i, tweet in top5_tweets.iterrows():
    print(f"{tweet['Embedded_text']}\nLikes: {tweet['Likes']}\n")


# In[ ]:


#polarity score for top 5 tweets
top5_tweets['polarity'] = top_5_tweets['Embedded_text'].apply(lambda x: TextBlob(x).sentiment.polarity)
top5_tweets['sentiment'] = top_5_tweets['polarity'].apply(lambda x: 'Positive' if x > 0 else ('Negative' if x < 0 else 'Neutral'))


# In[ ]:


# Create dataFrame showing counts of positive, negative, and neutral tweets
sentiment_counts = top5_tweets.groupby(['sentiment'])['Embedded_text'].count().reset_index()


# In[ ]:


# Create a stacked bar chart of the sentiment analysis results
sns.set_style('whitegrid')
plt.bar(sentiment_counts['sentiment'], sentiment_counts['Embedded_text'], color=['yellow', 'purple', 'red'])
plt.title('sentiment analysis of Top 5 tweets with most likes')
plt.xlabel('sentiment')
plt.ylabel('count')
plt.show()


# In[ ]:


Climate['polarity'] = Climate['polarity'].astype(int)


# In[ ]:


Climate.info()


# In[ ]:


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# In[ ]:


# set a random seed for reproducibility
np.random.seed(40379140)


# In[ ]:


# separate the target variable from the features
Y = Climate["Likes"]
X = Climate[["Retweets", "mentions","hashtags","link","contains_image","month","day","tweet_length","polarity"]]

# split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20)


# In[ ]:


# Run linear regression model
Climate_model = LinearRegression()

Climate_model.fit(X, Y)

# make predictions on the training set, calculate MSE and r2_score
Y_pred = Climate_model.predict(X)
Climate_mse = mean_squared_error(Y, Y_pred)

# print the model coefficients and MSE
print("Model coefficients:", Climate_model.coef_)
print("Mean squared error:", Climate_mse)


# In[ ]:


from sklearn.ensemble import RandomForestRegressor


# In[ ]:


# Run random forest model
Climate_model = RandomForestRegressor(n_estimators=100, random_state=45)

# fit model to training data
Climate_model.fit(X_train, Y_train)

# make predictions on the training set, calculate MSE and r2_score
Y_pred = Climate_model.predict(X_test)

Climate_mse = mean_squared_error(Y_test, Y_pred)


# print the MSE
print("Mean squared error:", Climate_mse)


# In[ ]:


from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler


# In[ ]:


# scale the data using standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Run the SVM model
Climate_model = SVR(kernel="linear", C=1)

# fit the model to the training data
Climate_model.fit(X_train_scaled, Y_train)

# make predictions on the training set, calculate MSE and r2_score
Y_pred = Climate_model.predict(X_test_scaled)

Climate_MSE = mean_squared_error(Y_test, Y_pred)

# print the mean squared error and SVM
print("Mean squared error:", Climate_MSE)


# In[ ]:




