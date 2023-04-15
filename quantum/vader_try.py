import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

df = pd.read_csv('datasets\sentiment_tweets - Copy1.csv')
print(df.info())

try:
    # df.insert(2, column="negative_score", value=0.0)
    # df.insert(3, column="positive_score", value=0.0)
    # df.insert(4, column="neutral_score", value=0.0)
    #df.insert(5, column="compound_score", value=0.0)
    df.insert(6, column="subjectivity_score", value=0.0)
except:
    print("Dataset updated already ")

print(df.head())
print(df.info())

#blob = TextBlob(df.iat[8, 1])

# sentence = "I'm not sure if I like this product, but it seems to work well."

# Create a sentiment analyzer object
analyzer = SentimentIntensityAnalyzer()

# Analyze the sentiment of the sentence
sentiment = analyzer.polarity_scores(df.iat[8954, 1])

# Print the sentiment scores
print('Negative score:', sentiment['neg'])
print('Positive score:', sentiment['pos'])
print('Neutral score:', sentiment['neu'])
print('Compound score :', sentiment['compound'])

print("Length = ", len(df['Tweet_Content']))

for i in range(len(df['Tweet_Content'])):
    sentiment = analyzer.polarity_scores(df.iat[i, 1])
    subjectivity_score = (max(sentiment["pos"], sentiment["neg"]) - min(
        sentiment["pos"], sentiment["neg"])) / (sentiment["pos"] + sentiment["neg"] + 0.001)
    subjectivity_score = round(subjectivity_score, 4)
    # Print the sentiment scores
    # print('Negative score:', sentiment['neg'])
    # df.iat[i, 2] = sentiment['neg']
    # df.iat[i, 3] = sentiment['pos']
    # df.iat[i, 4] = sentiment['neu']
    # df.iat[i, 5] = sentiment['compound']
    df.iat[i, 6] = subjectivity_score

print(df.info())
print(df.head())

# saving the dataframes
df.to_csv('datasets\sentiment_tweets - Copy2.csv', index=False)
# print("negative_score", df.iat[2003, 2])


# df = df.iloc[1:]
