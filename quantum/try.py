from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
analyzer = SentimentIntensityAnalyzer()

text = "  "

scores = analyzer.polarity_scores(text)

a = scores["pos"]
b = scores["neg"]
c = scores["neu"]
d = scores["compound"]

print(a, b, c)
print("Compound score ", d)
subjectivity_score = (max(scores["pos"], scores["neg"]) - min(
    scores["pos"], scores["neg"])) / (scores["pos"] + scores["neg"] + 0.001)

print("Subjectivity score ", subjectivity_score)
