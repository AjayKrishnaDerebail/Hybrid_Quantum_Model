##### We use BILSTM HERE #######
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from collections import Counter
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
np.random.seed(0)
# Plotting

# nltk
# nltk.download('stopwords')
# nltk.download('wordnet')

# nltk.download('punkt')

# Tensorflow / Keras
try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from textblob import TextBlob
    from collections import Counter
except:
    print("Couldn't run import statements ! ")
# Test


# Training Data
path = "datasets\\reddit.csv"
train_df = pd.read_csv(path)

train_df = train_df.dropna(axis=1)
train_df.shape

df = train_df.sample(frac=1, random_state=1)
# df = train_df
df.reset_index(drop=True, inplace=True)

# Data Exploration
# See overall information about the data frame
print(df.info())

# print(df.head)


# Checking Balance
# Checking balance of target classes
sentiments = list(df["label"].unique())

sentiment_nums = [len(df[df["label"] == sentiment]) / len(df)
                  for sentiment in sentiments]

plt.bar(sentiments, sentiment_nums)

plt.show()


# We can create an indexer to convert sentiments from labels to indexes, and back again. This is useful in understanding our predictions later on.

# We then convert the "Sentiment" column in the training data to the labels, which is what we will learn to predict

class_to_index = {"0": 0, "1": 1}

# Creates a reverse dictionary
index_to_class = dict((v, k) for k, v in class_to_index.items())

# Creates lambda functions, applying the appropriate dictionary


def names_to_ids(n): return np.array([class_to_index.get(x) for x in n])
def ids_to_names(n): return np.array([index_to_class.get(x) for x in n])


# Test each function

# print(names_to_ids([0, 1, 2]))
# print(ids_to_names(["0", "1", "2"]))

# Convert the "Sentiment" column into indexes
#df["label"] = names_to_ids(df["label"])

# df['Label'] = df['Label'].astype('int64')


def remove_stopwords(ls):
    # Lemmatises, then removes stop words
    ls = [lemmatiser.lemmatize(word) for word in ls if word not in (
        stop_english) and (word.isalpha())]

    # Joins the words back into a single string
    ls = " ".join(ls)
    return ls


# Splits each string into a list of words
df["Tweet_Content_Split"] = df["Tweet_Content"].apply(word_tokenize)

# Applies the above function to each entry in the DataFrame
lemmatiser = WordNetLemmatizer()
# Here we use a Counter dictionary on the cached
stop_english = Counter(stopwords.words())
# list of stop words for a huge speed-up
df["Tweet_Content_Split"] = df["Tweet_Content_Split"].apply(remove_stopwords)

# print(df.head())


# Tokenization

# Define the Tokeniser
tokeniser = Tokenizer(num_words=10000, lower=True)

# Create the corpus by finding the most common
tokeniser.fit_on_texts(df["Tweet_Content_Split"])
# Tokenise our column of edited Tweet content
tweet_tokens = tokeniser.texts_to_sequences(list(df["Tweet_Content_Split"]))
# Pad these sequences to make them the same length
tweet_tokens = pad_sequences(
    tweet_tokens, truncating='post', padding='post', maxlen=50)
print(tweet_tokens.shape)


y = df["label"]

# Drop all non-useful columns
df = pd.DataFrame(tweet_tokens)
# Display final shape
print(df.shape)

print(df.info())


X_train, X_test, y_train, y_test = train_test_split(
    df, y, test_size=0.3, random_state=1)

model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(10000, 16, input_length=50),
    tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(20, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(20)),
    tf.keras.layers.Dense(4, activation='softmax')
])

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

h = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=1,
    callbacks=[tf.keras.callbacks.EarlyStopping(
        monitor='accuracy', patience=5)]
)

# print("H = ", h)
# print("X_test \n", X_test)
# print("y_test \n", y_test)

y_pred = np.argmax(model.predict(X_test), axis=1)

# print("Numpy arr = \n", y_pred)

# Assign labels to predictions and test data
# y_pred_labels = ids_to_names(y_pred)
# print("funcpred arr = \n", y_pred_labels)
# y_test_labels = ids_to_names(y_test)
# print("Functest arr = \n", y_test_labels)

cm = confusion_matrix(y_test, y_pred,
                      normalize='true')

ac = accuracy_score(y_test, y_pred)
cm = metrics.confusion_matrix(y_test, y_pred)
print(ac)
# print(cm)

cm_display = metrics.ConfusionMatrixDisplay(
    confusion_matrix=cm)

cm_display.plot()

plt.show()
print(classification_report(y_test, y_pred))
#disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=y_unique)


######################### THE PATH ###########################

# python -u "C:\Users\ajayk\OneDrive\Desktop\8th sem\major project\phase-2\bilstm\\bilstm2.py"
