from sklearn import metrics
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.metrics import AUC
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras import layers
from copy import deepcopy
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
import numpy as np
#import wget
import pandas as pd
import nltk
# nltk.download('wordnet')


path = "datasets/MentalHealthTweets.xlsx"
train_df = pd.read_excel(path, names=["Tweet_Content", "label"])


train_df = train_df.dropna(axis=1)
train_df.reset_index(drop=True, inplace=True)

try:
    train_df['label'] = train_df['label'].astype('int')
except:
    print("Error in astype")

# print(train_df.info())
# print(train_df.head())

print(train_df['label'].value_counts())


# try:
#     url = 'http://downloads.cs.stanford.edu/nlp/data/glove.6B.zip'
#     filename = wget.download(url)
# except:
#     print("Could not download")


words = dict()


def add_to_dict(d, filename):
    with open(filename, 'r', encoding="utf8") as f:
        for line in f.readlines():
            line = line.split(' ')

            try:
                d[line[0]] = np.array(line[1:], dtype=float)
            except:
                continue


add_to_dict(words, "glove.6B/glove.6B.50d.txt")

# print(words)

# print(len(words))

tokenizer = nltk.RegexpTokenizer(r"\w+")

tokenizer.tokenize('@user when a father is dysfunctional and is')


lemmatizer = WordNetLemmatizer()

lemmatizer.lemmatize('feet')


def message_to_token_list(s):
    tokens = tokenizer.tokenize(s)
    lowercased_tokens = [t.lower() for t in tokens]
    lemmatized_tokens = [lemmatizer.lemmatize(t) for t in lowercased_tokens]
    useful_tokens = [t for t in lemmatized_tokens if t in words]

    return useful_tokens


#print(message_to_token_list('@user feet a fathers is dysfunctional and is'))


def message_to_word_vectors(message, word_dict=words):
    processed_list_of_tokens = message_to_token_list(message)

    vectors = []

    for token in processed_list_of_tokens:
        if token not in word_dict:
            continue

        token_vector = word_dict[token]
        vectors.append(token_vector)

    return np.array(vectors, dtype=float)


#print(message_to_word_vectors('@user when a father is dysfunctional and is').shape)

train_df = train_df.sample(frac=1, random_state=1)
train_df.reset_index(drop=True, inplace=True)

split_index_1 = int(len(train_df) * 0.7)
split_index_2 = int(len(train_df) * 0.85)

train_df, val_df, test_df = train_df[:split_index_1], train_df[split_index_1:split_index_2], train_df[split_index_2:]

#print(len(train_df), len(val_df), len(test_df))

# print(test_df)


def df_to_X_y(dff):
    y = dff['label'].to_numpy().astype(int)

    all_word_vector_sequences = []

    for message in dff['Tweet_Content']:
        message_as_vector_seq = message_to_word_vectors(message)

        if message_as_vector_seq.shape[0] == 0:
            message_as_vector_seq = np.zeros(shape=(1, 50))

        all_word_vector_sequences.append(message_as_vector_seq)

    return all_word_vector_sequences, y


X_train, y_train = df_to_X_y(train_df)

# print(len(X_train), len(X_train[0]))

# print(len(X_train), len(X_train[2]))

sequence_lengths = []

for i in range(len(X_train)):
    sequence_lengths.append(len(X_train[i]))


plt.hist(sequence_lengths)

# plt.show()

# print(pd.Series(sequence_lengths).describe())


def pad_X(X, desired_sequence_length=57):
    X_copy = deepcopy(X)

    for i, x in enumerate(X):
        x_seq_len = x.shape[0]
        sequence_length_difference = desired_sequence_length - x_seq_len

        pad = np.zeros(shape=(sequence_length_difference, 50))

        X_copy[i] = np.concatenate([x, pad])

    return np.array(X_copy).astype(float)


X_train = pad_X(X_train)

# print(X_train.shape)

# print(y_train.shape)

X_val, y_val = df_to_X_y(val_df)
X_val = pad_X(X_val)

#print(X_val.shape, y_val.shape)

X_test, y_test = df_to_X_y(test_df)
X_test = pad_X(X_test)

#print(X_test.shape, y_test.shape)


model = Sequential([])

model.add(layers.Input(shape=(57, 50)))
model.add(layers.LSTM(64, return_sequences=True))
model.add(layers.Dropout(0.2))
model.add(layers.LSTM(64, return_sequences=True))
model.add(layers.Dropout(0.2))
model.add(layers.LSTM(64, return_sequences=True))
model.add(layers.Dropout(0.2))
model.add(layers.Flatten())
model.add(layers.Dense(1, activation='sigmoid'))

# model.summary()


cp = ModelCheckpoint('model/', save_best_only=True)

model.compile(optimizer=Adam(learning_rate=0.0001),
              loss=BinaryCrossentropy(),
              metrics=['accuracy', AUC(name='auc')])

frequencies = pd.value_counts(train_df['label'])
print(frequencies)

weights = {0: frequencies.sum() / frequencies[0], 1: frequencies.sum(
) / frequencies[1], 2: frequencies.sum() / frequencies[2]}
# print(weights)

model.fit(X_train, y_train, validation_data=(X_val, y_val),
          epochs=14, callbacks=[cp], class_weight=weights)


best_model = load_model('model/')

y_pred = (best_model.predict(X_test) > 0.5).astype(int)

cm = metrics.confusion_matrix(y_test, y_pred)

cm_display = metrics.ConfusionMatrixDisplay(
    confusion_matrix=cm)

cm_display.plot()
plt.show()
# y_test = np.array(y_test)

try:
    print(classification_report(y_test, y_pred))
    # print(cm)
    # print(classification_report(y_test, y_pred))
except:
    print("Error")
