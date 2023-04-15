from matplotlib import pyplot as plt
from sklearn import metrics
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np  # linear algebra
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Dense, GlobalMaxPooling1D, Embedding
from tensorflow.keras.models import Model
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns

path = "datasets/MentalHealthTweets.xlsx"
df = pd.read_excel(path, names=["Tweet_Content", "Label"])


# print(df.info())

df = df.dropna(axis=1)
df.reset_index(drop=True, inplace=True)

# print(df.shape)
# print(df.head())

print(df['Label'])
print(df.info())
print(type(df["Label"]))

try:
    df['Label'] = df['Label'].astype('int')
except:
    print("Error in astype")

print(df.info())
print(df.head())

columns = df.columns
columns

df.columns = ['Tweet_Content', 'Label']
df.head()

y = df['Label']

df_train, df_test, y_train, y_test = train_test_split(
    df['Tweet_Content'], y, test_size=0.33, random_state=42)


# BUILDING DEEP LEARNING MODEL

max_words = 10000
tokenizer = Tokenizer(max_words)
tokenizer.fit_on_texts(df_train)
sequence_train = tokenizer.texts_to_sequences(df_train)
sequence_test = tokenizer.texts_to_sequences(df_test)

word2vec = tokenizer.word_index
V = len(word2vec)
print('dataset has %s number of independent tokens' % V)

data_train = pad_sequences(sequence_train)
data_train.shape

T = data_train.shape[1]
data_test = pad_sequences(sequence_test, maxlen=T)
data_test.shape

D = 150
i = Input((T,))
x = Embedding(V+1, D)(i)
x = Conv1D(32, 3, activation='relu')(x)
x = MaxPooling1D(3)(x)
x = Conv1D(64, 3, activation='relu')(x)
x = MaxPooling1D(3)(x)
x = Conv1D(128, 3, activation='relu')(x)
x = GlobalMaxPooling1D()(x)
x = Dense(5, activation='softmax')(x)
model = Model(i, x)
# model.summary()

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])
cnn_senti = model.fit(data_train, y_train, validation_data=(
    data_test, y_test), epochs=15, batch_size=100)

y_pred = model.predict(data_test).round()
print("y_pred before argmax:", y_pred)
y_pred = np.argmax(y_pred, axis=1)
print("\n y_pred after argmax:", y_pred)


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

# print(y_pred)
###### THE PATH ########

# C:\Users\ajayk\OneDrive\Desktop\8th sem\major project\phase-2\cnn.py
