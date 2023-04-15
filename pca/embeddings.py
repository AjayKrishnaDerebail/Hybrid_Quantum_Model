from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding
import pandas as pd


tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(tweets)
sequences = tokenizer.texts_to_sequences(tweets)

max_len = 20
padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post')

model = Sequential()
model.add(Embedding(input_dim=10000, output_dim=100, input_length=max_len))
model.compile('rmsprop', 'mse')

tweet_features = model.predict(padded_sequences)
