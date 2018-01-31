import numpy as np
import itertools
import os.path
import keras
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.preprocessing import sequence
from sklearn import decomposition
import matplotlib.pyplot as plt

np.random.seed(2018)
vocab_size = 2000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocab_size)
vocab = set(itertools.chain.from_iterable(X_train))
print(len(vocab))

# pad dataset to a maximum review length in words
max_length = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_length)
y_train = keras.utils.to_categorical(y_train, 2)
y_test = keras.utils.to_categorical(y_test, 2)

# id to word
word_to_id = keras.datasets.imdb.get_word_index()
word_to_id["<PAD>"] = 0
word_to_id["<START>"] = 1
word_to_id["<UNK>"] = 2
id_to_word = {value:key for key,value in word_to_id.items()}

# define the model
model = Sequential()
model.add(Embedding(vocab_size, 100, input_length=max_length))
model.add(Conv1D(filters=32, kernel_size=8, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(10, activation='relu'))
model.add(Dense(2, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

if os.path.exists("model_weights.h5"):
    model.load_weights("model_weights.h5")
else:
    # fit the model
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=4, batch_size=64, verbose=2)
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print('Accuracy: %f' % (accuracy*100))
    model.save_weights("model_weights.h5")

# plot
weights = model.layers[0].get_weights()[0]
pca = decomposition.PCA(n_components=2)
pca.fit(weights.T)
fig, ax = plt.subplots()
ax.scatter(pca.components_[0], pca.components_[1])
for i in vocab:
    word = id_to_word[i]
    ax.annotate(word, (pca.components_[0, i],pca.components_[1, i]))
fig.savefig('embedding.png')
plt.show()

