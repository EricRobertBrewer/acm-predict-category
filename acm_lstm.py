import os
import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Activation, Dense, Input, LSTM, Embedding, Dropout, GlobalMaxPool1D, Bidirectional
from keras.models import Model
from sklearn.preprocessing import LabelBinarizer
import stop_words
from stemming.porter2 import stem
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

TRAIN_FILE = 'train.csv'
DO_PREPROCESS = False
EMBEDDINGS_FILE = os.path.join('embeddings', 'glove.6B.50d.txt')


def preprocess_text(sentences):
    """
    Perform stop-word removal and stemming.
    """
    lines = list()
    english_stop_words = stop_words.get_stop_words('en')
    for sentence in sentences:
        sentence_words = [stem(word) for word in sentence.split() if word not in english_stop_words]
        lines.append(' '.join(sentence_words))
    return lines


def get_coefs(word, *arr):
    """
    Read the GloVe word vectors.
    """
    return word, np.asarray(arr, dtype='float32')


def main():
    columns = ['class', 'title', 'u1', 'authors', 'source', 'publisher', 'citations', 'abstract', 'keywords']
    data = pd.read_csv(TRAIN_FILE)
    data.dropna(subset=['class', 'title', 'abstract'])
    print('# of rows: {}'.format(len(data)))
    label_count = data['class'].nunique()
    print('# of unique labels: {}'.format(label_count))
    labels = np.array(data['class'])
    titles = list(data['title'])
    abstracts = list(data['abstract'])
    sentences = [titles[i] + ' ' + abstracts[i] for i in range(len(titles))]
    if DO_PREPROCESS:
        lines = preprocess_text(sentences)
    else:
        lines = np.array(sentences)
    # Shuffle the labels and lines.
    permutation = np.random.permutation(labels.shape[0])
    labels = labels[permutation]
    lines = lines[permutation]
    # Split the data.
    train_ratio = 0.8
    val_ratio = 0.25  # Proportion of the TRAINING data, not of the entire data set.
    train_end = int(train_ratio * len(lines))
    train_labels = labels[:train_end]
    train_lines = lines[:train_end]
    test_labels = labels[train_end:]
    test_lines = lines[train_end:]
    # Pre-process; tokenize the text and transform it into [padded] sequences for the RNN.
    # See: https://www.kaggle.com/sbongo/for-beginners-tackling-toxic-using-keras
    # See: https://www.kaggle.com/jhoward/improved-lstm-baseline-glove-dropout
    max_features = 20000  # The maximum number of total unique words to use.
    maxlen = 100  # The maximum number of words to pass through the network at a time.
    embed_size = 50  # The dimensionality of the word embedding.
    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(train_lines)
    train_sequences = tokenizer.texts_to_sequences(train_lines)
    test_sequences = tokenizer.texts_to_sequences(test_lines)
    x_train = pad_sequences(train_sequences, maxlen=maxlen)
    x_test = pad_sequences(test_sequences, maxlen=maxlen)
    # Prepare the GloVe embedding.
    with open(EMBEDDINGS_FILE, 'r', encoding='utf8') as fd:
        embeddings_index = dict(get_coefs(*o.strip().split()) for o in fd)
    all_embeddings = np.stack(embeddings_index.values())
    embeddings_mean, embeddings_std = all_embeddings.mean(), all_embeddings.std()
    word_index = tokenizer.word_index
    word_count = min(max_features, len(word_index))
    # Use these vectors to create our embedding matrix, with random initialization for words that aren't in GloVe.
    # We'll use the same mean and standard deviation of embeddings the GloVe has when generating the random init.
    embedding_matrix = np.random.normal(embeddings_mean, embeddings_std, (word_count, embed_size))
    for word, i in word_index.items():
        if i >= max_features:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    # Transform the labels into a one-hot encoding.
    encoder = LabelBinarizer()
    y_train = encoder.fit_transform(train_labels)
    y_test = encoder.fit_transform(test_labels)
    # Build the model.
    inp = Input(shape=(maxlen,))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
    x = Bidirectional(LSTM(50, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(x)
    x = GlobalMaxPool1D()(x)
    x = Dense(50, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(label_count)(x)
    x = Activation('softmax')(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['categorical_accuracy'])
    batch_size = 32
    epochs = 50
    model.fit(x_train,
              y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_split=val_ratio)
    # Test.
    # predictions = model.predict([test_sequences], batch_size=1024, verbose=1)
    model.save('lstm2.h5')
    score = model.evaluate(x_test, y_test, batch_size=1024, verbose=1)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])


if __name__ == '__main__':
    main()
