import os
import io
import sys
import re
import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Activation, Dense, Input, GRU, Embedding, Dropout, GlobalMaxPool1D, Bidirectional
from keras.models import Model
from sklearn.preprocessing import LabelBinarizer
import stop_words
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

LOGS_FOLDER = 'logs'
MODELS_FOLDER = 'models'
TRAIN_FILE = 'train.csv'
# This can depend on if the embedding contains cased words or not.
DO_LOWER_CASE = True
# When True, remove punctuation.
DO_REMOVE_PUNCTUATION = True
# When True, perform stop word removal.
DO_REMOVE_STOP_WORDS = True
# When True, perform stemming.
DO_STEMMING = False
# GloVe: https://nlp.stanford.edu/projects/glove/
EMBEDDING_GLOVE_50 = 'glove.6B.50d.txt'
EMBEDDING_GLOVE_300 = 'glove.6B.300d.txt'
# fastText: https://fasttext.cc/docs/en/english-vectors.html
EMBEDDING_FASTTEXT_300 = 'wiki-news-300d-1M.vec'
EMBEDDING_FILES = [os.path.join('..', 'embeddings', e) for e in [EMBEDDING_GLOVE_300]]
DO_TRAIN_EMBEDDING = False


def remove_stop_words(sentences):
    """
    Perform stop-word removal and stemming.
    """
    _sentences = list()
    english_stop_words = stop_words.get_stop_words('en')
    for sentence in sentences:
        words = [word for word in sentence.split() if word not in english_stop_words]
        _sentences.append(' '.join(words))
    return _sentences


def get_embedding(tokenizer, file_name=None, max_features=20000):
    """
    Get the embedding matrix.
    :param tokenizer: The Tokenizer which tokenized our text.
    :param file_name: The path to the embedding file.
    :param max_features: The upper limit on the number of words (vectors) to include in the matrix.
    :return: The length of word vectors, the embedding matrix.
    """
    word_index = tokenizer.word_index
    word_count = min(max_features, len(word_index))
    if file_name is None:
        embed_size = 64
        return embed_size, np.random.uniform(-1, 1, (word_count, embed_size))
    if 'glove' in file_name:
        with open(file_name, 'r', encoding='utf8') as fd:
            embeddings_index = dict(get_coefs(*o.strip().split()) for o in fd)
    else:
        embeddings_index = load_vectors(file_name)
    all_embeddings = np.stack(embeddings_index.values())
    embed_size = all_embeddings.shape[1]
    mean, std = all_embeddings.mean(), all_embeddings.std()
    # Use these vectors to create our embedding matrix, with random initialization for words that aren't in GloVe.
    # We'll use the same mean and standard deviation of embeddings that GloVe has when generating the random init.
    embedding_matrix = np.random.normal(mean, std, (word_count, embed_size))
    for word, i in word_index.items():
        if i >= max_features:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embed_size, embedding_matrix


def get_coefs(word, *arr):
    """
    Read the GloVe word vectors.
    """
    return word, np.asarray(arr, dtype='float32')


def load_vectors(file_name):
    """
    Read the fastText word vectors.
    """
    fin = io.open(file_name, 'r', encoding='utf-8', newline='\n', errors='ignore')
    # The first line contains the number of rows (n) and the dimensionality (d)
    n, d = map(int, fin.readline().split())
    data = dict()
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = map(float, tokens[1:])
    fin.close()
    return data


def get_spread(labels):
    spread = dict()
    for label in labels:
        try:
            spread[label] += 1
        except KeyError:
            spread[label] = 1
    return spread


def get_model(maxlen, embedding_matrix, time_steps, label_count):
    inp = Input(shape=(maxlen,))
    if embedding_matrix is not None:
        input_dim = embedding_matrix.shape[0]
        output_dim = embedding_matrix.shape[1]
        x = Embedding(input_dim, output_dim, weights=[embedding_matrix], trainable=DO_TRAIN_EMBEDDING)(inp)
    else:
        x = inp
    x = Bidirectional(GRU(time_steps, return_sequences=True, dropout=0.2, recurrent_dropout=0.1))(x)
    # x = Bidirectional(GRU(64, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(x)
    x = GlobalMaxPool1D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(label_count)(x)
    x = Dropout(0.2)(x)
    x = Activation('softmax')(x)
    return Model(inputs=inp, outputs=x)


def main():
    # columns = ['class', 'title', 'u1', 'authors', 'source', 'publisher', 'citations', 'abstract', 'keywords']
    data = pd.read_csv(TRAIN_FILE)
    data.dropna(subset=['class', 'title', 'abstract'])
    print('Rows of data: {}'.format(len(data)))
    label_count = data['class'].nunique()
    print('Unique labels: {}'.format(label_count))
    labels = np.array(data['class'])
    titles = list(data['title'])
    abstracts = list(data['abstract'])
    if DO_LOWER_CASE:
        sentences = [titles[i].lower() + ' ' + abstracts[i].lower() for i in range(len(titles))]
    else:
        sentences = [titles[i] + ' ' + abstracts[i] for i in range(len(titles))]
    if DO_REMOVE_PUNCTUATION:
        sentences = [re.sub(r'[~`!@#$%^&*()_\-+={}\[\]|\\:;"\'<>,.?/]+', ' ', sentence) for sentence in sentences]
    if DO_REMOVE_STOP_WORDS:
        sentences = remove_stop_words(sentences)
    lines = np.array(sentences)
    # Shuffle the labels and lines.
    permutation = np.random.permutation(labels.shape[0])
    labels = labels[permutation]
    lines = lines[permutation]
    # Split the data for training and testing.
    train_ratio = 0.8
    val_ratio = 0.25  # Proportion of the TRAINING data, not of the entire data set.
    train_end = int(train_ratio * len(lines))
    train_labels = labels[:train_end]
    train_lines = lines[:train_end]
    test_labels = labels[train_end:]
    test_lines = lines[train_end:]
    # Print class spreads in each data set.
    val_start = int((1 - val_ratio) * train_end)
    train_spread = get_spread(train_labels[:val_start])
    test_spread = get_spread(test_labels)
    val_spread = get_spread(train_labels[val_start:])
    longest_label = max(len(label) for label in train_spread.keys())
    for label in train_spread.keys():
        print('{}{}: TR:{} VAL:{} TS:{}'.format(label,
                                                ''.join(' ' for _ in range(longest_label - len(label))),
                                                train_spread[label] / val_start,
                                                val_spread[label] / (train_end - val_start),
                                                test_spread[label] / len(test_labels)))
    # Pre-process; tokenize the text and transform it into [padded] sequences for the RNN.
    # See: https://www.kaggle.com/sbongo/for-beginners-tackling-toxic-using-keras
    # See: https://www.kaggle.com/jhoward/improved-lstm-baseline-glove-dropout
    max_features = 10000  # The maximum number of total unique words to use.
    maxlen = 50  # The maximum number of words to pass through the network at a time.
    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(train_lines)
    train_sequences = tokenizer.texts_to_sequences(train_lines)
    test_sequences = tokenizer.texts_to_sequences(test_lines)
    x_train = pad_sequences(train_sequences, maxlen=maxlen)
    x_test = pad_sequences(test_sequences, maxlen=maxlen)
    # Prepare the embedding(s).
    embed_sizes, embedding_matrices = list(), list()
    for embedding_file in EMBEDDING_FILES:
        embed_size, embedding_matrix = get_embedding(tokenizer, embedding_file, max_features)
        embed_sizes.append(embed_size)
        embedding_matrices.append(embedding_matrix)
    embed_size, embedding_matrix = None, None
    if len(embed_sizes) > 0 and len(embedding_matrices) > 0:
        embed_size, embedding_matrix = embed_sizes[0], embedding_matrices[0]
        for i in range(1, len(embed_sizes)):
            if embed_sizes[i] != embed_size:
                raise Exception('Embeddings sizes are different: {} and {}.'
                                .format(embed_size, embed_sizes[i]))
            if embedding_matrices[i].shape != embedding_matrix.shape:
                raise Exception('Embedding matrices have different shapes: {} and {}'
                                .format(embedding_matrix.shape, embedding_matrices[i].shape))
            # Concatenate all embeddings matrices.
            embedding_matrix += embedding_matrices[i]
    # Transform the labels into a one-hot encoding.
    encoder = LabelBinarizer()
    y_train = encoder.fit_transform(train_labels)
    y_test = encoder.fit_transform(test_labels)
    # Build the model.
    time_steps = 32  # Should probably be less than `maxlen`.
    model = get_model(maxlen, embedding_matrix, time_steps, label_count)
    print(model.summary())
    optimizer = 'rmsprop'
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['categorical_accuracy'])
    history = model.fit(x_train,
                        y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_split=val_ratio)
    # Save the model and the history.
    save_str = 'gru1-{}-g{}-{}{}{}{}{}-{}-{}'.format(
        maxlen,
        '-rawembedding' if not DO_TRAIN_EMBEDDING else '',
        time_steps,
        '-lower' if DO_LOWER_CASE else '',
        '-nopunc' if DO_REMOVE_PUNCTUATION else '',
        '-nostop' if DO_REMOVE_STOP_WORDS else '',
        '-stem' if DO_STEMMING else '',
        optimizer,
        epochs)
    print('Saving model `{}.h5`...'.format(save_str))
    model.save(os.path.join(MODELS_FOLDER, '{}.h5'.format(save_str)))
    print('Saving training history `history-{}.txt`...'.format(save_str))
    with open(os.path.join(LOGS_FOLDER, 'history-{}.txt'.format(save_str)), 'w') as fd:
        for key in history.history.keys():
            values = history.history.get(key)
            fd.write(key + ' ' + ' '.join(str(value) for value in values) + '\n')
    # Test.
    # predictions = model.predict([test_sequences], batch_size=1024, verbose=1)
    test_loss, test_accuracy = model.evaluate(x_test, y_test, batch_size=1024, verbose=1)
    print('Test loss:', test_loss)
    print('Test accuracy:', test_accuracy)


if __name__ == '__main__':
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        raise Exception('Usage: <epochs> [<batch-size>]')
    epochs = int(sys.argv[1])
    batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    main()
