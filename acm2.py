import os
import sys
import re
import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.layers import Activation, Dense, Input, GRU, Embedding, Dropout, GlobalMaxPool1D, Bidirectional
from keras.models import Model
from sklearn.preprocessing import LabelBinarizer
import stop_words
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

LOGS_FOLDER = 'logs'
MODELS_FOLDER = 'models'
TRAIN_FILE = 'train.csv'
# This can depend on if the embedding contains cased words or not.
DO_LOWER_CASE = False
# When True, remove punctuation.
DO_REMOVE_PUNCTUATION = False
# When True, perform stop word removal.
DO_REMOVE_STOP_WORDS = True
# When True, perform stemming.
DO_STEMMING = True


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


def get_spread(labels):
    spread = dict()
    for label in labels:
        try:
            spread[label] += 1
        except KeyError:
            spread[label] = 1
    return spread


def get_model(max_features, label_count):
    inp = Input(shape=(max_features,))
    x = Dense(64, activation='relu')(inp)
    x = Dropout(0.5)(x)
    x = Dense(label_count)(x)
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
    max_features = 8000  # The maximum number of total unique words to use.
    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(train_lines)
    # tokenizer.fit_on_texts(test_lines)  # Can we do this?
    train_word_counts = tokenizer.texts_to_matrix(train_lines, mode='count')
    test_word_counts = tokenizer.texts_to_matrix(test_lines, mode='count')
    max_word_counts = np.array([max(train_word_counts[:, col]) for col in range(len(train_word_counts[0]))])
    min_word_counts = np.array([min(train_word_counts[:, col]) for col in range(len(train_word_counts[0]))])
    x_train = np.nan_to_num((train_word_counts - min_word_counts) / (max_word_counts - min_word_counts))
    x_test = np.nan_to_num((test_word_counts - min_word_counts) / (max_word_counts - min_word_counts))
    # Transform the labels into a one-hot encoding.
    encoder = LabelBinarizer()
    y_train = encoder.fit_transform(train_labels)
    y_test = encoder.fit_transform(test_labels)
    # Build the model.
    model = get_model(max_features, label_count)
    print(model.summary())
    optimizer = 'adam'
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['categorical_accuracy'])
    history = model.fit(x_train,
                        y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_split=val_ratio)
    # Save the model and the history.
    save_str = 'dense1-{}{}{}{}{}-{}-{}'.format(
        max_features,
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
