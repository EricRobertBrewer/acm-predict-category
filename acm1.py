import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
import keras
import h5py
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelBinarizer
import numpy as np
import random
import string
from stop_words import get_stop_words
from stemming.porter2 import stem
stop_words = get_stop_words('en')


batch_size = 128 #Change the number of batch size here
epochs = 50 #Change the number of training epoch here

with open('acm_result1-6.txt', encoding="utf8") as f: #Change the input file here
    lines = f.readlines()

with open('test.txt') as f: #Change the text file here
    tests = f.readlines()


def keras_text( lines ):
    docs = []
    for line in lines:
        attributes = line.split('|')
        if (len(attributes) < 8):
            continue
        if (attributes[7] == ""):
            continue
        if (attributes[1] == ""):
            continue
        paragraph = attributes[7] + " " + attributes[1]
        paragraph = [w for w in paragraph.split() if not w in stop_words]
        paragraph = [stem(word) for word in paragraph]
        words = ' '.join(paragraph)
        docs.append(words)
    return docs

def keras_text_ori( lines ):
    docs = []
    for line in lines:
        attributes = line.split('|')
        paragraph = attributes[7] + " " + attributes[1]
        docs.append(paragraph)
    return docs

def doc2vecProcess( lines ):
    taggedList = []
    i = 1
    for line in lines:
        attributes = line.split('|')
        if (len(attributes) < 8):
            continue
        if (attributes[7] == ""):
            continue
        words = "".join([c for c in attributes[7] if c in string.ascii_letters or c in string.whitespace])
        words = words.split(' ')
        while "" in words: words.remove("")
        tags = []
        tags.append(i)
        taggedList.append(TaggedDocument(words,tags))
        i+=1
    vectorSize = 15
    model = Doc2Vec(taggedList, vector_size=vectorSize, alpha=0.025, min_alpha=0.025)
    vectorList = []

    labelList = []
    for line in lines:
        attributes = line.split('|')
        if (len(attributes) < 8):
            continue
        if (attributes[7] == ""):
            continue
        words = "".join([c for c in attributes[7] if c in string.ascii_letters or c in string.whitespace])
        vector = model.infer_vector(words);
        vector1 = []
        for num in vector:
            vector1.append(num)
        vectorList.append(vector1)
        labelList.append("".join(attributes[0].split()))

docs = keras_text(lines)
tests1 = keras_text_ori(tests)

t = Tokenizer()
t.fit_on_texts(docs)
t.fit_on_texts(tests1)
print(t.word_counts)

encoded_docs = t.texts_to_matrix(docs, mode='count')
encoded_test = t.texts_to_matrix(tests1, mode='count')

vectorList = []
testvectorList = []
for doc in encoded_docs:
    vectorList.append(doc)

for test in encoded_test:
    testvectorList.append(test)

labelList = []
for line in lines:
    attributes = line.split('|')
    if (len(attributes) < 8):
        continue
    if (attributes[7] == ""):
        continue
    if (attributes[1] == ""):
        continue
    labelList.append("".join(attributes[0].split()))


c = list(zip(vectorList, labelList))
random.shuffle(c)

testCount = 1500 #Change the number of test count here

vectorList[:], labelList[:] = zip(*c)

x_test = vectorList[:testCount]
x_train = vectorList[testCount:]
y_test = labelList[:testCount]
y_train = labelList[testCount:]
unique, counts = np.unique(y_train, return_counts=True)
print(dict(zip(unique, counts)))
unique, counts = np.unique(y_test, return_counts=True)
print(dict(zip(unique, counts)))
num_classes = 14 #If there are more classes, change it here

# tokenizer = Tokenizer(num_words=max_words)
# x_train = tokenizer.sequences_to_matrix(x_train, mode='binary')
# x_test = tokenizer.sequences_to_matrix(x_test, mode='binary')
# print('x_train shape:', x_train.shape)
# print('x_test shape:', x_test.shape)
# ["socialandprofessionaltopics", "appliedcomputing", "computingmethodologies", "human-centeredcomputing", "securityandprivacy"
#                                 , "informationsystems", "mathematicsofcomputing", "theoryofcomputation", "softwareanditsengineering", "hardware"
#                                             , "generalandreference"]
print('Convert class vector to binary class matrix '
      '(for use with categorical_crossentropy)')
encoder = LabelBinarizer()
y_train = encoder.fit_transform(y_train)
y_test = encoder.fit_transform(y_test)
print(y_train)
print(y_test)
print('Building model...')
#This Section is the neural network
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(len(vectorList[0]),)))
model.add(Dropout(0.5))
model.add(Dense(512))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['categorical_accuracy'])

history = model.fit(np.array(x_train), np.array(y_train),
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_split=0.1)

result = model.predict_classes(np.array(x_test))
print(result)
result = model.predict_classes(np.array(testvectorList))
f = open('result2.txt', "w+") #Output file of the test result
result1 = np.nditer(result)
num = 0
for line in result1:
    attributes = tests[num].split('|')
    if (len(attributes) < 8):
        num += 1
        continue
    if (attributes[7] == ""):
        num += 1
        continue
    if (attributes[1] == ""):
        num += 1
        continue
    if line == 0:
        line = 'appliedcomputing'
    if line == 1:
        line = 'computersystems'
    if line == 2:
        line = 'computersystemsorganization'
    if line == 3:
        line = 'computingmethodologies'
    if line == 4:
        line = 'generalandreference'
    if line == 5:
        line = 'hardware'
    if line == 6:
        line = 'human-centeredcomputing'
    if line == 7:
        line = 'informationsystems'
    if line == 8:
        line = 'mathematicsofcomputing'
    if line == 9:
        line = 'networks'
    if line == 10:
        line = 'securityandprivacy'
    if line == 11:
        line = 'socialandprofessionaltopics'
    if line == 12:
        line = 'softwareanditsengineering'
    if line == 13:
        line = 'theoryofcomputation'
    f.write(attributes[7] + "|||" + str(line) + '\n')
    num+=1

# serialize model to JSON
model.save('my_model.h5')
# serialize weights to HDF5
print("Saved model to disk")
score = model.evaluate(np.array(x_test), np.array(y_test),
                       batch_size=batch_size, verbose=1)
print('Test score:', score [0])
print('Test accuracy:', score[1])
