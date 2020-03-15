import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import re
import sys
import warnings
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Embedding, Flatten, Dropout
from sklearn.metrics import f1_score
import pickle

data = pd.read_csv("sentisum-evaluation-dataset.csv", header=None)

# data preprocessing
data = data[data[1].notna()]
data = data.reset_index(drop=True)
shape_data = data.shape
for i in range(shape_data[1]):
    data[i] = data[i].str.replace('/', '')

split_data_from = 6000
data_train = data[:][:split_data_from]
data_test = data[:][split_data_from:]
data_test = data_test.reset_index(drop=True)

shape_data_train = data_train.shape
shape_data_test = data_test.shape

label_hash_map_train = {}
# label_hash_map_train_plot = {}
label_set = []
string_type = type(data_train[1][0])

for i in range(shape_data_train[0]):
    for j in range(1, shape_data_train[1]):
        # only string type data are labels rest are nan
        if type(data_train[j][i]) == string_type:
            extracted_label = data_train[j][i]
            if extracted_label in label_hash_map_train:
                label_hash_map_train[extracted_label][i] = 1
                # label_hash_map_train_plot[extracted_label] = sum(label_hash_map_train[extracted_label])
            else:
                temp = [0] * shape_data_train[0]
                temp[i] = 1
                temp_dict = {extracted_label: temp}
                label_hash_map_train.update(temp_dict)
                label_set.append(extracted_label)
                # label_hash_map_train_plot[extracted_label] = sum(label_hash_map_train[extracted_label])

# plt.bar(range(len(label_hash_map_train_plot)), list(label_hash_map_train_plot.values()), align='center')
# plt.xticks(range(len(label_hash_map_train_plot)), list(label_hash_map_train_plot.keys()), rotation='vertical', fontsize=10)
# plt.show()

label_hash_map_test = {}

for extracted_label in label_hash_map_train:
    temp = [0] * shape_data_test[0]
    temp_dict = {extracted_label: temp}
    label_hash_map_test.update(temp_dict)

for i in range(shape_data_test[0]):
    for j in range(1, shape_data_test[1]):
        if type(data_test[j][i]) == string_type:
            extracted_label = data_test[j][i]
            if extracted_label in label_hash_map_test:
                label_hash_map_test[extracted_label][i] = 1

label_train = pd.DataFrame()
i = 0
for j in label_hash_map_train:
    label_train[i] = label_hash_map_train[j]
    i += 1

label_test = pd.DataFrame()
i = 0
for j in label_hash_map_test:
    label_test[i] = label_hash_map_test[j]
    i += 1



def cleanPunctuation(sentence):  # function to clean the word of any punctuation or special characters
    cleaned = re.sub(r'[?|!|\'|"|#]', r'', sentence)
    cleaned = re.sub(r'[.|,|)|(|\|/]', r' ', cleaned)
    cleaned = cleaned.replace("\n", " ")
    return cleaned


def keepAlpha(sentence):
    alpha_sent = ""
    for word in sentence.split():
        alpha_word = re.sub('[^a-z A-Z]+', ' ', word)
        alpha_sent += alpha_word
        alpha_sent += " "
    alpha_sent = alpha_sent.strip()
    return alpha_sent


def dataRefactor(data_frame):
    data_frame = data_frame.str.lower()
    data_frame = data_frame.apply(cleanPunctuation)
    data_frame = data_frame.apply(keepAlpha)


dataRefactor(data_train[0])
dataRefactor(data_test[0])

reviews_train = data_train[0]
reviews_test = data_test[0]

# Data Generation
def checkLabelIntersection(data1,data2):
    for i in range(1,data1.shape[0]-1):
        if data1[i] + data2[i] == 2:
            return False
    return True

def joinData(data1,data2):
    # temp_data1 = data1.copy()
    # temp_data2 = data2.copy()
    appendedData = pd.DataFrame(data=data1).append(pd.DataFrame(data=data2))
    appendedData[0] = appendedData[0] + ' '
    generatedData = pd.DataFrame(appendedData.sum(axis =0 )).transpose()
    generatedData[0] = generatedData[0].str.rstrip(' ')
    # print(generatedData)


def generateData(review, labels):

    data = pd.concat([review,labels], axis=1, ignore_index = True)


    label_weight = np.array([0]*(data.shape[1]-1))
    for i in range(1,data.shape[1]):
        count_freq = Counter(data[i])
        weight[i-1] = count_freq[1]

    review_weight = np.array([0]*data.shape[0])


    for i in range(1,data.shape[1]):
        for j in range(data.shape[0]):
            review_weight[j] += data[i]*weight[i-1]

    data[data.shape[1]] = review_weight

    data = data.sort_values(by = (data.shape[1]-1), ascending=True)
    data = data.reset_index(drop=True)

    breakingPoint = int(data.shape[0]/2)
    threshold = 3

    for i in range(breakingPoint):
        dataF1 = df.loc[[i]]
        count = 0
        for j in range(i,data.shape[0]):
            dataF2 = df.loc[[j]]
            if checkLabelIntersection(dataF1,dataF2):
                dataF3 = joinData(dataF1,dataF2)
                data = data.append(dataF3, ignore_index=True)
                count+=1
                if count>=threshold:
                    break
    data = data.sort_values(by = (data.shape[1]-1), ascending=True)
    data = data.reset_index(drop=True)
    data = data.drop(data.shape[1]-1, axis = 1)
    labels_gen = data.drop(0, axis = 1)
    review_gen = data[0]

    data.to_pickle("./dummy.pkl")

    return review_gen, labels_gen

reviews_train, label_train = generateData(reviews_train, label_train)



filter_length = 300
vocab_size = 10000
embedding_dim = 16
max_length = 120
trunc_type = 'post'
oov_tok = "<OOV>"

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok, lower=True)
tokenizer.fit_on_texts(reviews_train)
# word_index = tokenizer.word_index

review_sequences_train = tokenizer.texts_to_sequences(reviews_train)
X_train = pad_sequences(review_sequences_train, maxlen=max_length, truncating=trunc_type)

review_sequences_test = tokenizer.texts_to_sequences(reviews_test)
X_test = pad_sequences(review_sequences_test, maxlen=max_length)

# Model

model = Sequential()
model.add(Embedding(vocab_size, 64, input_length=max_length))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dense(59, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, label_train, epochs=50, batch_size=128, validation_data=(X_test, label_test), verbose=2)
evaluation_scores = model.evaluate(X_test, label_test, verbose=0)

filename = '../model/finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))
print(evaluation_scores)

predicted_output = model.predict(X_test)
np.array(predicted_output)
for idx, i in enumerate(predicted_output):
    for idy, j in enumerate(i):
        if predicted_output[idx][idy] >= .5:
            # print(reviews_test[idx],end = '-----')
            # print(label_set[idy],end =' ')
            predicted_output[idx][idy] = 1
        else:
            predicted_output[idx][idy] = 0
    # print()
# print(predicted_output.shape)
print(f1_score(label_test, predicted_output, average='weighted'))

# Sub theme Sentiment prediction

test_case = ['Great price and fast, reliable fitting process.']
test_case_df = pd.DataFrame()
test_case_df[0]=test_case
dataRefactor(test_case_df[0])

test_sequence = tokenizer.texts_to_sequences(test_case_df[0])
padded_test = pad_sequences(test_sequence, maxlen=max_length, truncating=trunc_type)
predicted_test_output = model.predict(padded_test, batch_size=None, verbose=0, steps=None)
predicted_test_output = predicted_test_output[0]
print(predicted_test_output)
for idx,i in enumerate(predicted_test_output):
    if i >= 0.5:
        print(label_set[idx],end=' ')
