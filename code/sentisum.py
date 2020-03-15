import pandas as pd
import numpy as np
import re, sys
# import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, Flatten, Dropout
from sklearn.metrics import f1_score
import pickle

data = pd.read_csv("../dataset/sentisum-evaluation-dataset.csv", header=None)

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
pickle.dump(label_set, open('../model/label_set.pkl', 'wb'))
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

filter_length = 300
vocab_size = 10000
embedding_dim = 16
max_length = 120
trunc_type = 'post'
oov_tok = "<OOV>"

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok, lower=True)
tokenizer.fit_on_texts(reviews_train)
# word_index = tokenizer.word_index
with open('../model/tokenizer.pkl', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

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

filename = 'finalized_model.sav'
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