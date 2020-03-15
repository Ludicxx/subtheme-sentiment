from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import re
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from skmultilearn.problem_transform import LabelPowerset
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from skmultilearn.problem_transform import ClassifierChain
from xgboost import XGBClassifier,XGBRegressor
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
#########
def remove_special_char(sentence):
    return re.sub(r"[^a-zA-Z0-9.',:;?]+", ' ', sentence)
data = pd.read_csv('sentisum-evaluation-dataset.csv', header=None)
data = data[data[1].notna()]
data = data.reset_index(drop=True)
for i in range(data.shape[1]):
    data[i] = data[i].str.replace('/', '')
data.fillna('', inplace=True)
data['label'] = ""
for i in range(1, 15):
    data['label'] += data[i] + ","
data['label'] = data['label'].str.replace(',,', '')  # .apply(rstrip(','))
data['label'] = data['label'].str.rstrip(',')
data['label'] = data['label'].str.replace(' ', '').replace('/', '')
data['label'] = data['label'].str.replace('positive', '')
data['label'] = data['label'].str.replace('negative','')
# data['label'] = data['label'].str.replace(',',' ')
label = []
for it in data['label']:
	label.append(it.split(","))

data[0] = data[0].str.lower()
data[0] = data[0].str.replace(',','')
data[0] = data[0].map(lambda x: remove_special_char(x))
stop_words = set(stopwords.words('english'))

def replace_pronouns(text):
    words = word_tokenize(text)
    wordsFiltered = ""
    for w in words:
        if w not in stop_words:
            wordsFiltered += (w + " ")
        # print(wordsFiltered)
    return wordsFiltered


data["text_pro"] = data[0].map(lambda x: replace_pronouns(x))

stemmer = SnowballStemmer("english")
def stemming(sentence):
    stemSentence = ""
    for word in sentence.split():
        stem = stemmer.stem(word)
        stemSentence += stem
        stemSentence += " "
    stemSentence = stemSentence.strip()
    return stemSentence
data['text_pro'] = data['text_pro'].apply(stemming)
# data['text_pro'] = data['text_pro'].map(lambda x: remove_special_char(x))

# print(data.head()['text_pro'])
# print(data['label'])


mlb = MultiLabelBinarizer()
# print(data['label'][0][0])
y = mlb.fit_transform(label)
datasetLabel = pd.DataFrame(y)
garage_data = []
for i in range(0,y.shape[0]):
	garage_data.append(y[i][29])
# print(mlb.classes_)
X = data['text_pro']
# Split data into train and test set
train_data, test_data, y_train, y_test = train_test_split(X
    , garage_data, test_size=0.25, random_state=0)

#########
reviewsTrain = train_data
reviewsTest = test_data

filter_length = 300
vocab_size = 10000
embedding_dim = 32
max_length = 200
trunc_type = 'post'
oov_tok = "<OOV>"

tokenizer = Tokenizer(num_words = vocab_size, oov_token = oov_tok, lower=True)
tokenizer.fit_on_texts(reviewsTrain)
wordIndex = tokenizer.word_index

review_sequences_train = tokenizer.texts_to_sequences(reviewsTrain)
X_train = pad_sequences(review_sequences_train, maxlen = max_length, truncating=trunc_type)

review_sequences_test = tokenizer.texts_to_sequences(reviewsTest)
X_test = pad_sequences(review_sequences_test, maxlen = max_length)

model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, 64, input_length=max_length),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        # tf.keras.layers.Dense(32, activation='relu'),
        # tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(1, activation='sigmoid')
        ])
# sampling
#########
over = SMOTE(sampling_strategy=0.4)
under = RandomUnderSampler(sampling_strategy=0.7)
# X_res, y_res = sample.fit_resample(X_train_transform, y_train)
# X_res_1, y_res_1 = over.fit_resample(X_train, y_train)
print(Counter(np.array(y_train)))
X_res, y_res = under.fit_resample(X_train, y_train)
y_res = np.array(y_res)
y_test = np.array(y_test)
print(Counter(np.array(y_res)))
#######

model.compile(optimizer='adam', loss = 'binary_crossentropy' ,metrics=["accuracy"])
print(model.summary())

history = model.fit(X_res, y_res, epochs=50, batch_size=64, validation_data=(X_test, y_test), verbose=2)

scores = model.evaluate(X_test, y_test, verbose=0)

print(scores)

pred = model.predict(X_test)
for idx,i in enumerate(pred):
    for idy,j in enumerate(i):
        if pred[idx][idy] >= .5:
            pred[idx][idy] = 1
        else:
            pred[idx][idy] = 0
pred = np.array(pred)
y_test = np.array(y_test)
print(pred.shape)
print(y_test.shape)
def confusionMatrix(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    confusion = np.zeros((3, 3))
    for i in range(2032):
        if y_true[i] == -1 and y_pred[i] == -1:
            confusion[0][0]+=1
        elif y_true[i] == -1 and y_pred[i] == 0:
            confusion[0][1]+=1
        elif y_true[i] == -1 and y_pred[i] == 1:
            confusion[0][2]+=1
        elif y_true[i] == 0 and y_pred[i] == -1:
            confusion[1][0]+=1
        elif y_true[i] == 0 and y_pred[i] == 0:
            confusion[1][1]+=1
        elif y_true[i] == 0 and y_pred[i] == 1:
            confusion[1][2]+=1
        elif y_true[i] == 1 and y_pred[i] == -1:
            confusion[2][0]+=1
        elif y_true[i] == 1 and y_pred[i] == 0:
            confusion[2][1]+=1
        elif y_true[i] == 1 and y_pred[i] == 1:
            confusion[2][2]+=1
    confusion = confusion.astype(int)
    print(confusion)

confusionMatrix(y_test, pred)
