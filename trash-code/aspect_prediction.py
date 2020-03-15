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
from xgboost import XGBClassifier, XGBRegressor
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# nltk.download('punkt')
# nltk.download('stopwords')


data = pd.read_csv('sentisum-evaluation-dataset.csv', header=None)
# data cleaning
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
data['label'] = data['label'].str.replace('negative', '')

label = []
for it in data['label']:
    label.append(it.split(","))


def remove_special_char(sentence):
    return re.sub(r"[^a-zA-Z0-9.',:;?]+", ' ', sentence)


data[0] = data[0].str.lower()
data[0] = data[0].str.replace(',', '')
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


data["text_conv"] = data[0].map(lambda x: replace_pronouns(x))

stemmer = SnowballStemmer("english")


def stemming(sentence):
    stemSentence = ""
    for word in sentence.split():
        stem = stemmer.stem(word)
        stemSentence += stem
        stemSentence += " "
    stemSentence = stemSentence.strip()
    return stemSentence


data['text_conv'] = data['text_conv'].apply(stemming)
# data['text_conv'] = data['text_conv'].map(lambda x: remove_special_char(x))

# print(data.head()['text_conv'])
# print(data['label'])


mlb = MultiLabelBinarizer()
# print(data['label'][0][0])
y = mlb.fit_transform(label)
datasetLabel = pd.DataFrame(y)
garage_data = []
for i in range(0, y.shape[0]):
    garage_data.append(y[i][29])
# print(mlb.classes_)
X = data['text_conv']
# Split data into train and test set
X_train, X_test, y_train, y_test = train_test_split(X, garage_data, test_size=0.25, random_state=0)

# trainData = pd.concat([X_train, y_train], axis=1)
# trainData_X = trainData['text_conv']
# trainData_y = trainData.drop('text_conv',axis = 1)
# print(trainData_y)
vectorizer = TfidfVectorizer(strip_accents='unicode', stop_words="english", ngram_range=(1, 1), norm='l2')
vectorizer.fit(X_train)
vectorizer.fit(X_test)
X_train_transform = vectorizer.transform(X_train)
x_test_transform = vectorizer.transform(X_test)
print(Counter(np.array(y_train)))

over = SMOTE(sampling_strategy=0.3)
under = RandomUnderSampler(sampling_strategy=0.8)
# X_res, y_res = sample.fit_resample(X_train_transform, y_train)
X_res_1, y_res_1 = over.fit_resample(X_train_transform, y_train)
X_res, y_res = under.fit_resample(X_res_1, y_res_1)
print(Counter(np.array(y_res)))

pipeline = Pipeline([('clf', XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                         colsample_bytree=1, gamma=0, learning_rate=0.1, max_delta_step=0,
                         max_depth=8, min_child_weight=1, missing=None, n_estimators=1000,
                         n_jobs=-1, nthread=None, objective='reg:logistic', random_state=2,
                         reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
                         silent=True, subsample=1)), ])
pipeline = pipeline.fit(X_res, y_res)
predicted = pipeline.predict(x_test_transform)
for it in range(0, len(predicted)):
    if (predicted[it] >= 0.5):
        predicted[it] = 1
    else:
        predicted[it] = 0

print(f1_score(y_test, predicted))