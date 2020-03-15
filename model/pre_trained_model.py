import pickle,re,json
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import tokenizer_from_json

with open('label_set.pkl', 'rb') as f:
   label_set = pickle.load(f)

with open('tokenizer.json') as f:
    data = json.load(f)
    tokenizer = tokenizer_from_json(data)

with open('finalized_model.sav', 'rb') as handle:
    model = pickle.load(handle)

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

max_length = 120
trunc_type = 'post'

test_case = ['Great price and fast, reliable fitting process.']
test_case_df = pd.DataFrame()
test_case_df[0]=test_case
dataRefactor(test_case_df[0])

test_sequence = tokenizer.texts_to_sequences(test_case_df[0])
padded_test = pad_sequences(test_sequence, maxlen=max_length, truncating=trunc_type)
predicted_test_output = model.predict(padded_test, batch_size=None, verbose=0, steps=None)
predicted_test_output = predicted_test_output[0]
for idx,i in enumerate(predicted_test_output):
    if i >= 0.5:
        print(label_set[idx],end=' ')
