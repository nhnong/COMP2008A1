import re
import pandas as pd
import numpy as np
import nltk


nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

# directory: cd "C:\Users\anegl\OneDrive\Documents\2025 S1 COMP20008 A1"
# cmd prompt: py task2_1.py
# print(accidents_data)
# print(accidents_data.columns.tolist())

accidents_data = pd.read_csv('accident.csv')
print('initial')

# casefolding for all columns, converting all string values to lowercase
for column in accidents_data.columns:
    accidents_data[column] = accidents_data[column].apply(lambda x: x.lower() if type(x)==str else x)
print('casefolding done')


# noise removal for all columns, removing all non-alphanumeric characters
for column in accidents_data.columns:
    accidents_data[column] = accidents_data[column].apply(lambda x: re.sub(r'[^A-Za-z\s]', '', x) if type(x)==str else x)
print('nose removal done')

# turning the column accidents_data['DCA_DESC'] into tokens, and remove stop words
def remove_stopwords(text):
    if type(text)==str:
        tokens = nltk.word_tokenize(text)
        # put non stop words into list and join them back into a string
        text = ' '.join([word for word in tokens if word not in stop_words])
    return text

accidents_data['DCA_DESC'] = accidents_data['DCA_DESC'].apply(lambda x: remove_stopwords(x) if type(x)==str else x)

print(accidents_data['DCA_DESC'][0:10])
print('stop words removed')
print('works')


def task2_1():
    
    return