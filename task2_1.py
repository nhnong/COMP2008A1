import re
import pandas as pd
import numpy as np
import nltk
import matplotlib.pyplot as plt
import seaborn as sns

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

import wordcloud
from wordcloud import WordCloud

from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import WordNetLemmatizer

# ----------------------------------------------------------------------------
# root function for task2_1, calls other functions to process the data and create a word cloud
def task2_1():
    accidents_data = pd.read_csv('accident.csv')
    # accidents_data = casefolding_and_noise(accidents_data)
    accidents_data['DCA_DESC'] = accidents_data['DCA_DESC'].apply(lambda x: casefolding_and_noise(x) if type(x)==str else x)
    accidents_data['DCA_DESC'] = accidents_data['DCA_DESC'].apply(lambda x: stopwords_and_stem(x) if type(x)==str else x)
    bow = bag_of_words(accidents_data['DCA_DESC'])
    word_counts = word_counts_dataframe(bow)
    word_cloud = creating_word_cloud(word_counts)
    accidents_data.to_csv('task2_1_accidents.csv', index=False)
    return word_cloud, accidents_data

# ----------------------------------------------------------------------------
# casefolding and noise removal for all columns
def casefolding_and_noise(text):
    text = text.lower()
    text = re.sub(r'[^A-Za-z\s]', '', text)
    return text

# ----------------------------------------------------------------------------
# turning the column accidents_data['DCA_DESC'] into tokens, remove stop words and apply stemming
porter_stemmer = PorterStemmer()
def stopwords_and_stem(text):
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    stemmed = ' '.join([porter_stemmer.stem(word) for word in tokens])
    return stemmed

# ----------------------------------------------------------------------------
# bag of words, coverts text into BoW, where the matrix counts the number of times each word appears in the text
vectorizer = CountVectorizer()
def bag_of_words(dca_des):
    bow = vectorizer.fit_transform(dca_des)
    pd.DataFrame(bow.toarray())
    return bow

# ----------------------------------------------------------------------------
# getting most common words in the vectorizer
def word_counts_dataframe(bow):
    word_counts = np.asarray(bow.sum(axis=0)).flatten()
    vocabulary = vectorizer.get_feature_names_out()
    word_counts = pd.DataFrame(word_counts, index=vocabulary, columns=['count'])
    word_counts = word_counts.sort_values(by='count', ascending=False)
    return word_counts

# ----------------------------------------------------------------------------
# word cloud, visualisation of the most common words in the text
def creating_word_cloud(word_counts):
    word_cloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_counts['count'].to_dict())
    plt.figure(figsize=(10, 5))
    plt.title('Word Cloud of Most Common Words in DCA_DESC')
    plt.imshow(word_cloud, interpolation='bilinear')
    plt.axis('off')
    plt.savefig('task2_1_word_cloud.png', dpi=300, bbox_inches='tight')
    return word_cloud




