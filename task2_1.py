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
# print(accidents_data)
# print(accidents_data.columns.tolist())

accidents_data = pd.read_csv('accident.csv')
print('initial')

# ----------------------------------------------------------------------------

# casefolding for all columns, converting all string values to lowercase
for column in accidents_data.columns:
    accidents_data[column] = accidents_data[column].apply(lambda x: x.lower() if type(x)==str else x)
print('casefolding done')

# ----------------------------------------------------------------------------

# noise removal for all columns, removing all non-alphanumeric characters
for column in accidents_data.columns:
    accidents_data[column] = accidents_data[column].apply(lambda x: re.sub(r'[^A-Za-z\s]', '', x) if type(x)==str else x)
print('noise removal done')

# ----------------------------------------------------------------------------

# turning the column accidents_data['DCA_DESC'] into tokens, and remove stop words
def remove_stopwords(text):
    tokens = nltk.word_tokenize(text)
    # put non stop words into list and join them back into a string
    text = ' '.join([word for word in tokens if word not in stop_words])
    return text

accidents_data['DCA_DESC'] = accidents_data['DCA_DESC'].apply(lambda x: remove_stopwords(x) if type(x)==str else x)
print('stop words removed')

# ----------------------------------------------------------------------------
print(accidents_data['DCA_DESC'][0:10])
# applied stemming to get root form of words
# ended up with some words that did not make sense within context - e.g. "accid", "anoth"
porterStemmer = PorterStemmer()

def stem_text(text):
    tokens = nltk.word_tokenize(text)
    stemmed = [porterStemmer.stem(word) for word in tokens]
    text = ' '.join(stemmed)
    return text


accidents_data['DCA_DESC'] = accidents_data['DCA_DESC'].apply(lambda x: stem_text(x) if type(x)==str else x)

print('stemming done')

# ----------------------------------------------------------------------------
# LEMMATIZATION (doesn't work)

#def lemmatize_text(text):
#    lemmatizer = WordNetLemmatizer()
#    tokens = nltk.word_tokenize(text)
#    lemmatized = [lemmatizer.lemmatize(word) for word in tokens]
#    text = ' '.join(lemmatized)
#    return text

#accidents_data['DCA_DESC'] = accidents_data['DCA_DESC'].apply(lambda x: lemmatize_text(x) if type(x)==str else x)

#print('lemmatizing done')

# ----------------------------------------------------------------------------

# bag of words, coverts text into BoW, where the matrix counts the number of times each word appears in the text
print(len(accidents_data['DCA_DESC']))

vectorizer = CountVectorizer()
bow = vectorizer.fit_transform(accidents_data['DCA_DESC'])
vocabulary = vectorizer.get_feature_names_out()
pd.DataFrame(bow.toarray())

print('bag of words done')

# ----------------------------------------------------------------------------
# getting most common words in the vectorizer
word_counts = np.asarray(bow.sum(axis=0)).flatten()

# vocab = rows, column = freq
word_counts = pd.DataFrame(word_counts, index=vocabulary, columns=['count'])
word_counts = word_counts.sort_values(by='count', ascending=False)
print(word_counts[:19])

# NEXT: word cloud, visualisation of the most common words in the text
# wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(accidents_data['DCA_DESC']))

word_cloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_counts['count'])
plt.figure(figsize=(10, 5))
plt.title('Word Cloud of Most Common Words in DCA_DESC')
plt.imshow(word_cloud, interpolation='bilinear')
plt.axis('off')
plt.show()
plt.savefig('wordcloud_dca_des.png', dpi=300, bbox_inches='tight')

def task2_1():
    
    return