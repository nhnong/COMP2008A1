import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from task2_1 import task2_1
from collections import Counter


MORNING = "Morning"
AFTERNOON = "Afternoon"
EVENING = "Evening"
LATE_NIGHT = "Late Night"

# add new column TIME OF DAY

def task2_2():
    # importing the modified data from tsk2_1
    # _, accidents_data = task2_1()
    accidents_data = pd.read_csv('task2_1_accidents.csv')

    # classify accident times into time of day categories in a new column accidents_data['TIME_OF_DAY']
    accidents_data['TIME_OF_DAY'] = None
    accidents_data = classifying_time(accidents_data)

    # creating the bar chart for the number of accidents per category
    accidents_per_cat_df = accidents_per_category(accidents_data)
    # print(accidents_per_cat_df)

    words_per_category = ten_freq_words(accidents_data)
    #print(words_per_category[MORNING])
    #print(words_per_category[AFTERNOON])
    #print(words_per_category[EVENING])
    #print(words_per_category[LATE_NIGHT])


    # creating the pie chart for top 10 words per time category
    pie_chart = create_pie_chart(words_per_category)

    return

# assumes that the time of day is in the format HH:MM:SS and matches r'(^\d{2}:\d{2}:\d{2}$)'
def classifying_time(accidents_data):
    pattern = r'(^\d{2}:\d{2}:\d{2}$)'
    for i in range(len(accidents_data['ACCIDENT_TIME'])):
        if re.match(pattern, accidents_data['ACCIDENT_TIME'][i]):
            # match time to day category
            hour = int(accidents_data['ACCIDENT_TIME'][i][:2])
            if hour >= 6 and hour < 12:
                accidents_data.loc[i, 'TIME_OF_DAY'] = MORNING
            elif hour >= 12 and hour < 18:
                accidents_data.loc[i, 'TIME_OF_DAY'] = AFTERNOON
            elif hour >= 18 and hour < 24:
                accidents_data.loc[i, 'TIME_OF_DAY'] = EVENING
            else:
                accidents_data.loc[i, 'TIME_OF_DAY'] = LATE_NIGHT         
    return accidents_data

# calculating the number of accidents per category and creating a new dataframe with the results
def accidents_per_category(accidents_data):
    accidents_per_cat_df = accidents_data.groupby('TIME_OF_DAY').size().reset_index(name='COUNT')
    plt.bar(accidents_per_cat_df['TIME_OF_DAY'], accidents_per_cat_df['COUNT'])
    plt.title('Number of accidents in each TIME_OF_DAY category')
    plt.savefig('task2_2_timeofday.png', dpi=300, bbox_inches='tight')
    return accidents_per_cat_df

def ten_freq_words(accidents_data):
    most_freq_morning = Counter()
    most_freq_afternoon = Counter()
    most_freq_evening = Counter()
    most_freq_late_night = Counter()

    for i in range(len(accidents_data['TIME_OF_DAY'])):
        current_time = accidents_data['TIME_OF_DAY'][i]
        dca_desc = nltk.word_tokenize(accidents_data['DCA_DESC'][i])  
        if current_time == MORNING:
            most_freq_morning.update(dca_desc)
        elif current_time == AFTERNOON:
            most_freq_afternoon.update(dca_desc)
        elif current_time == EVENING:
            most_freq_evening.update(dca_desc)
        elif current_time == LATE_NIGHT:
            most_freq_late_night.update(dca_desc) 
        
    most_freq_morning = most_freq_morning.most_common(10)
    most_freq_afternoon = most_freq_afternoon.most_common(10)
    most_freq_evening = most_freq_evening.most_common(10)
    most_freq_late_night = most_freq_late_night.most_common(10)   

    return {MORNING: most_freq_morning, AFTERNOON: most_freq_afternoon, EVENING: most_freq_evening, LATE_NIGHT: most_freq_late_night}    
    

def create_pie_chart(words_per_category):
    # Create a pie chart for the top 10 words in each category
    # for item in words_per_category.values():
    #    label = []
    #    count = []
    #    for x, y in item:
    #        label.append(x)
    #        count.append(y)
    #    plt.pie(count, labels=label, autopct='%1.1f%%', startangle=140)
    #    plt.axis('equal')
    #    plt.show()
    fig, axes = plt.subplots(2,2, figsize=(13, 10))
    axes = axes.flatten()
    for ax, (time, word_tuple) in zip(axes, words_per_category.items()):
        labels = [word[0] for word in word_tuple]
        counts = [word[1] for word in word_tuple]
        ax.pie(counts, labels=labels, autopct='%1.1f%%', radius=1.0, textprops={'fontsize': 7})
        ax.set_title(f'{time}: Top 10 Most Frequent Words in DCA_DESC Column')
    plt.savefig('task2_2_wordpies.png', dpi=300, bbox_inches='tight')
    plt.show()

    return words_per_category

task2_2()
print('works')
