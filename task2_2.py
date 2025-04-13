import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from task2_1 import task2_1

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
    print(accidents_per_cat_df)

    ten_freq_words(accidents_data)

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
    return
    
task2_2()

print('works')
