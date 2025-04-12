import re
import pandas as pd
import numpy as np
import nltk

MORNING = "Morning"
AFTERNOON = "Afternoon"
EVENING = "Evening"
LATE_NIGHT = "Late Night"

# add new column TIME OF DAY

def task2_2():
    accidents_data = pd.read_csv('accident.csv')
    accidents_data['TIME_OF_DAY'] = None
    accidents_data = classifying_time(accidents_data)
    # print(accidents_data.columns)
    print(accidents_data['ACCIDENT_TIME'][0:10])
    print(accidents_data['TIME_OF_DAY'][0:10])
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

task2_2()

print('works')
