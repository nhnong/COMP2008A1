import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from task2_1 import task2_1
from collections import Counter, defaultdict

MORNING = "Morning"
AFTERNOON = "Afternoon"
EVENING = "Evening"
LATE_NIGHT = "Late Night"
MONDAY = "Monday"
FRIDAY = "Friday"
SUNDAY = "Sunday"

def task2_2():
    # calling function task2_1() to get modified accidents data: _, accidents_data = task2_1()
    # or use a separate, modified accidents data saved as csv file in task2_1
    accidents_data = pd.read_csv('task2_1_accidents.csv')

    # classify accident times into time of day categories in a new column accidents_data['TIME_OF_DAY']
    accidents_data['TIME_OF_DAY'] = None
    accidents_data = classifying_time(accidents_data)

    # creating the bar chart for the number of accidents per category
    accidents_time_of_day = accidents_per_category(accidents_data)

    # creating the pie chart for top 10 words per time category
    words_per_category = ten_freq_words(accidents_data)
    words_num_pie_chart = create_pie_chart(words_per_category)

    # creating a stacked bar chart comparing the number of accidents across different days of the week
    comparing_weekdays_dict = compare_across_days(accidents_data)
    weekday_stacked_bar_chart = create_stacked_bar_chart(comparing_weekdays_dict)

    return [accidents_time_of_day, words_num_pie_chart, weekday_stacked_bar_chart]

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
    accidents_time_of_day = accidents_data.groupby('TIME_OF_DAY').size().reset_index(name='COUNT')
    plt.bar(accidents_time_of_day['TIME_OF_DAY'], accidents_time_of_day['COUNT'])
    plt.title('Number of accidents in each TIME_OF_DAY category')
    plt.savefig('task2_2_timeofday.png', dpi=300, bbox_inches='tight')
    return accidents_time_of_day

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

    return fig


def compare_across_days(accidents_data):
    # Create a bar chart comparing the number of accidents across different days of the week
    # assumes accidents_data['DAY_OF_WEEK'] is in the format 'Monday', 'Friday', 'Sunday'
    weekdays = [MONDAY, FRIDAY, SUNDAY]
    time_of_day = [MORNING, AFTERNOON, EVENING, LATE_NIGHT]
    time_of_day_dict = dict.fromkeys(time_of_day, 0)
    weekdays_dict = {weekday: time_of_day_dict.copy() for weekday in weekdays}
    weekday_col = accidents_data['DAY_WEEK_DESC']
    for i in range(len(weekday_col)):
        if weekday_col[i] in weekdays:
            current_time = accidents_data['TIME_OF_DAY'][i]
            weekdays_dict[weekday_col[i]][current_time] += 1

    return weekdays_dict

def create_stacked_bar_chart(weekdays_dict):
    # Create a stacked bar chart comparing the number of accidents across different days of the week
    fig, ax = plt.subplots()
    bottom = np.zeros(3)
    num_weekdays = range((len(bottom)))
    legend_keys = {}
    for num, (weekday, time_tuple) in zip(num_weekdays, weekdays_dict.items()):
        labels = list(time_tuple.keys())
        counts = list(time_tuple.values())
        colours = ['yellow', 'orange', 'blue', 'purple']
        current_bottom = 0
        print(num)
        for label, count, box_colour in zip(labels, counts, colours):
            bar = ax.bar(weekday, count, bottom=current_bottom, label=label, color=box_colour)
            current_bottom += count
            bottom[num] += count
            if label not in legend_keys:
                legend_keys[label] = bar
        ax.set_title(f'{weekday}:')
    plt.title('Number of accidents during times of day on Monday, Friday, and Sunday')
    plt.ylabel('Number of accidents')
    plt.legend(legend_keys.values(), legend_keys.keys(), title='Time of Day', loc='upper left', bbox_to_anchor=(1, 1))
    #plt.legend(title='Time of Day', loc='upper left', bbox_to_anchor=(1, 1))
    plt.savefig('task2_2_stackbar.png', dpi=300, bbox_inches='tight')
    return fig

print('works')

