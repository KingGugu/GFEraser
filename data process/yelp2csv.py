# -*- coding: utf-8 -*-

import time
import pandas as pd

"""
Convert Yelp JSON data file to CSV data file.
"""


def convert_unique_idx(df, column_name):
    column_dict = {x: i for i, x in enumerate(df[column_name].unique())}
    df[column_name] = df[column_name].apply(column_dict.get)
    df[column_name] = df[column_name].astype('int')
    assert df[column_name].min() == 0
    assert df[column_name].max() == len(column_dict) - 1
    return df, column_dict


def convert_time(date):
    timeArray = time.strptime(date, "%Y-%m-%d %H:%M:%S")
    # timeArray = time.strptime(date, "%Y-%m-%d")
    return time.mktime(timeArray)


def k_core_check(df, u_core, i_core):
    user_counts = df['user'].value_counts()
    to_remove = user_counts[user_counts < u_core].index
    if len(to_remove) > 0:
        return False
    item_counts = df['item'].value_counts()
    to_remove = item_counts[item_counts < i_core].index
    if len(to_remove) > 0:
        return False
    return True


def k_core(df, u_core, i_core):
    is_cored = k_core_check(df, u_core, i_core)
    while not is_cored:
        print(df.shape)
        user_interactions = df.groupby('user').size().reset_index(name='UserInteractions')
        item_interactions = df.groupby('item').size().reset_index(name='ItemInteractions')
        filtered_users = user_interactions[user_interactions['UserInteractions'] >= u_core]['user']
        filtered_items = item_interactions[item_interactions['ItemInteractions'] >= i_core]['item']
        df = df[df['user'].isin(filtered_users) & df['item'].isin(filtered_items)]
        is_cored = k_core_check(df, u_core, i_core)
    return df


def parse(path):
    g = open(path, 'r', encoding='utf-8')
    for l in g:
        yield eval(l)


def getDF(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')


def yelp(data_file):
    df = getDF(data_file)
    df.drop(labels=['review_id', 'useful', 'funny', 'cool'], axis=1, inplace=True)
    names = ['user', 'item', 'rating', 'time']

    final = pd.DataFrame(columns=names)
    final['user'] = df['user_id']
    final['item'] = df['business_id']
    final['rating'] = df['stars']
    final['time'] = df['date']

    final['time'] = final['time'].apply(lambda x: convert_time(x))

    final.to_csv('Yelp.csv', index=False, header=0)
    return final


data_file = 'Yelp-2021/yelp_academic_dataset_review.json'
file_name_out = './Yelp.csv'
user_core = 20
item_core = 20

datasets = yelp(data_file)
# print(datasets.shape)
# print(datasets)

# If only run k-core code, no need to run the previous line of code
# datasets = pd.read_csv('Yelp.csv', sep=',', engine='python', names=['user', 'item', 'rate', 'time'])

datasets = k_core(datasets, user_core, item_core)
datasets, user_mapping = convert_unique_idx(datasets, 'user')
datasets, item_mapping = convert_unique_idx(datasets, 'item')
datasets.to_csv(file_name_out, index=False, header=0)

print(datasets.shape)
