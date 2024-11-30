# -*- coding: utf-8 -*-

import pandas as pd

"""
Convert inter data file (download from RecBole) to CSV data file.
"""


def convert_unique_idx(df, column_name):
    column_dict = {x: i for i, x in enumerate(df[column_name].unique())}
    df[column_name] = df[column_name].apply(column_dict.get)
    df[column_name] = df[column_name].astype('int')
    assert df[column_name].min() == 0
    assert df[column_name].max() == len(column_dict) - 1
    return df, column_dict


def k_core_check(df, u_core, i_core):
    user_counts = df['user_id:token'].value_counts()
    to_remove = user_counts[user_counts < u_core].index
    if len(to_remove) > 0:
        return False
    item_counts = df['item_id:token'].value_counts()
    to_remove = item_counts[item_counts < i_core].index
    if len(to_remove) > 0:
        return False
    return True


def k_core(df, u_core, i_core):
    is_cored = k_core_check(df, u_core, i_core)
    while not is_cored:
        print(df.shape)
        user_interactions = df.groupby('user_id:token').size().reset_index(name='UserInteractions')
        item_interactions = df.groupby('item_id:token').size().reset_index(name='ItemInteractions')
        filtered_users = user_interactions[user_interactions['UserInteractions'] >= u_core]['user_id:token']
        filtered_items = item_interactions[item_interactions['ItemInteractions'] >= i_core]['item_id:token']
        df = df[df['user_id:token'].isin(filtered_users) & df['item_id:token'].isin(filtered_items)]
        is_cored = k_core_check(df, u_core, i_core)
    return df


data_file = 'Amazon_Books.inter'
file_name_out = './Amazon_Books.csv'
user_core = 20
item_core = 20

datasets = pd.read_csv(data_file, sep='\t', engine='python')
print(datasets.shape)
print(datasets)
datasets = k_core(datasets, user_core, item_core)
datasets, user_mapping = convert_unique_idx(datasets, 'user_id:token')
datasets, item_mapping = convert_unique_idx(datasets, 'item_id:token')
datasets.to_csv(file_name_out, index=False, header=0)
print(datasets.shape)
