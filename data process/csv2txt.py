# -*- coding: utf-8 -*-

import tqdm
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

"""
Convert CSV data file to txt data file for model training and testing.
"""


def convert_unique_idx(df, column_name):
    column_dict = {x: i for i, x in enumerate(df[column_name].unique())}
    df[column_name] = df[column_name].apply(column_dict.get)
    df[column_name] = df[column_name].astype('int')
    assert df[column_name].min() == 0
    assert df[column_name].max() == len(column_dict) - 1
    return df, column_dict


def main(dataset, date_file_name, rate):
    df = pd.read_csv(date_file_name, sep=',', engine='python', names=['user', 'item', 'rate', 'time'])
    df, user_mapping = convert_unique_idx(df, 'user')
    df, item_mapping = convert_unique_idx(df, 'item')

    train_file = open(dataset + '_train.txt', mode='w')
    test_file = open(dataset + '_test.txt', mode='w')
    new_train_file = open(dataset + '_train_new.txt', mode='w')
    neg_train_file = open(dataset + '_train_neg.txt', mode='w')
    new_test_file = open(dataset + '_test_new.txt', mode='w')
    neg_test_file = open(dataset + '_test_neg.txt', mode='w')

    # create original train and test data
    df_copy = df.copy()
    train_data = df_copy.sample(frac=0.8, random_state=2023)
    temp_data = pd.concat([df_copy, train_data])
    test_data = temp_data.drop_duplicates(keep=False)
    for i in tqdm.tqdm(range(train_data.shape[0])):
        row = train_data.iloc[i]
        train_file.write(str(int(row['user'])) + " " + str(int(row['item'])) + '\n')
    for i in tqdm.tqdm(range(test_data.shape[0])):
        row = test_data.iloc[i]
        test_file.write(str(int(row['user'])) + " " + str(int(row['item'])) + '\n')
    train_file.close()
    test_file.close()

    # create new train and test data (delete rate <=3)
    train_neg = train_data.loc[(train_data['rate'] == 1) | (train_data['rate'] == 2) | (train_data['rate'] == 3)]
    selected_ratings = pd.DataFrame()
    for user_id, group in train_neg.groupby('user'):
        random_ratings = group.sample(n=int(rate * len(group)), random_state=2023)
        selected_ratings = selected_ratings.append(random_ratings)
    train_neg = selected_ratings

    # remove special_neg from the original data
    temp_data = pd.concat([train_data, train_neg])
    new_train_data = temp_data.drop_duplicates(keep=False)
    print('new train data num', new_train_data.shape)
    print('train negative num', train_neg.shape)

    for i in tqdm.tqdm(range(new_train_data.shape[0])):
        row = new_train_data.iloc[i]
        new_train_file.write(str(int(row['user'])) + " " + str(int(row['item'])) + '\n')
    for i in tqdm.tqdm(range(train_neg.shape[0])):
        row = train_neg.iloc[i]
        neg_train_file.write(str(int(row['user'])) + " " + str(int(row['item'])) + '\n')
    new_train_file.close()
    neg_train_file.close()

    test_neg = test_data[(test_data['rate'] == 1) | (test_data['rate'] == 2) | (test_data['rate'] == 3)]
    selected_ratings = pd.DataFrame()
    for user_id, group in test_neg.groupby('user'):
        random_ratings = group.sample(n=int(rate * len(group)), random_state=2023)
        selected_ratings = selected_ratings.append(random_ratings)
    test_neg = selected_ratings

    temp_data = pd.concat([test_data, test_neg])
    new_test_data = temp_data.drop_duplicates(keep=False)
    print('new test data num', new_test_data.shape)
    print('test negative num', test_neg.shape)

    for i in tqdm.tqdm(range(new_test_data.shape[0])):
        row = new_test_data.iloc[i]
        new_test_file.write(str(int(row['user'])) + " " + str(int(row['item'])) + '\n')
    for i in tqdm.tqdm(range(test_neg.shape[0])):
        row = test_neg.iloc[i]
        neg_test_file.write(str(int(row['user'])) + " " + str(int(row['item'])) + '\n')
    new_test_file.close()
    neg_test_file.close()


dataset = 'Yelp'
date_file_name = 'Yelp.csv'
# The ratio of PC interactions
rate = 1.0
main(dataset, date_file_name, rate)
