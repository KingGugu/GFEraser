# -*- coding: utf-8 -*-


import random
import numpy as np
from torch.utils.data import Dataset


class UIGraphDataset(Dataset):
    def __init__(self, args):
        self.args = args
        self.interaction_data = args.data_dict['interaction_data']
        self.user_positive_item = args.data_dict['user_positive_item']
        self.user_special_neg_item = args.data_dict['user_special_neg_item_A']
        self.n_negatives = args.n_negatives

    def get_user_positives(self, user):
        return self.user_positive_item[user]

    def get_user_negatives(self, user, n):
        neg = []
        positives = set(self.get_user_positives(user))
        while len(neg) < n:
            candidate = np.random.randint(1, self.args.num_items)
            if candidate not in positives:
                neg.append(candidate)
        return neg

    def get_special_user_negatives(self, user, n):
        positives = set(self.get_user_positives(user))
        special_neg = self.user_special_neg_item[user]

        if len(special_neg) != 0:
            return special_neg
        else:
            while len(special_neg) < n:
                candidate = np.random.randint(1, self.args.num_items)
                if candidate not in positives:
                    special_neg.append(candidate)
            return special_neg

    def __getitem__(self, index):
        interaction = self.interaction_data[index]
        user = interaction[0]
        pos_item = interaction[1]
        neg = self.get_user_negatives(user, self.n_negatives)
        special_neg = self.get_special_user_negatives(user, self.n_negatives)

        return user, pos_item, neg[0], random.choice(special_neg)

    def __len__(self):
        return len(self.interaction_data)


class UIGraphDataset_neg(Dataset):
    def __init__(self, args):
        self.args = args

        # For negative graphs, the positive samples are the transformed low rated items
        # and the negative samples are preferentially sampled from the high rated items of that user
        self.user_positive_item = args.data_dict['user_special_neg_item_A']
        self.interaction_data = args.data_dict['neg_interaction_data']
        self.user_special_neg_item = args.data_dict['user_positive_item']
        self.n_negatives = args.n_negatives

    def get_user_positives(self, user):
        return self.user_positive_item[user]

    def get_user_negatives(self, user, n):
        neg = []
        while len(neg) < n:
            if len(self.user_special_neg_item[user]) != 0:
                candidate = random.choice(self.user_special_neg_item[user])
                neg.append(candidate)
            else:
                positives = set(self.get_user_positives(user))
                while len(neg) < n:
                    candidate = np.random.randint(1, self.args.num_items)
                    if candidate not in positives:
                        neg.append(candidate)
        return neg

    def __getitem__(self, index):
        interaction = self.interaction_data[index]
        user = interaction[0]
        pos_item = interaction[1]
        neg = self.get_user_negatives(user, self.n_negatives)

        return user, pos_item, neg[0]

    def __len__(self):
        return len(self.interaction_data)


class TestUserDataset(Dataset):
    def __init__(self, args, user_list):
        self.args = args
        self.test_users = user_list

    def __getitem__(self, index):
        return self.test_users[index]

    def __len__(self):
        return len(self.test_users)
