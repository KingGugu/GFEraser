# -*- coding:utf-8 -*-

import os
import time
import math
import torch
import random
import numpy as np
import scipy.sparse as sp
from re import split


class Timer:
    def __init__(self):
        self.start_time = None
        self.elapsed_time = 0
        self.is_running = False

    def start(self):
        if not self.is_running:
            self.start_time = time.time() - self.elapsed_time
            self.is_running = True
            print("Timer started.")

    def pause(self):
        if self.is_running:
            self.elapsed_time = time.time() - self.start_time
            self.is_running = False
            print("Timer paused.")

    def reset(self):
        self.start_time = None
        self.elapsed_time = 0
        self.is_running = False
        print("Timer reset.")

    def get_elapsed_time(self):
        if self.is_running:
            current_time = time.time()
            self.elapsed_time = current_time - self.start_time
        return self.elapsed_time


def convert_sparse_mat_to_tensor(X):
    coo = X.tocoo()
    i = torch.LongTensor(np.array([coo.row, coo.col]))
    v = torch.from_numpy(coo.data).float()
    return torch.sparse.FloatTensor(i, v, coo.shape)


def normalize_graph_mat(adj_mat):
    shape = adj_mat.get_shape()
    rowsum = np.array(adj_mat.sum(1)) + 1e-7
    if shape[0] == shape[1]:
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj_tmp = d_mat_inv.dot(adj_mat)
        norm_adj_mat = norm_adj_tmp.dot(d_mat_inv)
    else:
        d_inv = np.power(rowsum, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj_mat = d_mat_inv.dot(adj_mat)
    return norm_adj_mat


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f'{path} created')


def get_graph_and_dataset(args):
    n_nodes = args.num_users + args.num_items

    # get user positive items
    user_positive_item = dict()
    for i in range(args.num_users):
        user_positive_item.setdefault(i, [])

    # get new training data
    interaction_data = []
    with open(args.train_data_file) as f:
        for line in f:
            items = split(' ', line.strip())
            temp = [int(i) for i in items[1:]]
            for i in range(len(temp)):
                user_id = int(items[0])
                item_id = int(temp[i])
                interaction_data.append([user_id, item_id])
                user_positive_item[user_id].append(item_id)

    row_idx = [pair[0] for pair in interaction_data]
    col_idx = [pair[1] for pair in interaction_data]
    user_np = np.array(row_idx)
    item_np = np.array(col_idx)
    ratings = np.ones_like(user_np, dtype=np.float32)
    tmp_adj = sp.csr_matrix((ratings, (user_np, item_np + args.num_users)), shape=(n_nodes, n_nodes), dtype=np.float32)
    new_adj_mat = tmp_adj + tmp_adj.T
    new_adj_mat = normalize_graph_mat(new_adj_mat)
    new_adj_mat = convert_sparse_mat_to_tensor(new_adj_mat).to('cuda')

    # get user special negative items
    neg_interaction_data = []
    user_special_neg_item_A = dict()
    for i in range(args.num_users):
        user_special_neg_item_A.setdefault(i, [])
    with open(args.special_neg_data_A) as f:
        for line in f:
            items = split(' ', line.strip())
            temp = [int(i) for i in items[1:]]
            for i in range(len(temp)):
                user_id = int(items[0])
                item_id = int(temp[i])
                neg_interaction_data.append([user_id, item_id])
                user_special_neg_item_A[user_id].append(item_id)

    row_idx = [pair[0] for pair in neg_interaction_data]
    col_idx = [pair[1] for pair in neg_interaction_data]
    user_np = np.array(row_idx)
    item_np = np.array(col_idx)
    ratings = np.ones_like(user_np, dtype=np.float32)
    tmp_adj = sp.csr_matrix((ratings, (user_np, item_np + args.num_users)), shape=(n_nodes, n_nodes), dtype=np.float32)
    new_adj_mat_neg = tmp_adj + tmp_adj.T
    new_adj_mat_neg = normalize_graph_mat(new_adj_mat_neg)
    new_adj_mat_neg = convert_sparse_mat_to_tensor(new_adj_mat_neg).to('cuda')

    user_special_neg_item_B = dict()
    for i in range(args.num_users):
        user_special_neg_item_B.setdefault(i, [])
    with open(args.special_neg_data_B) as f:
        for line in f:
            items = split(' ', line.strip())
            temp = [int(i) for i in items[1:]]
            for i in range(len(temp)):
                user_id = int(items[0])
                item_id = int(temp[i])
                user_special_neg_item_B[user_id].append(item_id)

    # get new test data
    test_data = dict()
    for i in range(args.num_users):
        test_data.setdefault(i, [])
    with open(args.test_data_file) as f:
        for line in f:
            items = split(' ', line.strip())
            temp = [int(i) for i in items[1:]]
            for i in range(len(temp)):
                user_id = int(items[0])
                item_id = int(temp[i])
                test_data[user_id].append(item_id)

    # build original graph to get original ranking
    original_interaction_data = []
    with open(args.original_data_file) as f:
        for line in f:
            items = split(' ', line.strip())
            temp = [int(i) for i in items[1:]]
            for i in range(len(temp)):
                user_id = int(items[0])
                item_id = int(temp[i])
                original_interaction_data.append([user_id, item_id])
                # user_positive_item[user_id].append(item_id)

    row_idx = [pair[0] for pair in original_interaction_data]
    col_idx = [pair[1] for pair in original_interaction_data]
    user_np = np.array(row_idx)
    item_np = np.array(col_idx)
    ratings = np.ones_like(user_np, dtype=np.float32)
    tmp_adj = sp.csr_matrix((ratings, (user_np, item_np + args.num_users)), shape=(n_nodes, n_nodes), dtype=np.float32)
    original_adj_mat = tmp_adj + tmp_adj.T
    original_adj_mat = normalize_graph_mat(original_adj_mat)
    original_adj_mat = convert_sparse_mat_to_tensor(original_adj_mat).to('cuda')

    ret = {
        # data for positive graph
        'interaction_data': interaction_data,
        'user_positive_item': user_positive_item,

        # data for negative graph, A is the PC interactions from original train data, B is the PC interactions from original test data
        # Both A and B used for unlearning evaluate, only A for negative graph training
        'neg_interaction_data': neg_interaction_data,
        'user_special_neg_item_A': user_special_neg_item_A,
        'user_special_neg_item_B': user_special_neg_item_B,

        # new positive graph and new negative graph
        'new_adj_mat': new_adj_mat,
        'new_adj_mat_neg': new_adj_mat_neg,

        # this original graph only used for get original ranking
        'original_adj_mat': original_adj_mat,

        # test data for evaluate the recommendation performance
        'test_data': test_data,
    }

    return ret


def recall_at_k(actual, predicted, topk):
    sum_recall = 0.0
    num_users = len(predicted)
    true_users = 0
    for i in range(num_users):
        act_set = set(actual[i])
        pred_set = set(predicted[i][:topk])
        if len(act_set) != 0:
            sum_recall += len(act_set & pred_set) / float(len(act_set))
            true_users += 1
    return sum_recall / true_users


def ndcg_k(actual, predicted, topk):
    res = 0
    for user_id in range(len(actual)):
        k = min(topk, len(actual[user_id]))
        idcg = idcg_k(k)
        dcg_k = sum([int(predicted[user_id][j] in
                         set(actual[user_id])) / math.log(j + 2, 2) for j in range(topk)])
        res += dcg_k / idcg
    return res / float(len(actual))


# Calculates the ideal discounted cumulative gain at k
def idcg_k(k):
    res = sum([1.0 / math.log(i + 2, 2) for i in range(k)])
    if not res:
        return 1.0
    else:
        return res
