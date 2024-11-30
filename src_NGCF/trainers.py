# -*- coding:utf-8 -*-


import os
import torch
import numpy as np
from tqdm import tqdm
from itertools import cycle
from utils import recall_at_k, ndcg_k


class MyTrainer:
    def __init__(self, model, original_model, ui_dataloader, ui_dataloader_neg, test_dataloader, optimizer, args):
        self.args = args
        self.model = model
        self.original_model = original_model
        self.ui_dataloader = ui_dataloader
        self.test_dataloader = test_dataloader
        self.ui_dataloader_neg = ui_dataloader_neg
        self.device = args.device
        self.user_positive_item = args.data_dict['user_positive_item']
        self.test_data = args.data_dict['test_data']
        self.optimizer = optimizer

        self.init_neg_rank = []

    def get_full_sort_score(self, epoch, answers, pred_list, mode='test'):
        recall, ndcg = [], []
        for k in [10, 20, 50]:
            recall.append(recall_at_k(answers, pred_list, k))
            ndcg.append(ndcg_k(answers, pred_list, k))
        post_fix = {
            "stage": mode,
            "epoch": epoch,
            "Recall@10": '{:.4f}'.format(recall[0]), "NDCG@10": '{:.4f}'.format(ndcg[0]),
            "Recall@20": '{:.4f}'.format(recall[1]), "NDCG@20": '{:.4f}'.format(ndcg[1]),
            "Recall@50": '{:.4f}'.format(recall[2]), "NDCG@50": '{:.4f}'.format(ndcg[2])
        }
        with open(self.args.log_file, 'a') as f:
            f.write(str(post_fix) + '\n')
        with open(self.args.test_log_file, 'a') as f:
            f.write(str(post_fix) + '\n')
        return [recall[0], ndcg[0], recall[1], ndcg[1], recall[2], ndcg[2]], str(post_fix)

    def train_iteration(self, epoch):
        self.model.train()
        if epoch == 0:
            print('epoch:', epoch, f"  rec dataset length: {len(self.ui_dataloader)}", '\n')
        rec_cf_data_iter = tqdm(enumerate(zip(self.ui_dataloader, cycle(self.ui_dataloader_neg))),
                                total=len(self.ui_dataloader))

        for i, rec_batch in rec_cf_data_iter:
            positive_data, neg_data = rec_batch
            users, pos_items, neg_items, special_neg = positive_data
            users = users.to(self.device)
            pos_items = pos_items.to(self.device).to(torch.int64)
            neg_items = neg_items.to(self.device).to(torch.int64)
            special_neg = special_neg.to(self.device).to(torch.int64)

            users_neg, pos_items_neg, neg_items_neg = neg_data
            users_neg = users_neg.to(self.device)
            pos_items_neg = pos_items_neg.to(self.device).to(torch.int64)
            neg_items_neg = neg_items_neg.to(self.device).to(torch.int64)

            neg_loss, loss = self.model.forward(users, pos_items, neg_items, special_neg, users_neg, pos_items_neg,
                                                neg_items_neg)
            self.optimizer.zero_grad()
            # Two-stage
            # if epoch > 20:
            #     loss.backward()
            # else:
            #     neg_loss.backward()
            loss.backward()
            self.optimizer.step()
            with open(self.args.log_file, 'a') as f:
                f.write('epoch:' + str(epoch) + '  loss:' + str(loss.item()) + '\n')

    def test_iteration(self, epoch):
        if epoch == -1:
            self.original_model.eval()
        else:
            self.model.eval()
        pred_list = None
        answer_list = None
        first_batch = True
        for batch_users in tqdm(self.test_dataloader):
            test_items = [self.test_data[int(u)] for u in batch_users]
            if epoch == -1:
                ranking_score = self.original_model.predict(batch_users)
            else:
                ranking_score = self.model.predict(batch_users)  # (B,N)
            # set the ranking scores of training items to -inf,
            # then the training items will be sorted at the end of the ranking list.
            for index, user in enumerate(batch_users):
                train_items = self.user_positive_item[int(user)]
                for item in train_items:
                    ranking_score[index][item] = -np.inf

            ind = np.argpartition(ranking_score, -50)[:, -50:]
            arr_ind = ranking_score[np.arange(len(ranking_score))[:, None], ind]
            arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(ranking_score)), ::-1]
            batch_pred_list = ind[np.arange(len(ranking_score))[:, None], arr_ind_argsort]

            if first_batch:
                pred_list = batch_pred_list
                answer_list = test_items
                first_batch = False
            else:
                pred_list = np.append(pred_list, batch_pred_list, axis=0)
                answer_list = np.append(answer_list, test_items, axis=0)
        result_list, result_str = self.get_full_sort_score(epoch, answer_list, pred_list)
        if epoch == self.args.epochs:
            save_path = os.path.join(self.args.checkpoint_path, 'epoch-' + str(epoch) + '.pt')
            torch.save(self.model.state_dict(), save_path)

        return result_list

    def test_iteration_get_neg_rank(self, epoch, test_user_dataloader):
        if epoch == 1:
            self.original_model.eval()
            for batch_users in tqdm(test_user_dataloader):
                batch_final = []
                ranking_score = self.original_model.predict_for_original_ranking(batch_users)  # (B,N)
                sorted_ranking = np.argsort(ranking_score, axis=1)[:, ::-1]
                sorted_ranking = sorted_ranking.tolist()
                for i in range(len(batch_users)):
                    user = batch_users[i]
                    result = sorted_ranking[i]
                    for item in self.args.data_dict['user_special_neg_item_A'][int(user)]:
                        batch_final.append(result.index(item))
                    for item in self.args.data_dict['user_special_neg_item_B'][int(user)]:
                        batch_final.append(result.index(item))
                self.init_neg_rank.extend(batch_final)
        else:
            final = []
            self.model.eval()
            for batch_users in tqdm(test_user_dataloader):
                batch_final = []
                ranking_score = self.model.predict(batch_users)  # (B,N)
                sorted_ranking = np.argsort(ranking_score, axis=1)[:, ::-1]
                sorted_ranking = sorted_ranking.tolist()
                for i in range(len(batch_users)):
                    user = batch_users[i]
                    result = sorted_ranking[i]
                    for item in self.args.data_dict['user_special_neg_item_A'][int(user)]:
                        batch_final.append(result.index(item))
                    for item in self.args.data_dict['user_special_neg_item_B'][int(user)]:
                        batch_final.append(result.index(item))
                final.extend(batch_final)
            init_rank = np.array(self.init_neg_rank)
            new_rank = np.array(final)
            diff = (new_rank - init_rank) / (init_rank + 1)
            ranking_decrease_rate = diff.sum() / len(diff)

            post_fix = {"epoch": epoch, "ranking_decrease_rate": '{:.4f}'.format(ranking_decrease_rate)}
            with open(self.args.test_log_file, 'a') as f:
                f.write(str(post_fix) + '\n')
