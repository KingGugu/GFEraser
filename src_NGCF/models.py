# -*- coding:utf-8 -*-


import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def InfoNCE(view1, view2, temperature, b_cos=True):
    if b_cos:
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
    pos_score = (view1 * view2).sum(dim=-1)
    pos_score = torch.exp(pos_score / temperature)
    ttl_score = torch.matmul(view1, view2.transpose(0, 1))
    ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
    cl_loss = -torch.log(pos_score / ttl_score + 1e-8)
    return torch.mean(cl_loss)


class AttentionFuse(nn.Module):
    def __init__(self, embedding_dim):
        super(AttentionFuse, self).__init__()
        self.query = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.Tanh(),
            nn.Linear(embedding_dim, 1, bias=False)
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input_feature1, input_feature2):
        embedding1 = input_feature1
        embedding2 = input_feature2
        att = torch.cat([self.query(embedding1), self.query(embedding2)], dim=-1)
        weight = self.softmax(att)
        h = weight[:, 0].unsqueeze(dim=1) * embedding1 + weight[:, 1].unsqueeze(dim=1) * embedding2

        return h


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different
        (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) *
        (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


class mlp_filter(nn.Module):
    def __init__(self, layers):
        super(mlp_filter, self).__init__()
        self.dense_1 = nn.Linear(layers[0], layers[1])
        nn.init.xavier_normal_(self.dense_1.weight.data)
        self.dense_2 = nn.Linear(layers[1], layers[2])
        nn.init.xavier_normal_(self.dense_2.weight.data)
        self.dense_3 = nn.Linear(layers[2], layers[3])
        nn.init.xavier_normal_(self.dense_3.weight.data)
        self.actfunction = gelu
        self.normal = Normalize()

    def forward(self, input):
        # layer 1
        output = self.dense_1(input)
        output = self.actfunction(output)
        # layer 2
        output = self.dense_2(output)
        output = self.actfunction(output)
        # layer 3
        output = self.dense_3(output)
        output = self.normal(output)
        return output


def jensen_shannon_divergence(embeddings1, embeddings2):
    probs1 = F.softmax(embeddings1, dim=1)
    probs2 = F.softmax(embeddings2, dim=1)
    avg_probs = 0.5 * (probs1 + probs2)
    js_div = 0.5 * (F.kl_div(probs1, avg_probs, reduction='batchmean') +
                    F.kl_div(probs2, avg_probs, reduction='batchmean'))
    return js_div


class PreFilter(nn.Module):
    def __init__(self, args):
        super(PreFilter, self).__init__()
        self.args = args
        self.embedding_size = args.embedding_size
        self.item_size = args.num_items
        self.user_size = args.num_users
        self.new_adj_mat = args.data_dict['new_adj_mat']
        self.new_adj_mat_neg = args.data_dict['new_adj_mat_neg']
        self.graph_n_layers = args.graph_n_layers

        self.user_embedding = nn.Embedding.from_pretrained(args.user_emb, freeze=False)
        self.item_embedding = nn.Embedding.from_pretrained(args.item_emb, freeze=False)
        # self.user_embedding = nn.Embedding(self.user_size, self.embedding_size)
        # self.item_embedding = nn.Embedding(self.item_size, self.embedding_size)
        self.neg_user_embedding = nn.Embedding(self.user_size, self.embedding_size)
        self.neg_item_embedding = nn.Embedding(self.item_size, self.embedding_size)
        nn.init.xavier_uniform_(self.neg_user_embedding.weight.data)
        nn.init.xavier_uniform_(self.neg_item_embedding.weight.data)
        # nn.init.xavier_uniform_(self.user_embedding.weight.data)
        # nn.init.xavier_uniform_(self.item_embedding.weight.data)

        self.mlps_u = mlp_filter([64, 128, 128, 64])
        self.mlps_i = mlp_filter([64, 128, 128, 64])
        self.item_att = AttentionFuse(self.embedding_size)
        self.user_att = AttentionFuse(self.embedding_size)

    def calculate_bpr_loss(self, u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings):
        sup_pos_ratings = torch.mul(u_g_embeddings, pos_i_g_embeddings).sum(dim=1)
        sup_neg_ratings = torch.mul(u_g_embeddings, neg_i_g_embeddings).sum(dim=1)
        sup_logits = -torch.log(1e-8 + torch.sigmoid(sup_pos_ratings - sup_neg_ratings))
        bpr_loss = torch.mean(sup_logits)

        return bpr_loss

    def l2_reg_loss(self, *args):
        emb_loss = 0
        for emb in args:
            emb_loss += torch.norm(emb, p=2)
        return emb_loss

    def forward(self, users, pos_items, neg_items, special_neg, users_neg, pos_items_neg, neg_items_neg):
        neg_bpr_loss = self.calculate_bpr_loss(self.neg_user_embedding(users_neg),
                                               self.neg_item_embedding(pos_items_neg),
                                               self.neg_item_embedding(neg_items_neg))
        neg_reg_loss = self.l2_reg_loss(self.neg_user_embedding(users_neg),
                                        self.neg_item_embedding(pos_items_neg),
                                        self.neg_item_embedding(neg_items_neg))
        neg_loss = neg_bpr_loss + self.args.reg_weight * neg_reg_loss

        # 原始表示的微调，mlp过滤，即unlearning preference
        user_emb = F.embedding(users, self.user_embedding.weight)
        pos_item_emb = F.embedding(pos_items, self.item_embedding.weight)
        neg_item_emb = F.embedding(neg_items, self.item_embedding.weight)
        filtered_user_emb = self.mlps_u(user_emb)
        filtered_item_emb = self.mlps_i(pos_item_emb)

        # 使用JS散度最大化过滤后的embedding和负图上学习到的embedding的差异，更新mlp的参数
        neg_graph_user_emb = self.neg_user_embedding(users_neg)
        neg_graph_pos_item_emb = self.neg_item_embedding(pos_items_neg)
        loss_user = 0
        if filtered_user_emb.shape[0] == neg_graph_user_emb.shape[0]:
            loss_user = jensen_shannon_divergence(filtered_user_emb, neg_graph_user_emb.detach())
        loss_item = 0
        if filtered_item_emb.shape[0] == neg_graph_pos_item_emb.shape[0]:
            loss_item = jensen_shannon_divergence(filtered_item_emb, neg_graph_pos_item_emb.detach())
        js_loss = -(loss_user + loss_item)  # 最大化损失

        # neg_graph_user_emb = self.neg_user_embedding(users)
        # neg_graph_pos_item_emb = self.neg_item_embedding(pos_items)
        # loss_item = jensen_shannon_divergence(filtered_item_emb, neg_graph_pos_item_emb.detach())
        # loss_user = jensen_shannon_divergence(filtered_user_emb, neg_graph_user_emb.detach())
        # js_loss = -(loss_user + loss_item)


        # 利用对比学习，避免过多的信息损失
        user_cl_loss = InfoNCE(user_emb, filtered_user_emb, self.args.temp)
        item_cl_loss = InfoNCE(pos_item_emb, filtered_item_emb, self.args.temp)
        cl_loss = (user_cl_loss + item_cl_loss)

        # 使用attention将过滤后的embedding与原始的embedding进行融合，计算推荐损失
        final_user_emb = self.user_att(user_emb, filtered_user_emb)
        final_pos_item_emb = self.item_att(pos_item_emb, filtered_item_emb)
        bpr_loss = self.calculate_bpr_loss(final_user_emb, final_pos_item_emb, neg_item_emb)
        reg_loss = self.l2_reg_loss(self.user_embedding(users),
                                    self.item_embedding(pos_items),
                                    self.item_embedding(neg_items))
        pos_loss = bpr_loss + self.args.reg_weight * reg_loss
        loss = neg_loss + cl_loss * self.args.cl_weight + pos_loss * self.args.pos_bpr_weight + js_loss

        return neg_loss, loss

    def predict(self, u):
        filtered_user_emb = self.mlps_u(self.user_embedding.weight)
        filtered_item_emb = self.mlps_i(self.item_embedding.weight)
        final_user_emb = self.user_att(self.user_embedding.weight, filtered_user_emb)
        final_item_emb = self.item_att(self.item_embedding.weight, filtered_item_emb)
        u = [int(i) for i in u]
        score = torch.matmul(final_user_emb[u], final_item_emb.transpose(0, 1))
        return score.detach().cpu().numpy()


class Original(nn.Module):
    def __init__(self, args):
        super(Original, self).__init__()
        self.args = args
        self.embedding_size = args.embedding_size
        self.item_size = args.num_items
        self.user_size = args.num_users
        self.new_adj_mat = args.data_dict['new_adj_mat']
        self.original_adj_mat = args.data_dict['original_adj_mat']
        self.graph_n_layers = args.graph_n_layers

        self.eye_matrix = self.get_eye_mat()
        self.hidden_size_list = [self.embedding_size, self.embedding_size, self.embedding_size, self.embedding_size]
        self.message_dropout = 0.1
        self.user_embedding = nn.Embedding(self.user_size, self.embedding_size)
        self.item_embedding = nn.Embedding(self.item_size, self.embedding_size)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        self.GNNlayers = torch.nn.ModuleList()
        for idx, (input_size, output_size) in enumerate(zip(self.hidden_size_list[:-1], self.hidden_size_list[1:])):
            self.GNNlayers.append(BiGNNLayer(input_size, output_size))

    def graph_forward(self, norm_adj, graph_user_embedding, graph_item_embedding):
        all_embeddings = torch.cat([graph_user_embedding, graph_item_embedding], dim=0)
        embeddings_list = [all_embeddings]
        for gnn in self.GNNlayers:
            all_embeddings = gnn(norm_adj, self.eye_matrix, all_embeddings)
            all_embeddings = nn.LeakyReLU(negative_slope=0.2)(all_embeddings)
            all_embeddings = nn.Dropout(self.message_dropout)(all_embeddings)
            all_embeddings = F.normalize(all_embeddings, p=2, dim=1)
            embeddings_list += [all_embeddings]  # storage output embedding of each layer
        all_embeddings = torch.stack(embeddings_list, dim=1)
        ngcf_all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        user_all_embeddings, item_all_embeddings = torch.split(ngcf_all_embeddings, [self.user_size, self.item_size])

        return user_all_embeddings, item_all_embeddings

    def get_emb(self):
        user_emb, item_emb = self.graph_forward(self.new_adj_mat, self.user_embedding.weight,
                                                self.item_embedding.weight)

        return user_emb, item_emb

    def predict(self, u):
        user_emb, item_emb = self.graph_forward(self.new_adj_mat, self.user_embedding.weight,
                                                self.item_embedding.weight)
        u = [int(i) for i in u]
        score = torch.matmul(user_emb[u], item_emb.transpose(0, 1))
        return score.detach().cpu().numpy()

    def predict_for_original_ranking(self, u):
        graph_user_emb, graph_item_emb = self.graph_forward(self.original_adj_mat,
                                                            self.user_embedding.weight,
                                                            self.item_embedding.weight)
        u = [int(i) for i in u]
        score = torch.matmul(graph_user_emb[u], graph_item_emb.transpose(0, 1))
        return score.detach().cpu().numpy()

    def get_eye_mat(self):
        r"""Construct the identity matrix with the size of  n_items+n_users.

        Returns:
            Sparse tensor of the identity matrix. Shape of (n_items+n_users, n_items+n_users)
        """
        num = self.item_size + self.user_size  # number of column of the square matrix
        i = torch.LongTensor([range(0, num), range(0, num)])
        val = torch.FloatTensor([1] * num)  # identity matrix
        return torch.sparse.FloatTensor(i, val)



class BiGNNLayer(nn.Module):
    r"""Propagate a layer of Bi-interaction GNN

    .. math::
        output = (L+I)EW_1 + LE \otimes EW_2
    """

    def __init__(self, in_dim, out_dim):
        super(BiGNNLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.linear = torch.nn.Linear(in_features=in_dim, out_features=out_dim)
        self.interActTransform = torch.nn.Linear(
            in_features=in_dim, out_features=out_dim
        )

    def forward(self, lap_matrix, eye_matrix, features):
        # for GCF ajdMat is a (N+M) by (N+M) mat
        # lap_matrix L = D^-1(A)D^-1 # 拉普拉斯矩阵
        x = torch.sparse.mm(lap_matrix, features)

        inter_part1 = self.linear(features + x)
        inter_feature = torch.mul(x, features)
        inter_part2 = self.interActTransform(inter_feature)

        return inter_part1 + inter_part2
