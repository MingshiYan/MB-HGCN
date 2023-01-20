#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : model.py
# @Author:
# @Date  : 2021/11/1 16:16
# @Desc  :
import os.path
from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F

from data_set import DataSet
from gcn_conv import GCNConv
from utils import BPRLoss, EmbLoss


class GraphEncoder(nn.Module):
    def __init__(self, layers, hidden_dim, dropout):
        super(GraphEncoder, self).__init__()
        self.gnn_layers = nn.ModuleList(
            [GCNConv(hidden_dim, hidden_dim, add_self_loops=False, cached=False) for i in range(layers)])
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, edge_index):

        result = [x]
        for i in range(len(self.gnn_layers)):
            x = self.gnn_layers[i](x=x, edge_index=edge_index)
            # x = self.dropout(x)
            x = F.normalize(x, dim=-1)
            result.append(x / (i + 1))
        result = torch.stack(result, dim=0)
        result = torch.sum(result, dim=0)
        return result


class Mutual_Attention(nn.Module):
    # input : batch_size * seq_len * input_dim
    # q : batch_size * input_dim * dim_k
    # k : batch_size * input_dim * dim_k
    # v : batch_size * input_dim * dim_v
    def __init__(self, input_dim, dim_qk, dim_v):
        super(Mutual_Attention, self).__init__()
        # self.q = nn.Linear(input_dim, dim_qk, bias=False)
        # self.k = nn.Linear(input_dim, dim_qk, bias=False)
        # self.v = nn.Linear(dim_v, dim_v, bias=False)
        self._norm_fact = 1 / sqrt(dim_qk)

    def forward(self, q_token, k_token, v_token):
        # Q = self.q(q_token)  # Q: batch_size * seq_len * dim_k
        # K = self.k(k_token)  # K: batch_size * seq_len * dim_k
        # V = self.v(v_token)  # V: batch_size * seq_len * dim_v

        # Q * K.T() # batch_size * seq_len * seq_len
        # att = nn.Softmax(dim=-1)(torch.matmul(Q, K.transpose(-1, -2)) * self._norm_fact)


        # Q * K.T() * V # batch_size * seq_len * dim_v
        # att = torch.matmul(att, V)

        att = nn.Softmax(dim=-1)(torch.matmul(q_token, k_token.transpose(-1, -2)) * self._norm_fact)
        att = torch.matmul(att, v_token)

        return att


class MB_HGCN(nn.Module):
    def __init__(self, args, dataset: DataSet):
        super(MB_HGCN, self).__init__()

        self.device = args.device
        self.layers = args.layers
        self.node_dropout = args.node_dropout
        self.message_dropout = nn.Dropout(p=args.message_dropout)
        self.n_users = dataset.user_count
        self.n_items = dataset.item_count
        self.edge_index = dataset.edge_index
        self.all_edge_index = dataset.all_edge_index
        self.item_behaviour_degree = dataset.item_behaviour_degree.to(self.device)
        self.behaviors = args.behaviors
        self.embedding_size = args.embedding_size
        self.user_embedding = nn.Embedding(self.n_users + 1, self.embedding_size, padding_idx=0)
        self.item_embedding = nn.Embedding(self.n_items + 1, self.embedding_size, padding_idx=0)
        self.Graph_encoder = nn.ModuleDict({
            behavior: GraphEncoder(self.layers, self.embedding_size, self.node_dropout) for behavior in self.behaviors
        })
        self.global_graph_encoder = GraphEncoder(self.layers, self.embedding_size, self.node_dropout)
        self.W = nn.Parameter(torch.ones(len(self.behaviors)))

        self.dim_qk = args.dim_qk
        self.dim_v = args.dim_v
        self.attention = Mutual_Attention(self.embedding_size, self.dim_qk, self.dim_v)

        self.reg_weight = args.reg_weight
        self.layers = args.layers
        self.bpr_loss = BPRLoss()
        self.emb_loss = EmbLoss()

        self.model_path = args.model_path
        self.check_point = args.check_point
        self.if_load_model = args.if_load_model
        self.message_dropout = nn.Dropout(p=args.message_dropout)

        self.storage_user_embeddings = None
        self.storage_item_embeddings = None

        self.apply(self._init_weights)

        self._load_model()

    def _init_weights(self, module):

        if isinstance(module, nn.Embedding):
            nn.init.xavier_uniform_(module.weight.data)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight.data)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def _load_model(self):
        if self.if_load_model:
            parameters = torch.load(os.path.join(self.model_path, self.check_point))
            self.load_state_dict(parameters, strict=False)

    def gcn_propagate(self, total_embeddings):
        """
        gcn propagate in each behavior
        """
        all_user_embeddings, all_item_embeddings = [], []
        for behavior in self.behaviors:
            indices = self.edge_index[behavior].to(self.device)
            behavior_embeddings = self.Graph_encoder[behavior](total_embeddings, indices)
            # behavior_embeddings = F.normalize(behavior_embeddings, dim=-1)
            # all_embeddings.append(behavior_embeddings + total_embeddings)
            user_embedding, item_embedding = torch.split(behavior_embeddings, [self.n_users + 1, self.n_items + 1])
            all_user_embeddings.append(user_embedding)
            all_item_embeddings.append(item_embedding)

        # target_user_embeddings = torch.stack(all_user_embeddings, dim=1)
        # target_user_embeddings = torch.sum(target_user_embeddings, dim=1)
        # all_user_embeddings[-1] = target_user_embeddings

        all_user_embeddings = torch.stack(all_user_embeddings, dim=1)
        all_item_embeddings = torch.stack(all_item_embeddings, dim=1)
        return all_user_embeddings, all_item_embeddings

    def gcn(self, total_embeddings, indices):
        total_embeddings = self.global_graph_encoder(total_embeddings, indices.to(self.device))
        # behavior_embeddings = F.normalize(behavior_embeddings, dim=-1)
        # return total_embeddings + behavior_embeddings
        return total_embeddings

    def forward(self, batch_data):
        self.storage_user_embeddings = None
        self.storage_item_embeddings = None

        all_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        all_embeddings = self.gcn(all_embeddings, self.all_edge_index)

        user_embedding, item_embedding = torch.split(all_embeddings, [self.n_users + 1, self.n_items + 1])
        all_user_embeddings, all_item_embeddings = self.gcn_propagate(all_embeddings)

        all_user_embeddings = self.attention(all_user_embeddings, all_user_embeddings, all_user_embeddings)
        all_user_embeddings = all_user_embeddings + user_embedding.unsqueeze(1)

        weight = self.item_behaviour_degree * self.W
        weight = weight / (torch.sum(weight, dim=1).unsqueeze(-1) + 1e-8)
        all_item_embeddings = all_item_embeddings * weight.unsqueeze(-1)
        all_item_embeddings = torch.sum(all_item_embeddings, dim=1) + item_embedding

        total_loss = 0
        for i in range(len(self.behaviors)):
            data = batch_data[:, i]
            users = data[:, 0].long()
            items = data[:, 1:].long()
            user_feature = all_user_embeddings[:, i][users.view(-1, 1)]
            item_feature = all_item_embeddings[items]
            # user_feature, item_feature = self.message_dropout(user_feature), self.message_dropout(item_feature)
            scores = torch.sum(user_feature * item_feature, dim=2)
            total_loss += self.bpr_loss(scores[:, 0], scores[:, 1])
        total_loss = total_loss + self.reg_weight * self.emb_loss(self.user_embedding.weight, self.item_embedding.weight)

        return total_loss

    def full_predict(self, users):
        if self.storage_user_embeddings is None:
            all_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
            all_embeddings = self.gcn(all_embeddings, self.all_edge_index)

            user_embedding, item_embedding = torch.split(all_embeddings, [self.n_users + 1, self.n_items + 1])
            all_user_embeddings, all_item_embeddings = self.gcn_propagate(all_embeddings)

            target_embeddings = all_user_embeddings[:, -1].unsqueeze(1)
            target_embeddings = self.attention(target_embeddings, all_user_embeddings, all_user_embeddings)
            self.storage_user_embeddings = target_embeddings.squeeze() + user_embedding

            weight = self.item_behaviour_degree * self.W
            weight = weight / (torch.sum(weight, dim=1).unsqueeze(-1) + 1e-8)
            all_item_embeddings = all_item_embeddings * weight.unsqueeze(-1)
            self.storage_item_embeddings = torch.sum(all_item_embeddings, dim=1) + item_embedding

        user_emb = self.storage_user_embeddings[users.long()]
        scores = torch.matmul(user_emb, self.storage_item_embeddings.transpose(0, 1))

        return scores

