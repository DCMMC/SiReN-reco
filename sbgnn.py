#!/usr/bin/env python3
#-*- coding: utf-8 -*-
"""
@author: huangjunjie
@file: sbgnn.py
@time: 2021/03/28
"""
import os
import sys
import time
import random
import argparse
import subprocess
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from tqdm import tqdm
import logging


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def construct_edges(edge_dic_list, node_num_a):
    edges = []
    for node in range(node_num_a):
        neighs = np.array(edge_dic_list[node]).reshape(-1, 1)
        a = np.array([node]).repeat(len(neighs)).reshape(-1, 1)
        edges.append(np.concatenate([a, neighs], axis=1))

    edges = np.vstack(edges)
    edges = torch.LongTensor(edges).to(device)
    return edges


class MeanAggregator(nn.Module):
    def __init__(self, a_dim, b_dim):
        super(MeanAggregator, self).__init__()

        self.out_mlp_layer = nn.Sequential(
            nn.Linear(b_dim, b_dim)
        )

    def forward(self, edges, feature_a, feature_b, node_num_a, node_num_b):
        matrix = torch.sparse_coo_tensor(edges.t(), torch.ones(edges.shape[0]), torch.Size([node_num_a, node_num_b]), device=device)
        row_sum = torch.spmm(matrix, torch.ones(size=(node_num_b, 1)).to(device))
        row_sum = torch.where(row_sum == 0, torch.ones(row_sum.shape).to(device), row_sum)

        new_emb = feature_b
        new_emb = self.out_mlp_layer(new_emb)
        output_emb = torch.spmm(matrix, new_emb)
        output_emb = output_emb.div(row_sum)

        return output_emb


class AttentionAggregator(nn.Module):
    def __init__(self, a_dim, b_dim):
        super(AttentionAggregator, self).__init__()

        self.out_mlp_layer = nn.Sequential(
            nn.Linear(b_dim, b_dim),
        )

        self.a = nn.Parameter(torch.FloatTensor(a_dim + b_dim, 1))
        nn.init.kaiming_normal_(self.a.data)

    def forward(self, edges, feature_a, feature_b, node_num_a, node_num_b):
        new_emb = feature_b
        new_emb = self.out_mlp_layer(new_emb)

        edge_h_2 = torch.cat([feature_a[edges[:, 0]], new_emb[edges[:, 1]] ], dim=1)
        edges_h = torch.exp(F.elu(torch.einsum("ij,jl->il", [edge_h_2, self.a]), 0.1))
        matrix = torch.sparse_coo_tensor(edges.t(), edges_h[:, 0],
                                         torch.Size([node_num_a, node_num_b]), device=device)
        row_sum = torch.sparse.mm(matrix, torch.ones(size=(node_num_b, 1)).to(device))
        row_sum = torch.where(row_sum == 0, torch.ones(row_sum.shape).to(device), row_sum)

        output_emb = torch.sparse.mm(matrix, new_emb)
        output_emb = output_emb.div(row_sum)
        return output_emb


class SBGNNLayer(nn.Module):
    def __init__(self, set_a_num, set_b_num, edgelist_a_b_pos, edgelist_a_b_neg, edgelist_b_a_pos,
                 edgelist_b_a_neg, edgelist_a_a_pos, edgelist_a_a_neg, edgelist_b_b_pos,
                 edgelist_b_b_neg, dropout=0.5, emb_size_a=32, emb_size_b=32,
                 aggregator=MeanAggregator):
        super(SBGNNLayer, self).__init__()
        self.set_a_num = set_a_num
        self.set_b_num = set_b_num
        self.edgelist_a_b_pos, self.edgelist_a_b_neg, self.edgelist_b_a_pos, self.edgelist_b_a_neg = \
                edgelist_a_b_pos, edgelist_a_b_neg, edgelist_b_a_pos, edgelist_b_a_neg
        self.edgelist_a_a_pos, self.edgelist_a_a_neg, self.edgelist_b_b_pos, self.edgelist_b_b_neg = \
                edgelist_a_a_pos, edgelist_a_a_neg, edgelist_b_b_pos, edgelist_b_b_neg

        self.agg_a_from_b_pos = aggregator(emb_size_b, emb_size_a)
        self.agg_a_from_b_neg = aggregator(emb_size_b, emb_size_a)
        self.agg_a_from_a_pos = aggregator(emb_size_a, emb_size_a)
        self.agg_a_from_a_neg = aggregator(emb_size_a, emb_size_a)

        self.agg_b_from_a_pos = aggregator(emb_size_a, emb_size_b)
        self.agg_b_from_a_neg = aggregator(emb_size_a, emb_size_b)
        self.agg_b_from_b_pos = aggregator(emb_size_b, emb_size_b)
        self.agg_b_from_b_neg = aggregator(emb_size_b, emb_size_b)

        self.update_func = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(emb_size_a * 5, emb_size_a * 2),
            nn.PReLU(),
            nn.Linear(emb_size_b * 2, emb_size_b)
        )


    def forward(self, feature_a, feature_b):
        assert feature_a.size()[0] == self.set_a_num, 'set_b_num error'
        assert feature_b.size()[0] == self.set_b_num, 'set_b_num error'

        node_num_a, node_num_b = self.set_a_num, self.set_b_num

        m_a_from_b_pos = self.agg_a_from_b_pos(self.edgelist_a_b_pos, feature_a, feature_b, node_num_a, node_num_b)
        m_a_from_b_neg = self.agg_a_from_b_neg(self.edgelist_a_b_neg, feature_a, feature_b, node_num_a, node_num_b)
        m_a_from_a_pos = self.agg_a_from_a_pos(self.edgelist_a_a_pos, feature_a, feature_a, node_num_a, node_num_a)
        m_a_from_a_neg = self.agg_a_from_a_neg(self.edgelist_a_a_neg, feature_a, feature_a, node_num_a, node_num_a)
        
        new_feature_a = torch.cat([feature_a, m_a_from_b_pos, m_a_from_b_neg, m_a_from_a_pos, m_a_from_a_neg], dim=1)
        new_feature_a = self.update_func(new_feature_a)

        m_b_from_a_pos = self.agg_b_from_a_pos(self.edgelist_b_a_pos, feature_b, feature_a, node_num_b, node_num_a)
        m_b_from_a_neg = self.agg_b_from_a_neg(self.edgelist_b_a_neg, feature_b, feature_a, node_num_b, node_num_a)
        m_b_from_b_pos = self.agg_b_from_b_pos(self.edgelist_b_b_pos, feature_b, feature_b, node_num_b, node_num_b)
        m_b_from_b_neg = self.agg_b_from_b_neg(self.edgelist_b_b_neg, feature_b, feature_b, node_num_b, node_num_b)

        new_feature_b = torch.cat([feature_b, m_b_from_a_pos, m_b_from_a_neg, m_b_from_b_pos, m_b_from_b_neg], dim=1)
        new_feature_b = self.update_func(new_feature_b)

        return new_feature_a, new_feature_b



class SBGNN(nn.Module):
    def __init__(self, edgelists, set_a_num, set_b_num, reg, dropout=0.5,
                    layer_num=1, embed_dim=32, aggregator=AttentionAggregator):
        super(SBGNN, self).__init__()
        self.reg = reg
        self.set_a_num = set_a_num
        self.set_b_num = set_b_num
        self.embed_dim = embed_dim
        emb_size_a, emb_size_b = embed_dim, embed_dim

        # assert edgelists must compelte
        assert len(edgelists) == 8, 'must 8 edgelists'
        (
            edgelist_a_b_pos, edgelist_a_b_neg, edgelist_b_a_pos, edgelist_b_a_neg,
            edgelist_a_a_pos, edgelist_a_a_neg, edgelist_b_b_pos, edgelist_b_b_neg
        ) = map(lambda x: construct_edges(x, set_a_num), edgelists)

        self.features_a = nn.Embedding(self.set_a_num, emb_size_a)
        self.features_b = nn.Embedding(self.set_b_num, emb_size_b)
        self.features_a.weight.requires_grad = True
        self.features_b.weight.requires_grad = True

        self.layers = nn.ModuleList(
            [SBGNNLayer(
                set_a_num, set_b_num,
                edgelist_a_b_pos, edgelist_a_b_neg, edgelist_b_a_pos, edgelist_b_a_neg,
                edgelist_a_a_pos, edgelist_a_a_neg, edgelist_b_b_pos, edgelist_b_b_neg,
                dropout=dropout, emb_size_a=32, emb_size_b=32, aggregator=aggregator
            ) for _ in range(layer_num)]
        )

    def aggregate(self):
        emb_a = self.features_a(torch.arange(self.set_a_num).to(device))
        emb_b = self.features_b(torch.arange(self.set_b_num).to(device))
        for m in self.layers:
            emb_a, emb_b = m(emb_a, emb_b)
        emb = torch.concat([emb_a, emb_b], dim=0)
        return emb

    def forward(self,u,v,w,n):
        margin = 0.4
        neg_weight = 1
        emb = self.aggregate()
        u = emb[u]
        v = emb[v]
        n = emb[n]
        positivebatch = torch.mul(u , v)
        negativebatch = torch.mul(u.view(len(u),1,self.embed_dim),n)
        # sBPR_loss =  F.logsigmoid((((-1/2*torch.sign(w)+3/2)).view(len(u),1) * (positivebatch.sum(dim=1).view(len(u),1))) - negativebatch.sum(dim=2)).sum(dim=1) # weight
        sBPR_loss =  F.logsigmoid(torch.sign(w).view(len(u),1) * (
            negativebatch.shape[1] * positivebatch.sum(dim=1).view(len(u),1) - negativebatch.sum(dim=2))).sum(dim=1) # weight
        loss = -torch.sum(sBPR_loss)
        # CCL
        # pos_loss = torch.relu(1 - positivebatch.sum(dim=1))
        # neg_loss = torch.relu(F.sigmoid(negativebatch.sum(dim=2)) - margin)
        # loss = (pos_loss + neg_loss.mean(dim=-1) * neg_weight).mean()
        reg_loss = u.norm(dim=1).pow(2).sum() + v.norm(dim=1).pow(2).sum() + n.norm(dim=2).pow(2).sum()
        return loss + self.reg * reg_loss


# ============= load data
def load_edgelists(edge_lists):
    edgelist_a_b_pos, edgelist_a_b_neg = defaultdict(list), defaultdict(list) 
    edgelist_b_a_pos, edgelist_b_a_neg = defaultdict(list), defaultdict(list)
    edgelist_a_a_pos, edgelist_a_a_neg = defaultdict(list), defaultdict(list)
    edgelist_b_b_pos, edgelist_b_b_neg = defaultdict(list), defaultdict(list)

    for a, b, s in edge_lists:
        if s == 1:
            edgelist_a_b_pos[a].append(b)
            edgelist_b_a_pos[b].append(a)
        elif s== -1:
            edgelist_a_b_neg[a].append(b)
            edgelist_b_a_neg[b].append(a)
        else:
            print(a, b, s)
            raise Exception("s must be -1/1")

    edge_list_a_a = defaultdict(lambda: defaultdict(int))
    edge_list_b_b = defaultdict(lambda: defaultdict(int))
    for a, b, s in edge_lists:
        for b2 in edgelist_a_b_pos[a]:
            edge_list_b_b[b][b2] += 1 * s
        for b2 in edgelist_a_b_neg[a]:
            edge_list_b_b[b][b2] -= 1 * s
        for a2 in edgelist_b_a_pos[b]:
            edge_list_a_a[a][a2] += 1 * s
        for a2 in edgelist_b_a_neg[b]:
            edge_list_a_a[a][a2] -= 1 * s

    for a1 in edge_list_a_a:
        for a2 in edge_list_a_a[a1]:
            v = edge_list_a_a[a1][a2]
            if a1 == a2: continue
            if v > 0:
                edgelist_a_a_pos[a1].append(a2)
            elif v < 0:
                edgelist_a_a_neg[a1].append(a2) 

    for b1 in edge_list_b_b:
        for b2 in edge_list_b_b[b1]:
            v = edge_list_b_b[b1][b2]
            if b1 == b2: continue
            if v > 0:
                edgelist_b_b_pos[b1].append(b2)
            elif v < 0:
                edgelist_b_b_neg[b1].append(b2) 

    return edgelist_a_b_pos, edgelist_a_b_neg, edgelist_b_a_pos, edgelist_b_a_neg,\
                    edgelist_a_a_pos, edgelist_a_a_neg, edgelist_b_b_pos, edgelist_b_b_neg

