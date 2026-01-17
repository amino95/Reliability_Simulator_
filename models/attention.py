#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 15:01:33 2022

@author: kaouther
"""
import torch 
import torch.nn as nn


class GraphContext(nn.Module):
    
    def __init__(self, in_feats, activation):
        super(GraphContext, self).__init__()
        self.linear = nn.Linear(in_feats, in_feats)
        self.layer_norm = nn.LayerNorm(in_feats)
        self.activation = activation
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, embedding):
        embedding = embedding.to(self.device)
        h = self.linear(embedding)
        h = self.layer_norm(h)
        h = self.activation(h)
        return h


class Sn_AttentionLayer(nn.Module):
    
    def __init__(self, in_feats, activation):
        super(Sn_AttentionLayer, self).__init__()
        self.context = GraphContext(in_feats, activation)
        self.device = self.context.device
        self.to(self.device)

    def forward(self, matrix):
        matrix = matrix.to(self.device)
        mean = torch.mean(matrix, dim=0)
        graph_context = self.context(mean)
        attention_weights = torch.sigmoid(torch.mm(matrix, graph_context.view(-1, 1)))
        encoding = torch.mm(torch.t(matrix), attention_weights)
        return encoding


class Vnr_AttentionLayer(nn.Module):
    
    def __init__(self, in_feats, activation):
        super(Vnr_AttentionLayer, self).__init__()
        self.context = GraphContext(in_feats, activation)
        self.device = self.context.device
        self.to(self.device)

    def forward(self, matrix, vnf):
        matrix = matrix.to(self.device)
        graph_context = self.context(matrix[vnf])
        attention_weights = torch.sigmoid(torch.mm(matrix, graph_context.view(-1, 1)))
        encoding = torch.mm(torch.t(matrix), attention_weights)
        return encoding
