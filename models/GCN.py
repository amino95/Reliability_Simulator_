#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 15:01:33 2022

@author: kaouther
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn

# Message and reduce functions for GCN
gcn_msg = fn.copy_u('h', 'm')
gcn_reduce = fn.sum(msg='m', out='h')


class GCNLayer(nn.Module):
    """A single layer of the Graph Convolutional Network (GCN)."""
    def __init__(self, in_feats, out_feats, device=None):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.ln = nn.LayerNorm(in_feats)
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, g, feature):
        """Forward pass for GCN layer.

        Args:
            g: The graph input.
            feature: The input features for the nodes.

        Returns:
            Transformed node features after applying GCN operations.
        """
        g = g.to(self.device)
        feature = feature.to(self.device)

        with g.local_scope():
            g.ndata["h"] = feature
            g.update_all(gcn_msg, gcn_reduce)
            h = g.ndata["h"]
            h = self.ln(h)
            h = self.linear(h)
            return h


class GCNModule(nn.Module):
    """A multi-layer Graph Convolutional Network module."""
    def __init__(self, in_feats, hidden_dim, out_feats, device=None):
        super(GCNModule, self).__init__()
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.layer1 = GCNLayer(in_feats, hidden_dim, self.device)
        self.layer2 = GCNLayer(hidden_dim, hidden_dim, self.device)
        self.layer3 = GCNLayer(hidden_dim, out_feats, self.device)
        self.to(self.device)

    def forward(self, g, features):
        """Forward pass for the GCN module.

        Args:
            g: The graph input.
            features: The input features for the nodes.

        Returns:
            The final output features after passing through all layers.
        """
        g = g.to(self.device)
        features = features.to(self.device)

        x = F.relu(self.layer1(g, features))
        x = F.relu(self.layer2(g, x))
        x = self.layer3(g, x)
        return x
