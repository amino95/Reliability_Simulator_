#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Graph Convolutional Network (GCN) module for VNE / VNF placement.
Includes multiple GCN layers with LayerNorm and DGL message passing.

Author: Kaouther (modified by Amine Rguez, 2025)
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn

gcn_msg = fn.copy_u('h', 'm')
gcn_reduce = fn.sum(msg='m', out='agg_h')


class GCNLayer(nn.Module):
    """A single layer of the Graph Convolutional Network (GCN)."""
    def __init__(self, in_feats, out_feats, device=None):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.ln = nn.LayerNorm(out_feats)   # LN sur la sortie de la Linear
        self.device = device if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.to(self.device)

    def forward(self, g, feature):
        g = g.to(self.device)
        feature = feature.to(self.device)

        with g.local_scope():
            # feature: [N, in_feats]
            g.ndata["h"] = feature
            g.update_all(gcn_msg, gcn_reduce)
            h = g.ndata["agg_h"]          # [N, in_feats]

            # print(
            #     "GCNLayer: h.shape before Linear =", h.shape,
            #     "| Linear in_features =", self.linear.in_features,
            #     "out_features =", self.linear.out_features,
            # )

            h = self.linear(h)            # [N, out_feats]
            h = self.ln(h)                # LN sur out_feats
            return h


class GCNModule(nn.Module):
    """A multi-layer Graph Convolutional Network module."""
    def __init__(self, in_feats, hidden_dim, out_feats, device=None):
        super(GCNModule, self).__init__()
        self.device = device if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.layer1 = GCNLayer(in_feats, hidden_dim, self.device)
        self.layer2 = GCNLayer(hidden_dim, hidden_dim, self.device)
        self.layer3 = GCNLayer(hidden_dim, out_feats, self.device)
        self.to(self.device)

    def forward(self, g, features):
        g = g.to(self.device)
        features = features.to(self.device)

        # If feature dim is 7 but model expects 8, pad one extra zero column
        if features.size(1) == 7 and self.layer1.linear.in_features == 8:
            pad = torch.zeros(features.size(0), 1, device=features.device)
            features = torch.cat([features, pad], dim=1)  # [N, 8]

        x = F.relu(self.layer1(g, features))
        x = F.relu(self.layer2(g, x))
        x = self.layer3(g, x)
        return x

