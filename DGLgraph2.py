#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 14:29:35 2022
@author: kaouther
"""

import dgl 
import torch

class Graph():
    def __init__(self, network, device=torch.device("cpu")): 
        """Initialize graph on specified device (default: CPU)"""
        self.device = device
        self.graph = self.generatGraph(network, network.getNetworkx())

    def getFeatures(self):
        return self.graph.ndata['features']
    
    def getGraph(self):
        return self.graph

    def generatGraph(self, network, networkx):
        graph = dgl.from_networkx(networkx).to(self.device)  
        features = torch.tensor(network.getFeatures2(), dtype=torch.float32, device=self.device)
        graph.ndata['features'] = features
        return graph
    
    def updateFeatures(self, features):
        self.graph.ndata['features'] = torch.tensor(features, dtype=torch.float32, device=self.device)

        
class SnGraph(Graph):
    def __init__(self, network, vnf_cpu, device=torch.device("cpu")):
        self.vnf_cpu = vnf_cpu
        super().__init__(network, device)

    def generatGraph(self, network, networkx):
        graph = dgl.from_networkx(networkx).to(self.device) 
        features = torch.tensor(network.getFeatures2(self.vnf_cpu), dtype=torch.float32, device=self.device)
        graph.ndata['features'] = features
        return graph
