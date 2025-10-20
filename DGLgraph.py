#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 14:29:35 2022

@author: kaouther
"""

import dgl
import torch

class Graph():
    def __init__(self, network, device=None):
        """ A DGL graph that represents the network's topology and associated features """
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.graph = self.generatGraph(network, network.getNetworkx()).to(self.device)

    def getFeatures(self):
        """ Retrieves node features from the DGL graph """
        return self.graph.ndata['features']

    def getGraph(self):
        """ Returns the DGL graph """
        return self.graph

    def generatGraph(self, network, networkx):
        """
        Generates a DGL graph from a NetworkX graph and assigns node features.

        Args:
            network: The network object containing network information (e.g., VNR).
            networkx: A NetworkX graph representing the network topology.

        Returns:
            DGL graph with node features.
        """
        graph = dgl.from_networkx(networkx)
        graph = graph.to(self.device)
        features = network.getFeatures()
        graph.ndata['features'] = torch.tensor(features, dtype=torch.float32).to(self.device)
        return graph

    def updateFeatures(self, features):
        """
        Updates the node features of the DGL graph.
        """
        self.graph.ndata['features'] = torch.tensor(features, dtype=torch.float32).to(self.device)


class SnGraph(Graph):
    def __init__(self, network, vnf_cpu, device=None):
        """
        Initializes the SN graph with CPU-aware filtering for VNF placement.

        Args:
            network: The substrate network object.
            vnf_cpu: The CPU requirement for the VNF.
        """
        self.vnf_cpu = vnf_cpu
        super().__init__(network, device)

    def generatGraph(self, network, networkx):
        """
        Generates a DGL graph for the SN with a CPU filter flag on each node.

        Returns:
            DGL graph with CPU-aware features for each node.
        """
        graph = dgl.from_networkx(networkx)
        graph = graph.to(self.device)
        features = network.getFeatures(self.vnf_cpu)
        graph.ndata['features'] = torch.tensor(features, dtype=torch.float32).to(self.device)
        return graph
