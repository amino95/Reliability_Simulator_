#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 14:42:55 2022

@author: kaouther
"""
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import torch
from sympy.codegen.ast import continue_

from node import Snode
from edege import Sedege

class SN :
    
    def __init__(self, num_nodes, cpu_range, bw_range,lt_range, rel_range, topology):

        self.num_nodes = num_nodes
        """  Total number of nodes in the substrate network"""
        self.snode = []
        """ List of Snodes, containing objects of type Snode, representing the actual nodes of the SN """
        self.sedege = []
        """ List of Edges, containing objects of type Sedege, representing the actual Edges of the SN """
        self.graph = topology
        """ The SN topology"""
        self.edges = list(self.graph.edges())
        """ List of edges in the SN, containing source and destination nodes, used for managing topology """
        self.numedges = len(self.edges)
        """ Total number of Edges in the SN"""

        # Snodes Creation 
        #-------------------------------------------------------------#
        for i in range(self.num_nodes):
            cpu = np.random.randint(cpu_range[0],cpu_range[1])
            rel = np.random.uniform(rel_range[0],rel_range[1])

            self.snode.append(Snode(i,cpu,rel))
        #-------------------------------------------------------------#

        # Sedges Creation
        #-------------------------------------------------------------#
        for i in range(self.numedges):
            bw = np.random.randint(bw_range[0],bw_range[1])
            lt = np.random.randint(lt_range[0], lt_range[1])
            rel= np.random.uniform(rel_range[0], rel_range[1])
            a_t_b = [self.edges[i][0], self.edges[i][1]]
            self.sedege.append(Sedege(i, bw,lt, rel,a_t_b))
            self.snode[a_t_b[0]].links.append(i)
            self.snode[a_t_b[1]].links.append(i)
        #-------------------------------------------------------------#

        # Calculating the total bandwidth of connected edges for each Snode.
        #----------------------------------------------------------------#
        for n in self.snode:
            for e in self.sedege:
                if n.index in e.nodeindex:
                    n.bw += e.bandwidth
            n.lastbw = n.bw
        #----------------------------------------------------------------# 
        # Calculating the total bandwidth of connected edges for each Snode.
        # ----------------------------------------------------------------#
        for n in self.snode:
            for e in self.sedege:
                if n.index in e.nodeindex:
                    n.rel *= e.rel
        # ----------------------------------------------------------------#

        # Assigning neighbors to each Snode and Snode degree
        #----------------------------------------------------------------#
        for el in self.snode:
            el.neighbors = [n for n in self.graph.neighbors(el.index)]
        for el in self.snode:
            el.degree = len(el.neighbors)
        #----------------------------------------------------------------#



    def updateCpu(self,nodemapping,req_cpu):
        """ 
        Updates the CPU of substrate nodes after a vertical scaling up operation.
        
        For each node in the `nodemapping` list, the corresponding substrate node's CPU 
        is updated with the required CPU specified in the `req_cpu` list.
        
        Args:
            nodemapping (list): A list of indices representing the substrate nodes to be updated.
            req_cpu (list): A list of CPU requirements to be applied to the corresponding substrate nodes.
        """
        for i in range(len(nodemapping)):
            self.snode[nodemapping[i]].updateCpu(req_cpu[i])


    def remCpu(self,nodemapping):
        """ 
        Calculates the remaining CPU for each substrate node in the nodemapping list.
        
        This method iterates over the `nodemapping` list and appends the remaining CPU 
        (stored in `lastcpu`) of each corresponding substrate node to a list.
        
        Args:
            nodemapping (list): A list of indices representing the substrate nodes.

        Returns:
            list: A list containing the remaining CPU for each node in the nodemapping list.
        """
        remCpu=[]
        for i in nodemapping:
            remCpu.append(self.snode[i].lastcpu)  
        return remCpu

    def getCpu(self):
        """ 
        Calculates the remaining CPU for each substrate node in the SN.
        
        This method iterates over the snode list and appends the remaining CPU 
        (stored in `lastcpu`) of each corresponding substrate node to a list.
        
        Returns:
            list: A list containing the remaining CPU for each node in the SN.
        """
        cpu = []
        for node in self.snode:
            cpu.append(node.lastcpu)
        return cpu

    def get_vnf_reliability(self, vnr_id, vnode_idx):
        # vnf = vnr_id.vnode[vnr_id]
        # print('vnr_id.vnode[vnode_idx]', vnr_id)
        return vnr_id.vnode[vnode_idx].rel

    def removenodemapping(self, vnr, VNRSS):

        sn = len(self.snode)

        for i in range(sn):

            had_mapping = False
            new_vnodeindexs = []

            for (mapped_vnr_id, vnode_idx) in self.snode[i].vnodeindexs:

                if mapped_vnr_id == vnr.id:
                    had_mapping = True
                    # free CPU
                    self.snode[i].lastcpu += vnr.vnode[vnode_idx].cpu
                else:
                    new_vnodeindexs.append((mapped_vnr_id, vnode_idx))

            self.snode[i].vnodeindexs = new_vnodeindexs

            # ⛔ If this VNR had no mapping on this node → skip reliability
            if not had_mapping:
                continue

            # Recompute reliability
            new_rel = 1.0
            for (other_vnr_id, other_vnode_idx) in self.snode[i].vnodeindexs:
                idx = VNRSS.reqs_ids.index(other_vnr_id)
                other_vnr = VNRSS.reqs[idx]
                other_vnf = other_vnr.vnode[other_vnode_idx]
                new_rel *= other_vnf.rel

            self.snode[i].rel = new_rel

    def removeedegemapping(self, vnr):
        """ 
        Removes the edge mapping of a VNR from the substrate network.
        
        This method updates the substrate edges by removing the virtual edges (vedge) that were
        previously mapped to them. It performs the following steps:
        
        1. Iterates over all substrate edges (`self.sedege`).
        2. For each substrate edge, checks if any virtual edge from the given VNR is mapped to it.
        3. If a match is found, the substrate edge's remaining bandwidth (`lastbandwidth`) is 
        incremented by the bandwidth of the corresponding virtual edge.
        4. The matching virtual edge is removed from the `vedegeindexs` list of the substrate edge.
        5. If a substrate edge no longer has any mapped virtual edges, it is marked as closed (`open = False`).
        6. Resets the bandwidth usage (`lastbw`) of each substrate node by summing the bandwidth 
        of its connected edges.
        7. Clears the shortest path connections (`spc`) for each virtual edge in the VNR.
        
        Args:
            vnr: The VNR whose edge mappings should be removed.
        """
        en = len(self.sedege)
        vn = len(vnr.vedege)
        for e in range(en):
            tempx = []
            for x in self.sedege[e].vedegeindexs:
                if vnr.id == x[0]:
                    self.sedege[e].lastbandwidth = self.sedege[e].lastbandwidth + vnr.vedege[x[1]].bandwidth
                    tempx.append(x)
            for x in tempx:
                self.sedege[e].vedegeindexs.remove(x)
            if not self.sedege[e].vedegeindexs:
                self.sedege[e].open = False
        for n in self.snode:
            n.lastbw = 0
            for e in self.sedege:
                if n.index in e.nodeindex:
                    n.lastbw += e.lastbandwidth
        for ve in range(vn):
            vnr.vedege[ve].spc = []
    
    
    def Sn2_networkxG(self, bandlimit=0):
        """Converts the substrate network to a NetworkX graph representation."""
        g = nx.Graph()
        for snod in self.snode:
            g.add_node(snod.index, index=snod.index, cpu=snod.cpu, lastcpu=snod.lastcpu)
        en = len(self.sedege)
        for i in range(en):
            if self.sedege[i].lastbandwidth > bandlimit:
                g.add_edge(self.sedege[i].nodeindex[0], self.sedege[i].nodeindex[1], index=self.sedege[i].index,lastbandwidth=self.sedege[i].lastbandwidth, bandwidth=self.sedege[i].bandwidth, reliability=self.sedege[i].reliability, capacity=1)
        return g
    
    def drawSN(self,edege_label=False,classflag=False):
        """Draws the substrate network using NetworkX with optional edge labels and color-coding."""
        plt.figure()
        g = self.Sn2_networkxG()
        pos = nx.fruchterman_reingold_layout(g)
        if classflag:
            color = {}
            colomap = []
            for i in range(len(self.snode)):
                if self.snode[i].classflag not in color.keys():
                    color[self.snode[i].classflag] = self.randRGB()
                colomap.append(color[self.snode[i].classflag])
            nx.draw(g, node_color=colomap, font_size=8, node_size=300, pos=pos, with_labels=True,
                    nodelist=g.nodes())
            if  edege_label:nx.draw_networkx_edge_labels(g, pos, edge_labels={edege: g[edege[0]][edege[1]]["lastbandwidth"] for edege in g.edges()})
        else:
            nx.draw(g, node_color=[[0.5, 0.8, 0.8]], font_size=8, node_size=300, pos=pos, with_labels=True,nodelist=g.nodes())
            if  edege_label:nx.draw_networkx_edge_labels(g, pos, edge_labels={edege: g[edege[0]][edege[1]]["lastbandwidth"] for edege in g.edges()})
        plt.show()
        
    def randRGB(self):
        return (np.random.randint(0, 255) / 255.0,
                np.random.randint(0, 255) / 255.0,
                np.random.randint(0, 255) / 255.0)

    def getNetworkx(self):
        """ 
        Returns the NetworkX graph representation of the SN.
        This method provides access to the internal graph structure.
        """
        return self.g

    def msg(self):
        """calls the msg method for each Snode and Sedge to print their information."""
        n = len(self.snode)
        for i in range(n):
            print('--------------------')
            self.snode[i].msg()
        n = len(self.sedege)
        for i in range(n):
            print('--------------------')
            self.sedege[i].msg()

    def getFeatures(self, vnf_cpu):
        """
        Extract substrate features, including reliability.
        """

        # --------------------
        # CPU (normalized)
        # --------------------
        cpu = np.array([el.lastcpu for el in self.snode])
        cpu = cpu / (cpu.max() if cpu.max() > 0 else 1)
        cpu = cpu.reshape(1, -1)

        # --------------------
        # Bandwidth (normalized)
        # --------------------
        bw = np.array([el.lastbw for el in self.snode])
        bw = bw / (bw.max() if bw.max() > 0 else 1)
        bw = bw.reshape(1, -1)

        # --------------------
        # Reliability (normalized)
        # --------------------
        rel = np.array([el.rel for el in self.snode])
        rel = rel / (rel.max() if rel.max() > 0 else 1)
        rel = rel.reshape(1, -1)

        # --------------------
        # Avg BW / degree
        # --------------------
        bw_av = np.array([el.bw / el.degree for el in self.snode])
        bw_av = bw_av / (bw_av.max() if bw_av.max() > 0 else 1)
        bw_av = bw_av.reshape(1, -1)

        # --------------------
        # Max / Min BW
        # --------------------
        bw_max = np.array([el.max_bw(self.sedege) for el in self.snode])
        bw_max = bw_max / (bw_max.max() if bw_max.max() > 0 else 1)
        bw_max = bw_max.reshape(1, -1)

        bw_min = np.array([el.min_bw(self.sedege) for el in self.snode])
        bw_min = bw_min / (bw_min.max() if bw_min.max() > 0 else 1)
        bw_min = bw_min.reshape(1, -1)

        # --------------------
        # Degree
        # --------------------
        degree = np.array([el.degree for el in self.snode])
        degree = degree / (degree.max() if degree.max() > 0 else 1)
        degree = degree.reshape(1, -1)

        # --------------------
        # Feasible flag (1 if node can host VNF)
        # --------------------
        feasible_flag = np.array([1 if el.lastcpu >= vnf_cpu else 0 for el in self.snode])
        feasible_flag = feasible_flag.reshape(1, -1)

        # --------------------
        # p_load
        # --------------------
        p_load = np.array([el.p_load for el in self.snode])
        p_load = p_load.reshape(1, -1)

        # --------------------
        # FINAL FEATURES
        # order is important !
        # --------------------
        features = np.concatenate((
            feasible_flag,
            cpu,
            bw,
            rel,  # ⭐ reliability here
            bw_av,
            bw_max,
            bw_min,
            degree,
            p_load
        ), axis=0).T

        return features

    def getFeatures2(self,vnf_cpu):
        """
        Extract features for the substrate network without  (p_load) information.
        
        Similar to `getFeatures`, this function extracts and scales various substrate node features,
        but excludes p_load (scalability metrics). It returns the feature matrix excluding p_load.
        
        Args:
            vnf_cpu (int): CPU requirement of the VNF being considered for mapping.
            
        Returns:
            features (np.ndarray): A matrix of substrate network features, where each row corresponds 
                                to a node and its associated features.
        """
        cpu = [el.lastcpu for el in self.snode]
        cmax = np.max(cpu)
        scaled_cpu = cpu  / cmax
        cpu = torch.from_numpy(np.squeeze(scaled_cpu))
        cpu = torch.unsqueeze(cpu, dim=0).numpy()
        
        bw = [el.lastbw  for el in self.snode]
        bmax = np.max(bw)
        scaled_bw = bw / bmax
        bw = torch.from_numpy(np.squeeze(scaled_bw))
        bw = torch.unsqueeze(bw, dim=0).numpy()

        rel = [el.rel for el in self.snode]
        max_rel = np.max(rel)
        scaled_rel = rel / max_rel
        rel = torch.from_numpy(np.squeeze(scaled_rel))
        rel = torch.unsqueeze(rel, dim=0).numpy()
        
        bw_av = [el.bw / el.degree for el in self.snode]
        maxb = np.max(bw_av)
        scaled_bw_av = bw_av/maxb
        bw_av = torch.from_numpy(np.squeeze(scaled_bw_av))
        bw_av = torch.unsqueeze(bw_av, dim=0).numpy()

        bw_max = [el.max_bw(self.sedege) for el in self.snode]
        scaled_max = bw_max/np.max(bw_max)
        bw_max = torch.from_numpy(np.squeeze(scaled_max))
        bw_max = torch.unsqueeze(bw_max, dim=0).numpy()

        bw_min = [el.min_bw(self.sedege) for el in self.snode]
        scaled_min = bw_min/np.max(bw_min)
        bw_min = torch.from_numpy(np.squeeze(scaled_min))
        bw_min = torch.unsqueeze(bw_min, dim=0).numpy()

        degree = [el.degree for el in self.snode]
        scaled_degree = degree/np.max(degree)
        degree = torch.from_numpy(np.squeeze(scaled_degree))
        degree = torch.unsqueeze(degree, dim=0).numpy()

        feasible_flag = [1 if el.lastcpu > vnf_cpu else 0 for el in self.snode]
        feasible_flag = torch.from_numpy(np.squeeze(feasible_flag))
        feasible_flag = torch.unsqueeze(feasible_flag, dim=0).numpy()

        features = np.transpose(np.concatenate((feasible_flag , cpu,bw, rel, bw_av, bw_max, bw_min, degree)))
        return features
    

    def get_used_ressources(self):
        """
        Calculate the used resources in the substrate network.
        
        This function iterates through all substrate nodes and edges to determine 
        the amount of resources used, counting how many nodes and links are in use.
        
        Returns:
            dict: A dictionary containing the used CPU, number of used nodes, 
                used bandwidth, and number of used links.
        """
        used_cpu=0
        cpu =0
        used_bw = 0
        bw = 0
        used_nodes = 0
        used_links = 0
        for node in self.snode:
            used_cpu +=node.cpu- node.lastcpu
            #cpu+= node.cpu
            if node.cpu > node.lastcpu :
                used_nodes+=1
        for edge in self.sedege:
            used_bw+=edge.bandwidth-edge.lastbandwidth
            #bw += edge.bandwidth
            if edge.bandwidth > edge.lastbandwidth:
                used_links+=1
        
        return {"used_cpu":used_cpu,"used_nodes":used_nodes,"used_bw":used_bw,"used_links":used_links}
            
    def sn_state(self):
        return {"snodes_state" : [el.__str__() for el in self.snode] ,
                "sedges_state" : [el.__str__() for el in self.sedege]}
        
 
        


    
