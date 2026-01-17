# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 20:33:50 2022

@author: amirgu1
"""
from solver import Solver
import networkx as nx
import numpy as np
from copy import deepcopy as dc
import random

class Grasp(Solver):
    def __init__(self, sigma, rejection_penalty,
                 max_iter=20,           # nombre d'itérations GRASP
                 alpha=0.3,             # paramètre de RCL (0..1)
                 reliability_weight=0.15):

        super().__init__(sigma, rejection_penalty, reliability_weight)

        self.max_iter = max_iter
        self.alpha = alpha             # fraction of candidates in RCL

    def nodemapping(self, sb, vnr):
        """
        Maps the virtual nodes (VNFs) of a VNR to substrate nodes in the substrate network (sb).
        Uses randomized greedy with RCL for GRASP.

        Returns:
            - success: Boolean indicating if the mapping was successful or not.
            - v2sindex: List mapping each virtual node to a corresponding substrate node.
        """
        success = True 
        vn = vnr.num_vnfs 
        v2sindex = [] 
        sn = random.sample(list(range(len(sb.snode))), len(sb.snode))
        
        for vi in range(vn):
            cpu_req = vnr.vnode[vi].cpu

            # Find all feasible substrate nodes
            candidates = []
            for si in sn:
                if vnr.vnode[vi].cpu < sb.snode[si].lastcpu and si not in v2sindex:
                    candidates.append(si)

            if len(candidates) == 0:
                return False, []

            # Greedy selection based on available CPU (RCL)
            candidates_sorted = sorted(
                candidates,
                key=lambda si: sb.snode[si].lastcpu,
                reverse=True
            )

            # GRASP RCL - Restricted Candidate List
            RCL_size = max(1, int(self.alpha * len(candidates_sorted)))
            RCL = candidates_sorted[:RCL_size]

            chosen_snode = random.choice(RCL)
            v2sindex.append(chosen_snode)

        return success, v2sindex

    def construct_solution(self, sn, vnr):
        """
        GRASP construction phase:
            1) Build a feasible node mapping using a randomized greedy strategy
            2) Build a feasible edge mapping based on constructed nodes
            3) Return node and edge mappings without modifying sn

        Returns:
            - success (bool)
            - v2sindex (list of node mappings)
            - vese2index (list of edge mappings)
        """
        success = True 
        vn = vnr.num_vnfs 
        vese2index = []
        
        # Node mapping using GRASP
        nodesuccess, v2sindex = self.nodemapping(sn, vnr)
        if not nodesuccess:
            return False, [], []
            
        # Set sn_host for edge mapping
        for i, snode_index in enumerate(v2sindex):
            vnr.vnode[i].sn_host = snode_index
        
        # Edge mapping on deep copies
        asnode = dc(sn.snode)
        asedge = dc(sn.sedege)
        edgesuccess, vese2index = self.edegemapping(asnode, asedge, vnr, v2sindex)
        
        if not edgesuccess:
            # Reset sn_host if edge mapping fails
            for i in range(vn):
                vnr.vnode[i].sn_host = -1
            return False, [], []

        return True, v2sindex, vese2index

    def local_search(self, sn, vnr, v2sindex, vese2index):
        """
        Try to improve the current mapping by moving one VNF to a less-loaded node.
        Returns improved mappings.
        """
        best_nodemapping = list(v2sindex)
        best_edgemapping = list(vese2index)
        
        # Try swapping nodes
        for vnf_id in range(len(v2sindex)):
            for candidate_idx in range(len(sn.snode)):
                if candidate_idx in best_nodemapping:
                    continue
                    
                # Create new mapping
                new_nodemapping = best_nodemapping[:]
                new_nodemapping[vnf_id] = candidate_idx
                
                # Check feasibility
                if self.check_nodemapping_feasible(sn, vnr, new_nodemapping):
                    # Set sn_host for edge mapping
                    for i, snode_index in enumerate(new_nodemapping):
                        vnr.vnode[i].sn_host = snode_index
                    
                    # Try edge mapping
                    asnode = dc(sn.snode)
                    asedge = dc(sn.sedege)
                    edge_success, new_edgemapping = self.edegemapping(asnode, asedge, vnr, new_nodemapping)
                    
                    if edge_success:
                        # Return first improvement found
                        return new_nodemapping, new_edgemapping
        
        return best_nodemapping, best_edgemapping
    
    def check_nodemapping_feasible(self, sn, vnr, v2sindex):
        """
        Check if node mapping respects CPU constraints.
        """
        used_cpu = {}
        for n in sn.snode:
            used_cpu[n.index] = 0
        
        for vnf_id, snode_id in enumerate(v2sindex):
            used_cpu[sn.snode[snode_id].index] += vnr.vnode[vnf_id].cpu
        
        for n in sn.snode:
            if used_cpu[n.index] > n.lastcpu + 1e-6:
                return False
        return True

    def is_feasible(self, sn, vnr, v2sindex, vese2index):
        """
        Check CPU and bandwidth constraints.
        """
        # CPU constraints
        used_cpu = {}
        for n in sn.snode:
            used_cpu[n.index] = 0
        
        for vnf_id, snode_id in enumerate(v2sindex):
            used_cpu[sn.snode[snode_id].index] += vnr.vnode[vnf_id].cpu
        
        for n in sn.snode:
            if used_cpu[n.index] > n.lastcpu + 1e-6:
                return False
        
        # Bandwidth constraints
        used_bw = {}
        for e in sn.sedege:
            used_bw[e.index] = 0
        
        for e_idx, path in enumerate(vese2index):
            if path:  # if mapping exists
                bw_req = vnr.vedege[e_idx].bandwidth
                for edge_id in path:
                    used_bw[sn.sedege[edge_id].index] += bw_req
        
        for e in sn.sedege:
            if used_bw[e.index] > e.lastbandwidth + 1e-6:
                return False
        
        return True

    def apply_solution(self, sn, vnr, v2sindex, vese2index):
        """
        Apply GRASP solution to the real substrate network.

        Returns:
            updated substrate network 'sn'
        """
        vn = vnr.num_vnfs
        
        # Apply node mappings
        for i in range(vn):
            sn.snode[v2sindex[i]].lastcpu = sn.snode[v2sindex[i]].lastcpu - vnr.vnode[i].cpu
            sn.snode[v2sindex[i]].vnodeindexs.append([vnr.id, vnr.vnode[i].index])
            sn.snode[v2sindex[i]].p_load = (sn.snode[v2sindex[i]].p_load * sn.snode[v2sindex[i]].cpu + vnr.vnode[i].p_maxCpu) / sn.snode[v2sindex[i]].cpu
            vnr.vnode[i].sn_host = v2sindex[i]
        
        # Apply edge mappings
        for i in range(len(vese2index)):
            pathindex = vese2index[i]
            for j in pathindex:
                sn.sedege[j].lastbandwidth = sn.sedege[j].lastbandwidth - vnr.vedege[i].bandwidth
                sn.sedege[j].vedegeindexs.append([vnr.id, i])
                nodeindex = sn.sedege[j].nodeindex
                sn.snode[nodeindex[0]].lastbw -= vnr.vedege[i].bandwidth
                sn.snode[nodeindex[1]].lastbw -= vnr.vedege[i].bandwidth
            vnr.vedege[i].spc = pathindex
        
        vnr.nodemapping = v2sindex
        vnr.edgemapping = vese2index
        return sn

    def mapping(self, sb, vnr):
        """
        Maps the VNR (virtual network request) to the substrate network (sb).
        Uses GRASP (Greedy Randomized Adaptive Search Procedure):
        1. Construction phase: Build initial solution with RCL
        2. Local search: Try to improve the solution
        3. Repeat for max_iter iterations and keep the best
        
        Returns a dictionary containing the mapping results.
        """
        success = True 
        vn = vnr.num_vnfs 
        
        best_solution_nodes = None
        best_solution_edges = None
        best_reward = float('-inf')
        
        # GRASP iterations
        for iteration in range(self.max_iter):
            # Reset sn_host for new iteration
            for i in range(vn):
                vnr.vnode[i].sn_host = -1
            
            # Construction phase
            const_success, const_nodes, const_edges = self.construct_solution(sb, vnr)
            if not const_success:
                continue
            
            # Local search phase
            ls_nodes, ls_edges = self.local_search(sb, vnr, const_nodes, const_edges)
            
            # Check feasibility
            if not self.is_feasible(sb, vnr, ls_nodes, ls_edges):
                continue
            
            # Evaluate solution (better = higher reward)
            # Set mappings temporarily for reward calculation
            vnr.nodemapping = ls_nodes
            for i, path in enumerate(ls_edges):
                vnr.vedege[i].spc = path
            
            r2c, p_load, reward = self.getReward(vnr, sb)
            
            if reward > best_reward:
                best_reward = reward
                best_solution_nodes = ls_nodes
                best_solution_edges = ls_edges
        
        # If no solution found
        if best_solution_nodes is None:
            # Reset sn_host
            for i in range(vn):
                vnr.vnode[i].sn_host = -1
            return {
                'success': False,
                'nodemapping': [],
                'edgemapping': [],
                'nb_vnfs': 0,
                'nb_vls': 0,
                'R2C': 0.0,
                'p_load': 0.0,
                'reward': self.rejection_penalty,
                'sn': sb,
                'cause': 'GRASP',
                'nb_iter': self.max_iter
            }
        
        # Apply best solution
        sb = self.apply_solution(sb, vnr, best_solution_nodes, best_solution_edges)
        
        # Compute final metrics
        r2c, p_load, reward = self.getReward(vnr, sb)
        
        return {
            'success': True,
            'nodemapping': vnr.nodemapping,
            'edgemapping': vnr.edgemapping,
            'nb_vnfs': vnr.num_vnfs,
            'nb_vls': vnr.num_vedges,
            'R2C': r2c,
            'p_load': p_load,
            'reward': reward,
            'sn': sb,
            'cause': None,
            'nb_iter': self.max_iter
        }

    def scaling_down(self, vnr, sn, scaling_chaine):
        """
        Scales down the resource allocation for the virtual nodes in the VNR.
        """
        for i in scaling_chaine:
            vnr.vnode[i].cpu -= vnr.vnode[i].req_cpu
            sn.snode[vnr.vnode[i].sn_host].lastcpu += vnr.vnode[i].req_cpu
        return sn
    
    def scaling_up(self, vnr, sn, scaling_chaine):
        """
        Scales up the resource allocation for the virtual nodes in the VNR.
        """
        sn_c = sn.copy_for_placement()
        vnr_c = dc(vnr)
        remapping_nodes = []
        for i in scaling_chaine:
            if sn_c.snode[vnr_c.vnode[i].sn_host].lastcpu > vnr_c.vnode[i].req_cpu:
                sn_c.snode[vnr_c.vnode[i].sn_host].lastcpu -= vnr_c.vnode[i].req_cpu
                vnr_c.vnode[i].cpu += vnr_c.vnode[i].req_cpu
            else:
                remapping_nodes.append(i)
        if len(remapping_nodes) > 0:
            return {"success": False, "sn": sn, "vnr": vnr}
        else:
            return {"success": True, "sn": sn_c, "vnr": vnr_c}

