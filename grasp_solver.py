# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 20:33:50 2022

@author: amirgu1
"""
# from dgl.examples.pytorch.graph_matching.examples import edge_mapping
from solver import GlobalSolver, Solver
import networkx as nx
import numpy as np
from copy import deepcopy as dc
import random

class Grasp(Solver):
    def __init__(self, sigma, rejection_penalty,
                 max_iter=20,           # nombre d’itérations GRASP
                 alpha=0.3):            # paramètre de RCL (0..1)

        super().__init__(sigma, rejection_penalty)

        self.max_iter = max_iter
        self.alpha = alpha             # fraction of candidates in RCL


    def rev2cost(self, vnr):
        """
        Calculate the revenue-to-cost (R2C) ratio for a VNR placement.

        The R2C ratio is determined by dividing the total bandwidth of the VNR edges by
        the sum of the total bandwidth and the duplicate bandwidth (when an edge is mapped
        to multiple paths).

        Args:
            vnr: The virtual network request (VNR) containing edge information.

        Returns:
            float: The R2C ratio (revenue to cost).
        """
        vsum_edeges = 0
        for i in range(len(vnr.vedege)):
            vsum_edeges = vsum_edeges + vnr.vedege[i].bandwidth
        dup = 0
        for i in range(len(vnr.vedege)):
            if len(vnr.vedege[i].spc) > 1:
                dup = dup + vnr.vedege[i].bandwidth * (len(vnr.vedege[i].spc) - 1)
        return vsum_edeges / (vsum_edeges + dup)

    def evaluateSolution(self, subNet, vnr, nodemapping, LinksPlacementSol):
        """
            Evaluate solution and return the cost of the placement of this sol
                ○ Input : solution
		           ○ Output : cost of the solution
        """
        r2c = self.rev2cost(vnr)
        return r2c

    def nodemapping(self, sb, vnr):
        """
        Randomized greedy node mapping (compatible with FF/GRASP + simulator).

        Returns:
            success (bool)
            v2sindex (list of substrate node indices)
        """

        vn = vnr.num_vnfs
        v2sindex = [None] * vn

        # Randomized order for GRASP/FF
        vnf_order = list(range(vn))
        random.shuffle(vnf_order)

        for vi in vnf_order:

            cpu_req = vnr.vnode[vi].cpu

            # Find all feasible substrate nodes
            candidates = []
            for si, snode in enumerate(sb.snode):
                if cpu_req <= snode.lastcpu:  # <= instead of <
                    candidates.append(si)

            if len(candidates) == 0:
                return False, []

            # Greedy selection based on available CPU
            candidates_sorted = sorted(
                candidates,
                key=lambda si: sb.snode[si].lastcpu,
                reverse=True
            )

            # Optional: GRASP RCL (if you want)
            RCL_size = max(1, int(0.3 * len(candidates_sorted)))
            RCL = candidates_sorted[:RCL_size]

            chosen_snode = random.choice(RCL)

            v2sindex[vi] = chosen_snode

        return True, v2sindex

    def construct_solution(self, sn, vnr):
        """
        GRASP construction phase:
            1) Build a feasible node mapping using a randomized greedy strategy
            2) Build a feasible edge mapping based on constructed nodes
            3) Return a "solution" dict without modifying sn

        Output format:
            solution = {
                "nodes": {vnf_id: substrate_node_id},
                "links": {vedge_id: [path_edge_ids]}
            }
        """

        num_vnfs = vnr.num_vnfs
        num_vedges = vnr.num_vedges
        solution = {"nodes": {}, "links": {}}

        # ---------------------------------------------
        # PHASE 1 — NODE MAPPING (Randomized Greedy)
        # ---------------------------------------------

        # We will try to place VNFs in random order
        vnf_order = list(range(num_vnfs))
        random.shuffle(vnf_order)

        # Temporary sn_host required for edegemapping()
        temp_hosts = {i: None for i in range(num_vnfs)}

        for vnf_id in vnf_order:
            cpu_req = vnr.vnode[vnf_id].cpu

            # Build list of feasible substrate nodes
            candidates = []
            for snode in sn.snode:
                if snode.lastcpu >= cpu_req:
                    candidates.append(snode.index)

            # No feasible node → failure
            if len(candidates) == 0:
                return None

            # -------------------------------
            # RCL — Restricted Candidate List
            # -------------------------------
            # Greedy criterion = available CPU
            candidates_sorted = sorted(
                candidates,
                key=lambda idx: sn.snode[idx].lastcpu,
                reverse=True
            )

            # RCL size = max(1, alpha * len(candidates))
            RCL_size = max(1, int(self.alpha * len(candidates_sorted)))
            RCL = candidates_sorted[:RCL_size]

            # Choose random node from RCL
            chosen_snode = random.choice(RCL)

            # Add mapping
            solution["nodes"][vnf_id] = chosen_snode
            temp_hosts[vnf_id] = chosen_snode
            vnr.vnode[vnf_id].sn_host = chosen_snode  # needed for edge phase

        # --------------------------------------------------------
        # PHASE 2 — EDGE MAPPING (using your edegemapping())
        # --------------------------------------------------------

        # Use substrate copies for edge feasibility check
        asnode = dc(sn.snode)
        asedge = dc(sn.sedege)

        success, edge_map = self.edegemapping(asnode, asedge, vnr,
                                              [temp_hosts[i] for i in range(num_vnfs)])

        if not success or len(edge_map) != num_vedges:
            return None

        # Fill solution["links"]
        for e_idx in range(num_vedges):
            solution["links"][e_idx] = edge_map[e_idx]

        return solution

    def local_search(self, solution, sn, vnr):
        """
        Try to improve the current mapping by moving one VNF to a less-loaded node.
        """
        best_sol = solution
        best_cost = self.evaluate_solution(solution, sn)

        for vnf_id, snode_id in solution["nodes"].items():
            for candidate in sn.snode:
                if candidate.lastcpu < vnr.vnode[vnf_id].cpu:
                    continue
                new_sol = {
                    "nodes": dict(solution["nodes"]),
                    "links": dict(solution["links"])
                }
                new_sol["nodes"][vnf_id] = candidate.index
                cost = self.evaluate_solution(new_sol, sn)
                if cost < best_cost and self.is_feasible(new_sol, sn, vnr):
                    best_cost = cost
                    best_sol = new_sol
        return best_sol

    def is_feasible(self, solution, sn, vnr):
        """
        Check CPU, bandwidth, and reliability constraints.
        """

        # -----------------------------------------------------
        # 1) CPU constraints
        # -----------------------------------------------------
        used_cpu = {n.index: 0 for n in sn.snode}

        for vnf_id, snode_id in solution["nodes"].items():
            used_cpu[snode_id] += vnr.vnode[vnf_id].cpu

        for n in sn.snode:
            if used_cpu[n.index] > n.lastcpu + 1e-6:
                return False

        # -----------------------------------------------------
        # 2) Bandwidth constraints
        # -----------------------------------------------------
        used_bw = {e.index: 0 for e in sn.sedege}

        for e_idx, path in solution["links"].items():
            bw_req = vnr.vedege[e_idx].bandwidth
            for edge_id in path:
                used_bw[edge_id] += bw_req

        for e in sn.sedege:
            if used_bw[e.index] > e.lastbandwidth + 1e-6:
                return False

        # -----------------------------------------------------
        # 3) Reliability constraints (per virtual edge)
        # -----------------------------------------------------
        for e_idx, path in solution["links"].items():

            # 1) Compute reliability of the substrate path
            rel_path = self.calculate_rel(path)

            # 2) Required reliability for this virtual edge
            rel_required = vnr.vedege[e_idx].rel

            # 3) Check constraint
            if rel_path < rel_required:
                return False

        # Everything is feasible
        return True

    def apply_solution(self, sn, vnr, solution):
        """
        Apply a GRASP solution to the REAL substrate network 'sn'
        and update the VNR mapping structures.

        solution format:
            solution["nodes"] = {vnf_id: snode_id}
            solution["links"] = {vedge_id: [edge_ids]}

        Returns:
            updated substrate network 'sn'
        """

        # -------------------------------------------------------
        # 1) APPLY NODE MAPPING
        # -------------------------------------------------------
        nodemap = solution["nodes"]
        vnr.nodemapping = [None] * vnr.num_vnfs

        for vnf_id, snode_id in nodemap.items():
            vnode = vnr.vnode[vnf_id]
            snode = sn.snode[snode_id]

            # CPU update
            snode.lastcpu -= vnode.cpu

            # Track mapping in substrate
            # (vnr.id, vnf_id) tuple required!! (not list)
            snode.vnodeindexs.append((vnr.id, vnf_id))

            # Update potential load
            snode.p_load = (snode.p_load * snode.cpu + vnode.p_maxCpu) / snode.cpu

            # Update VNF host
            vnode.sn_host = snode_id

            # Store mapping in VNR
            vnr.nodemapping[vnf_id] = snode_id

        # -------------------------------------------------------
        # 2) APPLY EDGE MAPPING
        # -------------------------------------------------------
        edgemap = solution["links"]
        vnr.edgemapping = [None] * vnr.num_vedges

        for e_idx, path in edgemap.items():

            vedge = vnr.vedege[e_idx]
            bw_req = vedge.bandwidth

            # For each substrate edge used
            for edge_id in path:
                sedege = sn.sedege[edge_id]

                # Update remaining BW
                sedege.lastbandwidth -= bw_req

                # Track mapping
                sedege.vedegeindexs.append((vnr.id, e_idx))

                # Update node lastbw
                n1, n2 = sedege.nodeindex
                sn.snode[n1].lastbw -= bw_req
                sn.snode[n2].lastbw -= bw_req

            # Store mapping in VNR
            vedge.spc = path
            vnr.edgemapping[e_idx] = path

        # -------------------------------------------------------
        # 3) Return updated substrate
        # -------------------------------------------------------
        return sn

    def mapping(self, sb, vnr):
        """
        Attempt to map a Virtual Network Request (VNR) to a substrate network (sb).

        Steps:
              1. Map virtual nodes (VNFs) to substrate nodes.
              2. If node mapping succeeds, map virtual links to substrate paths.
              3. Apply resource updates (CPU, bandwidth, load) only if both succeed.

        Returns:
                dict: A mapping result with:
                    - success (bool)
                    - nodemapping (list)
                    - edgemapping (list)
                    - sn (updated substrate network)
                    - cause ("node" | "edge" | "exception" | None)
                    - metrics: R2C, p_load, reward, reliability
        """
        # print('GRASP mapping function')

        # Default result
        result = {
            'success': False,
            'nodemapping': [],
            'edgemapping': [],
            'nb_vnfs': 0,
            'nb_vls': 0,
            'R2C': 0,
            'p_load': 0,
            'reward': self.rejection_penalty,
            'reliability': 1.0,
            'sn': sb,
            'cause': None,
            'nb_iter': None
        }

        # --- Step 1: Node mapping ---
        node_success, node_map = self.nodemapping(sb, vnr)
        if not node_success or not node_map:
            result['cause'] = 'node'
            return result

        # IMPORTANT: set sn_host BEFORE edge mapping
        for i, snode_index in enumerate(node_map):
            vnr.vnode[i].sn_host = snode_index

        # --- Step 2: Edge mapping (on deep copies) ---
        asnode = dc(sb.snode)
        asedge = dc(sb.sedege)
        edge_success, edge_map = self.edegemapping(asnode, asedge, vnr, node_map)
        if not edge_success or not edge_map:
            result['cause'] = 'edge'
            return result

        # --- Step 3: Apply mappings on real substrate ---
        try:
            # Apply node mappings
            for i, snode_index in enumerate(node_map):
                snode = sb.snode[snode_index]
                vnode = vnr.vnode[i]

                # CPU
                snode.lastcpu -= vnode.cpu

                # Store mapping as TUPLE, with index i (not vnode.index)
                snode.vnodeindexs.append((vnr.id, i))

                # Update p_load
                snode.p_load = (
                                       snode.p_load * snode.cpu + vnode.p_maxCpu
                               ) / snode.cpu

                # Redundant but safe: ensure sn_host is consistent
                vnode.sn_host = snode_index

            # Apply edge mappings
            for i, path_indices in enumerate(edge_map):
                vedge = vnr.vedege[i]

                for e_idx in path_indices:
                    sedege = sb.sedege[e_idx]
                    sedege.lastbandwidth -= vedge.bandwidth

                    # Store mapping as TUPLE
                    sedege.vedegeindexs.append((vnr.id, i))

                    # Update residual bandwidth in both endpoint nodes
                    n1, n2 = sedege.nodeindex
                    sb.snode[n1].lastbw -= vedge.bandwidth
                    sb.snode[n2].lastbw -= vedge.bandwidth

                # Store substrate path in VNR
                vedge.spc = path_indices

            # Store mappings in VNR
            vnr.nodemapping = node_map
            vnr.edgemapping = edge_map

            # --- Step 4: Compute metrics ---
            # (You can keep or remove this first rev2cost, since getReward recomputes)
            # r2c = self.rev2cost(vnr)
            r2c, p_load, reliability, reward = self.getReward(vnr, sb)

            # --- Step 5: Build success result ---
            result.update({
                'success': True,
                'nodemapping': node_map,
                'edgemapping': edge_map,
                'nb_vnfs': vnr.num_vnfs,
                'nb_vls': vnr.num_vedges,
                'R2C': r2c,
                'p_load': p_load,
                'reward': reward,
                'reliability': reliability,
                'sn': sb,
                'cause': None
            })

        except Exception as e:
            # You can log e if needed
            result['cause'] = 'exception'

        return result

    def evaluate_solution(self, solution, sn, vnr):
        """
        Cost function for GRASP.
        Lower cost = better solution.

        Includes:
            - CPU usage
            - Bandwidth * hop count
            - Reliability penalty (optional)
        """

        cpu_cost = 0
        bw_cost = 0
        rel_cost = 0

        # --- CPU cost ---
        for vnf_id, snode_id in solution["nodes"].items():
            cpu_cost += vnr.vnode[vnf_id].cpu

        # --- Bandwidth cost (= hop count * bw) ---
        for e_idx, path in solution["links"].items():
            bw_req = vnr.vedege[e_idx].bandwidth
            bw_cost += bw_req * len(path)

            # --- Reliability penalty ---
            rel_path = self.calculate_rel(path)
            rel_required = vnr.vedege[e_idx].rel
            if rel_path < rel_required:
                # Heavy penalty for unreliability
                rel_cost += (rel_required - rel_path) * 1000

        return cpu_cost + bw_cost + rel_cost

    def run(self, sn, vnr):
        """
        GRASP main execution:
          1. Construct initial solution
          2. Improve with local search
          3. Select best feasible solution
          4. Apply solution to substrate
          5. Return mapping results
        """

        best_solution = None
        best_cost = float('inf')

        # --- GRASP iterations ---
        for _ in range(self.max_iter):

            # 1) Construct greedy randomized solution
            solution = self.construct_solution(sn, vnr)
            if solution is None:
                continue

            # 2) Improve with local search
            improved = self.local_search(solution, sn, vnr)

            # 3) Check feasibility
            if not self.is_feasible(improved, sn, vnr):
                continue

            # 4) Evaluate solution
            # cost = self.evaluate_solution(improved, sn, vnr)
            cost = self.getReward(vnr, sn)
            print("cost:", cost)

            # 5) Keep best solution
            if cost < best_cost:
                best_cost = cost
                best_solution = improved

        # -----------------------------------------------------
        # Mapping failed → return rejection
        # -----------------------------------------------------
        if best_solution is None:
            return {
                'success': False,
                'nodemapping': [],
                'edgemapping': [],
                'nb_vnfs': 0,
                'nb_vls': 0,
                'R2C': 0.0,
                'p_load': 0.0,
                'reward': self.rejection_penalty,
                'reliability': 0.0,
                'sn': sn,
                'cause': 'GRASP',
                'nb_iter': self.max_iter
            }

        # -----------------------------------------------------
        # Apply best solution to substrate (REAL allocation)
        # -----------------------------------------------------
        sn = self.apply_solution(sn, vnr, best_solution)

        # Compute metrics AFTER mapping
        r2c, p_load, reliability, reward = self.getReward(vnr, sn)

        return {
            'success': True,
            'nodemapping': vnr.nodemapping,
            'edgemapping': vnr.edgemapping,
            'nb_vnfs': vnr.num_vnfs,
            'nb_vls': vnr.num_vedges,
            'R2C': r2c,
            'p_load': p_load,
            'reward': reward,
            'reliability': reliability,
            'sn': sn,
            'cause': None,
            'nb_iter': self.max_iter
        }

