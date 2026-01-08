import random
import numpy as np
from copy import deepcopy as dc
from solver import Solver
from DQNAgent import DQNAgent
from observation import Observation

import torch
from ..DGLgraph import SnGraph
from DGLgraph import Graph as DGLgraph

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import dgl


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class GNNGRASP(Solver):
    def __init__(self, sigma, gamma, rejection_penalty,
                 learning_rate, epsilon, memory_size, batch_size,
                 num_inputs_sn, num_inputs_vnr, hidden_size, GCN_out,
                 num_actions, max_iteration, eps_min, eps_dec, alpha=0.1):
        super().__init__(sigma, rejection_penalty)

        self.agent = DQNAgent(gamma, learning_rate, epsilon, memory_size, batch_size,
                              num_inputs_sn, num_inputs_vnr, hidden_size, GCN_out,
                              num_actions, eps_min, eps_dec)

        self.max_iteration = max_iteration
        self.alpha = alpha  # GRASP threshold: 0 is pure greedy, 1 is pure random
        self.saved_transition = None

    # ------------------------------------------------------------------
    #  GRASP Node Mapping with Restricted Candidate List (RCL)
    # ------------------------------------------------------------------
    def nodemapping_grasp(self, observation, sn, vnr, vnf_idx):
        """
        Uses the GNN to score nodes, builds an RCL, and selects randomly.
        """
        # 1. Get Q-values for all actions from the GNN
        q_values = self.agent.get_q_values(observation)
        cpu_req = vnr.vnode[vnf_idx].cpu

        # 2. Filter candidates (Must have enough CPU)
        candidates = []
        for i in range(len(sn.snode)):
            if sn.snode[i].lastcpu >= cpu_req:
                candidates.append({'id': i, 'q': q_values[i].item()})

        if not candidates:
            return False, -1

        # 3. Build Restricted Candidate List (RCL)
        # Rule: q_val >= q_max - alpha * (q_max - q_min)
        q_max = max(c['q'] for c in candidates)
        q_min = min(c['q'] for c in candidates)
        threshold = q_max - self.alpha * (q_max - q_min)

        rcl = [c['id'] for c in candidates if c['q'] >= threshold]

        # 4. Randomized Selection
        action = random.choice(rcl)

        # 5. Apply Mapping
        sn.snode[action].lastcpu -= cpu_req
        sn.snode[action].vnodeindexs.append([vnr.id, vnf_idx])
        vnr.nodemapping[vnf_idx] = action
        vnr.vnode[vnf_idx].sn_host = action

        return True, action

    # ------------------------------------------------------------------
    #  Main GRASP Mapping Loop
    # ------------------------------------------------------------------
    def mapping(self, sn, vnr):
        best_results = None
        best_reward = -float('inf')

        # Run multiple construction iterations
        for i in range(self.max_iteration):
            iteration_success = True
            sn_c = dc(sn)
            vnr_i = dc(vnr)
            vnr_i.nodemapping = [-1] * vnr_i.num_vnfs
            ve2seindex = [-1] * vnr_i.num_vedges

            # Reset observation for new iteration
            sn_graph = SnGraph(sn_c, vnr_i.vnode[0].cpu, device=device)
            vnr_graph = DGLgraph(vnr_i, device=device)
            obs = Observation(sn_graph, vnr_graph, 0, [])

            # Sequential Placement (Construction Phase)
            for idx in range(vnr_i.num_vnfs):
                n_success, action = self.nodemapping_grasp(obs, sn_c, vnr_i, idx)

                if n_success:
                    # Attempt Edge Mapping for this node
                    nodes_mapped_so_far = [j for j in range(idx + 1)]
                    e_success = self.edegemapping(sn_c, vnr_i, idx, nodes_mapped_so_far, ve2seindex)

                    if e_success:
                        if idx < vnr_i.num_vnfs - 1:
                            # Update observation for next VNF
                            sn_graph = SnGraph(sn_c, vnr_i.vnode[idx + 1].cpu, device=device)
                            obs = Observation(sn_graph, vnr_graph, idx + 1, vnr_i.nodemapping[:idx + 1])
                    else:
                        iteration_success = False;
                        break
                else:
                    iteration_success = False;
                    break

            # Evaluate this iteration
            if iteration_success:
                r2c, p_load, rel, reward = self.getReward(vnr_i, sn_c)
                rel = self.calculate_reliability(vnr_i, sn_c)

                # Update Best Solution
                if reward > best_reward:
                    best_reward = reward
                    best_results = {
                        'success': True, 'nodemapping': vnr_i.nodemapping, 'edgemapping': ve2seindex,
                        'nb_vnfs': vnr_i.num_vnfs, "nb_vls": vnr_i.num_vedges, 'R2C': r2c,
                        'p_load': p_load, 'reward': reward, 'reliability': rel, 'sn': sn_c,
                        'cause': None, 'nb_iter': i
                    }

        # If any iteration succeeded, return the best one
        if best_results:
            return best_results

        # Fallback if all iterations failed
        return {'success': False, 'nodemapping': [], 'edgemapping': [], 'nb_vnfs': 0,
                'reward': self.rejection_penalty, 'reliability': 0, 'sn': sn, 'cause': 'grasp_fail'}