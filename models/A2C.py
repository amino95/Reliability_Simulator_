#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 10:12:48 2022

@author: kaouther
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import dgl


class GCNModule(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super().__init__()
        # Mockup, assuming 3 layers
        self.linear1 = nn.Linear(in_feats, hid_feats)
        self.linear2 = nn.Linear(hid_feats, hid_feats)
        self.linear3 = nn.Linear(hid_feats, out_feats)

    def forward(self, graph, feat):
        # Pour le test, on retourne feat après quelques transformations linéaires
        feat = F.relu(self.linear1(feat))
        feat = F.relu(self.linear2(feat))
        feat = self.linear3(feat)
        return feat


class Vnr_AttentionLayer(nn.Module):
    def __init__(self, in_feats, activation):
        super().__init__()
        # Simule un output qui n'est PAS un readout
        self.linear = nn.Linear(in_feats, in_feats)
        # Si vous retournez une représentation de noeud, le problème se pose ici.

    def forward(self, h, idx):
        # Hypothèse 1: Elle ne fait que prendre le noeud sélectionné 'idx'
        if h.dim() == 2:
            return h[idx, :].unsqueeze(0)  # [1, GCN_out]
        # Hypothèse 2: Elle fait une agrégation correcte (doit être [1, GCN_out])
        return h.mean(dim=0, keepdim=True)  # Fallback pour l'exemple


class Sn_AttentionLayer(nn.Module):
    def __init__(self, in_feats, activation):
        super().__init__()
        # Simule un output qui n'est PAS un readout
        self.linear = nn.Linear(in_feats, in_feats)

    def forward(self, h):
        # Hypothèse: Elle fait une agrégation correcte (doit être [1, GCN_out])
        return h.mean(dim=0, keepdim=True)  # Fallback pour l'exemple


# --------------------------------------------------------------------


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQN(nn.Module):

    def __init__(self, num_inputs, num_actions, hidden_size):
        super(DQN, self).__init__()

        # VÉRIFICATION: num_inputs doit être 128 (2*GCN_out)
        self.layer1 = nn.Linear(num_inputs, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, num_actions)

    def forward(self, state):
        # state.shape DOIT être [batch_size, 128]
        # Si state.shape est [17728, 2], l'erreur se produit ici.
        state = F.relu(self.layer1(state))
        state = F.relu(self.layer2(state))
        return self.layer3(state)


class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size):
        super(ActorCritic, self).__init__()

        self.num_actions = num_actions

        self.critic_linear1 = nn.Linear(num_inputs, hidden_size)

        self.critic_linear2 = nn.Linear(hidden_size, hidden_size)
        self.critic_linear3 = nn.Linear(hidden_size, 1)

        self.actor_linear1 = nn.Linear(num_inputs, hidden_size)
        self.ln1 = nn.LayerNorm(num_inputs)
        self.actor_linear2 = nn.Linear(hidden_size, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.actor_linear3 = nn.Linear(hidden_size, num_actions)
        self.ln3 = nn.LayerNorm(hidden_size)
        self.ln4 = nn.LayerNorm(num_actions)

    def forward(self, state):
        # print(">>> Using ActorCritic forward, state shape:", state.shape)

        value = F.relu(self.critic_linear1(state))
        value = F.relu(self.critic_linear2(value))
        value = self.critic_linear3(value)

        norm_state = self.ln1(state)
        policy_dist = F.tanh(self.actor_linear1(norm_state))
        policy_dist = self.ln2(policy_dist)
        policy_dist = F.tanh(self.actor_linear2(policy_dist))
        policy_dist = self.ln3(policy_dist)
        policy_dist = F.softmax(self.ln4(self.actor_linear3(policy_dist)), dim=1)
        # Clipping probabilities
        # You can set a threshold below which probabilities will be clipped
        # For example, setting mib=1e-4 will clip all probabilities below 1e-4 to 1e-4
        policy_dist = torch.clamp(policy_dist, min=1e-4)

        return value, policy_dist


class GNNDQN(nn.Module):

    def __init__(self, num_inputs_sn, num_inputs_vnr, hidden_size, GCN_out, learning_rate, num_actions):
        super(GNNDQN, self).__init__()

        self.learning_rate = learning_rate

        self.gcn_vnr = GCNModule(num_inputs_vnr, hidden_size, GCN_out)
        self.gcn_sn = GCNModule(num_inputs_sn, hidden_size, GCN_out)

        self.att_vnr = Vnr_AttentionLayer(GCN_out, torch.tanh)
        self.att_sn = Sn_AttentionLayer(GCN_out, torch.tanh)

        # num_inputs = 2 * GCN_out (128 si GCN_out=64)
        self.actor_critic = DQN(2 * GCN_out, num_actions, hidden_size)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.loss = nn.MSELoss()
        self.to(device)

    def forward_single(self, obs):
        """Forward pour un seul Observation"""

        sn_graph = obs.sn.graph
        vnr_graph = obs.vnr.graph

        h_sn = self.gcn_sn(sn_graph, sn_graph.ndata['features'])
        h_vnr = self.gcn_vnr(vnr_graph, vnr_graph.ndata['features'])

        sn_rep = self.att_sn(h_sn)
        vnr_rep = self.att_vnr(h_vnr, obs.idx)

        # ATTENTION: Si sn_rep ou vnr_rep n'a pas la bonne forme ([1, GCN_out]),
        # vous aurez une erreur ici ou en aval.
        state = torch.cat([sn_rep, vnr_rep], dim=-1).unsqueeze(0)

        # print(">>> h_sn:", h_sn.shape)
        # print(">>> h_vnr:", h_vnr.shape)
        #
        # print(">>> att_sn:", sn_rep.shape)
        # print(">>> att_vnr:", vnr_rep.shape)
        # print(">>> final state:", state.shape)
        return self.actor_critic(state)

    def forward_batch(self, obs_list):
        """
        Version compatible avec les classes d'attention originales
        (ne prend qu'un seul argument).
        """
        sn_reps = []
        vnr_reps = []

        for obs in obs_list:
            # 1. Extraction et conversion locale
            # (On est obligé de traiter un par un pour respecter votre signature .forward(matrix))
            nx_sn = obs.sn.graph
            nx_vnr = obs.vnr.graph

            d_sn = dgl.from_networkx(nx_sn).to(device)
            d_vnr = dgl.from_networkx(nx_vnr).to(device)

            # 2. Injection des features
            s_feats = torch.tensor(obs.sn.getFeatures(obs.vnr.getFeatures()[obs.idx][0]),
                                   dtype=torch.float32).to(device)
            v_feats = torch.tensor(obs.vnr.getFeatures(),
                                   dtype=torch.float32).to(device)

            # 3. Inférence GCN (unitairement pour respecter l'attention)
            h_sn = self.gcn_sn(d_sn, s_feats)
            h_vnr = self.gcn_vnr(d_vnr, v_feats)

            # 4. Attention (Signature originale respectée : 1 ou 2 arguments selon la classe)
            # Sn_AttentionLayer.forward(matrix) -> 1 argument
            sn_rep = self.att_sn(h_sn)

            # Vnr_AttentionLayer.forward(matrix, vnf) -> 2 arguments
            vnr_rep = self.att_vnr(h_vnr, obs.idx)

            # On s'assure que les reps sont bien plates [1, hidden_dim]
            sn_reps.append(sn_rep.view(1, -1))
            vnr_reps.append(vnr_rep.view(1, -1))

        # 5. Re-combinaison en batch pour l'Actor-Critic
        sn_batch = torch.cat(sn_reps, dim=0)
        vnr_batch = torch.cat(vnr_reps, dim=0)

        combined_state = torch.cat([sn_batch, vnr_batch], dim=-1)

        return self.actor_critic(combined_state)


    def forward(self, observation):
        """
        Gère l'appel pour un seul objet ou un batch (liste d'objets).
        """
        # Vérifiez si l'entrée est une liste d'observations (un batch)
        if isinstance(observation, list):
            # Utilisez la méthode batch
            return self.forward_batch(observation)
        else:
            # Sinon, supposez que c'est une seule observation
            return self.forward_single(observation)


class GNNA2C(nn.Module):

    def __init__(self, num_inputs_sn, num_inputs_vnr, hidden_size, GCN_out, learning_rate, num_actions):
        super(GNNA2C, self).__init__()

        self.learning_rate = learning_rate

        self.gcn_vnr = GCNModule(num_inputs_vnr, hidden_size, GCN_out)
        self.gcn_sn = GCNModule(num_inputs_sn, hidden_size, GCN_out)

        self.att_vnr = Vnr_AttentionLayer(GCN_out, torch.tanh)
        self.att_sn = Sn_AttentionLayer(GCN_out, torch.tanh)

        self.actor_critic = ActorCritic(2 * GCN_out, num_actions, hidden_size)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.loss = nn.MSELoss()
        self.to(device)

    def forward(self, observation):
        sn_graph = dgl.batch([el.sn.graph for el in observation]).to(device)
        vnr_graph = dgl.batch([el.vnr.graph for el in observation]).to(device)
        h_sn = sn_graph.ndata['features']
        h_vnr = vnr_graph.ndata['features']

        h_sn = self.gcn_sn(sn_graph, h_sn)
        h_vnr = self.gcn_vnr(vnr_graph, h_vnr)

        sn_graph.ndata['h'] = h_sn
        vnr_graph.ndata['h'] = h_vnr

        sn_rep = [el.ndata['h'] for el in dgl.unbatch(sn_graph)]
        vnr_rep = [el.ndata['h'] for el in dgl.unbatch(vnr_graph)]

        # Le problème est dans cette boucle si self.att_sn(sn_rep[i]) ne fait pas de readout.
        state = [torch.cat([self.att_sn(sn_rep[i]), self.att_vnr(vnr_rep[i], observation[i].idx)]).to(device)
                 for i in range(len(observation))]

        # Assurez-vous que chaque élément de 'state' est un tenseur [1, 128] ou [128].
        state = torch.cat(state).view(len(observation), -1).to(device)

        value, policy_dist = self.actor_critic(state)
        return value, policy_dist
        

        
