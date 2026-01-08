from models.A2C import GNNDQN
import numpy as np
import torch
import torch.nn as nn
from models.replayBuffer import ReplayBufferDQN as ReplayBuffer
from copy import deepcopy as dc

torch.set_default_dtype(torch.float32)

class DQNTransition(object):
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.next_states = []

    def store_step(self, state, action, reward, done, next_state):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.next_states.append(next_state)


import os
import sys
import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy as dc

# --- FIX: Ajout du chemin pour trouver 'observation' ---
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from observation import Observation


# Assurez-vous que GNNDQN et ReplayBuffer sont importés ou définis ici
# from model_file import GNNDQN, ReplayBuffer

class DQNAgent(object):
    def __init__(self, gamma, learning_rate, epsilon, memory_size, batch_size,
                 num_inputs_sn, num_inputs_vnr, hidden_size, GCN_out, num_actions,
                 eps_min=0.01, eps_dec=5.5e-5, tau=0.01, memory_init=3000):

        self.gamma = gamma
        self.epsilon = epsilon
        self.tau = tau
        self.num_actions = num_actions
        self.memory = ReplayBuffer(memory_size)
        self.batch_size = batch_size
        self.memory_init = memory_init
        self.eps_min = eps_min
        self.eps_dec = eps_dec

        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Réseaux
        self.policy_net = GNNDQN(num_inputs_sn, num_inputs_vnr, hidden_size, GCN_out, learning_rate, num_actions).to(
            self.device)
        self.target_net = GNNDQN(num_inputs_sn, num_inputs_vnr, hidden_size, GCN_out, learning_rate, num_actions).to(
            self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def get_q_values(self, observation):
        """Retourne les valeurs Q pour un état donné."""
        self.policy_net.eval()
        with torch.no_grad():
            # On passe une liste car le GNN attend souvent un batch
            q_values = self.policy_net([observation])
        return q_values[0]

    def store_transition(self, transition):
        self.memory.store(transition)

    def choose_action(self, observation, vnf_cpu, sn_cpu):
        """Action classique Epsilon-Greedy."""
        illegal_actions = dc(observation.node_mapping)

        if np.random.random() < self.epsilon:
            legal_actions = np.array([index for index, element in enumerate(sn_cpu) if element >= vnf_cpu])
            legal_actions = legal_actions[~np.isin(legal_actions, illegal_actions)]
            action = np.random.choice(legal_actions) if len(legal_actions) > 0 else -1
        else:
            values = self.get_q_values(observation).clone()
            values[illegal_actions] = -float('Inf')
            action = torch.argmax(values).item()
        return action

    def choose_action_grasp(self, observation, vnf_cpu, sn_cpu, alpha):
        """Action utilisant la logique GRASP (Restricted Candidate List)."""
        illegal_actions = dc(observation.node_mapping)

        # 1. Identifier les indices légaux
        legal_indices = [
            i for i, cpu in enumerate(sn_cpu)
            if cpu >= vnf_cpu and i not in illegal_actions
        ]

        if not legal_indices:
            return -1

        # 2. Obtenir les Q-values pour ces indices
        q_values = self.get_q_values(observation)
        legal_q_values = q_values[legal_indices]

        # 3. Calculer le seuil GRASP
        q_max = torch.max(legal_q_values).item()
        q_min = torch.min(legal_q_values).item()
        threshold = q_max - alpha * (q_max - q_min)

        # 4. Construire la liste RCL
        rcl = [idx for idx in legal_indices if q_values[idx] >= threshold]

        # 5. Sélection aléatoire dans la RCL
        return np.random.choice(rcl)

    def learn(self):
        if self.memory.mem_cntr < self.memory_init:
            return

        states, actions, rewards, dones, next_states, _ = self.memory.sample_buffer(self.batch_size)

        # Tensors conversion avec correction de Dtype
        actions_tensor = torch.tensor(actions, device=self.device, dtype=torch.long).unsqueeze(1)
        rewards_tensor = torch.tensor(rewards, device=self.device, dtype=torch.float)
        done_tensor = torch.tensor(dones, device=self.device, dtype=torch.bool)

        # Target Q-values
        self.target_net.eval()
        with torch.no_grad():
            next_q_values = self.target_net(next_states)
            next_state_values = next_q_values.max(1).values
            expected_q = rewards_tensor + (self.gamma * next_state_values * (~done_tensor).float())

        # Current Q-values
        self.policy_net.train()
        current_q = self.policy_net(states).gather(1, actions_tensor)

        # Loss & Optimization
        loss = nn.MSELoss()(current_q, expected_q.unsqueeze(1))
        self.policy_net.optimizer.zero_grad()
        loss.backward()
        self.policy_net.optimizer.step()

        self.decrement_epsilon()
        self.soft_update(self.policy_net, self.target_net)

    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def decrement_epsilon(self):
        self.epsilon = max(self.eps_min, self.epsilon - self.eps_dec)