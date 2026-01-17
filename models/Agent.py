#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 10:55:48 2022

@author: kaouther
"""
from models.A2C import GNNA2C
import numpy as np
import torch
from models.replayBuffer import ReplayBuffer
from copy import deepcopy as dc

class Transition(object):
    """ 
    A class to represent a transition in the context of VNR placement.

    A transition captures a sequence of steps taken during the placement of 
    a Virtual Network Request (VNR). It stores the states, actions, 
    rewards, and whether the episode has ended at each step.
    """
    def __init__(self):
        self.states= []
        """ A list of states encountered during the transition. """
        self.actions = []
        """ A list of actions taken at each state. """
        self.rewards= []
        """ A list of rewards received after each action. """
        self.dones = []
        """ A list indicating whether each step was terminal. """
        self.next_value =None
        """ The value of the next state after the placement or the failure (After the last step) """

        
    def store_step(self,state,action,reward,done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        
   
    
    

class Agent(object):
    
    def __init__(self,gamma,learning_rate,epsilon,memory_size,batch_size,num_inputs_sn, num_inputs_vnr, hidden_size, GCN_out, num_actions,eps_min = 0.01, eps_dec = 5.5e-5):
        self.gamma = gamma
        """ Discount Factor """
        self.epsilon = epsilon
        """
        Used to balance exploration and exploitation in reinforcement learning.
        """
        self.num_actions=num_actions
        """ The number of actions, representing the total number of nodes in the substrate network (SN). """
        self.memory = ReplayBuffer(memory_size)
        """ A replay buffer for storing transitions. """
        self.batch_size = batch_size
        """ The size of the batch of transitions used during learning. """
        self.memory_init = 3000 
        """ The number of transitions to store before starting the learning process; this value must always be greater than the batch size. """
        self.eps_min = eps_min
        """ The minimum value for the exploration rate (epsilon) during training. """
        self.eps_dec = eps_dec
        """ The amount by which the exploration rate (epsilon) decreases with each step in the learning process. """
        self.learn_step_counter = 0
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #selecting GPU if available
        self.A2C = GNNA2C(num_inputs_sn, num_inputs_vnr, hidden_size, GCN_out, learning_rate,num_actions).to(self.device)
 
        
    def store_transition(self,transition):
        self.memory.store(transition)
        
    def choose_action(self,observation,vnf_cpu,sn_cpu):
            """
            Chooses an action based on the current observation using an epsilon-greedy strategy.

            Args:
                observation: An Observation object that contains the current state of the system.
                vnf_cpu: The CPU required by the VNF being placed.
                sn_cpu: A list containing the available CPU resources of each node in the substrate network.

            Returns:
                action: The chosen action (node index) for placing the VNF.
                value: The estimated value of the current state from the A2C model.
                log_prob: The logarithm of the probability of the chosen action.
            """
            # Get the illegal actions from the observation's node mapping
            illegal_actions=dc(observation.node_mapping)
            # Get the estimated value and policy distribution from the A2C model
            value,policy_dist= self.A2C([observation])
            # Clone the probability distribution for manipulation
            probs = policy_dist[0].clone()  
            probs.detach()
 
            # Epsilon-greedy action selection
            if  np.random.random() < self.epsilon:
                # Select legal actions based on available CPU resources
                legal_actions = np.array([index for index, element in enumerate(sn_cpu) if element > vnf_cpu])
                legal_actions=legal_actions[~np.isin(legal_actions,illegal_actions)]
                
                # Randomly choose an action from the legal actions if available
                if len(legal_actions)>0:
                    action = np.random.choice(legal_actions)
                else: 
                    action = -1  # No legal actions available

            else:
                # Set the probabilities of illegal actions to negative infinity to avoid selection
                probs[illegal_actions] = -float('Inf')
                action = torch.argmax(probs).item() # Choose the action with the highest probability

            # Calculate the log probability of the chosen action
            log_prob = torch.log(probs[action])

            return action, value, log_prob
            

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min
        


    def learn(self):
        # Enable anomaly detection for debugging
        torch.autograd.set_detect_anomaly(True)

        # Return early if we haven't collected enough experiences
        if self.memory.mem_cntr < self.memory_init:
            return
        
        # Sample a batch of experiences from memory
        states, actions,rewards, dones, next_values,transition_lens = self.memory.sample_buffer(self.batch_size)
        
        # Get predicted values and policy distributions from the A2C model
        values, policy_dists = self.A2C(states)

        # Compute log probabilities of the actions taken
        log_probs =[]
        for i, policy_dist in enumerate(policy_dists):
            log_probs.append(torch.log(policy_dist[actions[i]]))
        
        # Initialize target Q-value from the last next value
        q_val = next_values[len(next_values)-1]
        lenght= len(rewards) 
        q_vals = np.zeros((lenght, 1))
          
        # target values are calculated backward
        # it's super important to handle correctly done states,
        # for those cases we want our to target to be equal to the reward only
        j = len(next_values)-1
        k = 0
        for i, reward in enumerate(rewards[::-1]):
            done = dones[lenght-1 - i]
            q_val = reward + self.gamma*q_val*(1.0-done)
            q_vals[lenght-1 - i] = q_val # store values from the end to the beginning
            k+=1
            if k == transition_lens[j]:
                j-= 1
                k = 0
                q_val = next_values[j]

        advantage = torch.tensor(q_vals, device=self.device).squeeze() - values.squeeze()
        critic_loss = advantage.pow(2).mean()
        actor_loss = (-torch.stack(log_probs)*advantage.detach()).mean()
        actor_critic_loss = critic_loss+actor_loss 

        self.A2C.optimizer.zero_grad()
        actor_critic_loss.backward()
        self.A2C.optimizer.step()
        self.decrement_epsilon()