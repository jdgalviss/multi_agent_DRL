from model import Actor, Critic
from utilities import hard_update, soft_update
from torch.optim import Adam
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim

from replay_buffer import ReplayBuffer
from utilities import OUNoise
import numpy as np
import random

# add OU noise for exploration
from utilities import OUNoise

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DDPGAgent:
    def __init__(self, state_size, action_size, random_seed, num_agents = 2, lr_actor = 1e-3, lr_critic = 1e-3, gamma = 0.99, tau = 1e-3, batch_size =512, buffer_size = int(1e5), update_every = 20, num_updates = 10):
        super(DDPGAgent, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.n_agents = num_agents
        self.seed = random.seed(random_seed)
        self.seed = torch.manual_seed(random_seed)
        self.update_every = update_every
        self.num_updates = num_updates
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        
        # Actor and Critic with their respective targets
        self.actor = Actor(state_size, action_size, random_seed).to(device)
        self.target_actor = Actor(state_size, action_size, random_seed).to(device)  
        self.critic = Critic(state_size, action_size, random_seed).to(device)
        self.target_critic = Critic(state_size, action_size, random_seed).to(device)
        
        # Optimizers
        self.actor_optimizer = Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr_critic)
        
        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)
        
        # Noise Process
        self.noise = OUNoise((num_agents, action_size), random_seed)
        self.memory = ReplayBuffer(buffer_size, batch_size, random_seed)
        self.t_step = 0
    
    # Reset Noise
    def reset(self):
        self.noise.reset()
        
    def step(self, states, actions, rewards, next_states, dones ):
        # Add to replay buffer for each agent
        for i in range(self.n_agents):
            self.memory.add(states[i, :], actions[i, :], rewards[i], next_states[i, :], dones[i])
        
        # Update every 'update_every' steps for 'num_updates' times
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            # if enough samples are there then learn
            if len(self.memory) > self.batch_size:
                for i in range(self.num_updates):
                    samples = self.memory.sample()
                    self.learn(samples)   
                    
    def learn(self, samples):
        states, actions, rewards, next_states, dones = samples
        
        #Train the critic
        # Get the actions corresponding to next states from actor and then their Q-values 
        # from target critic
        actions_next = self.target_actor(next_states)
        Q_targets_next = self.target_critic(next_states, actions_next)
        
        # Compute Q targets using TD-difference
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        
        # Compute critic loss, perform backward pass and training step
        Q_expected = self.critic(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1)
        self.critic_optimizer.step()
        
        
        # Update Actor
        # Compute Actor loss
        actions_pred = self.actor(states)
        # -ve sign because we want to maximise this value
        actor_loss = -self.critic(states, actions_pred).mean()
        
        # minimizing the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        
        # update target networks
        soft_update(self.critic, self.target_critic, self.tau)
        soft_update(self.actor, self.target_actor, self.tau)
    
    
    def act(self, states, add_noise=True):
        """Returns actions for given state as per current policy"""
        states = torch.from_numpy(states).float().to(device)
        actions = np.zeros((self.n_agents, self.action_size))
                               
        self.actor.eval()
        with torch.no_grad():
            for i in range(self.n_agents):
                action_i = self.actor(states[i]).cpu().data.numpy()
                actions[i, :] = action_i
        self.actor.train()

        if add_noise:
            actions += self.noise.sample()

        return np.clip(actions, -1, 1)
