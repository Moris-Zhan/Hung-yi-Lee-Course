import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from torch.distributions import Categorical

device = "cuda" if torch.cuda.is_available() else "cpu"

"""## Actor-Critic
It's time to define my own network!
Notice that both actor and critic can share some of the net.
"""

class sharedNetwork(nn.Module):
    def __init__(self):
        super(sharedNetwork, self).__init__()
        # Shared FC
        self.sfc1 = nn.Linear(8, 16)
        self.sfc2 = nn.Linear(16, 32)
        self.sfc3 = nn.Linear(32, 32)

        # Actor FC
        self.actorfc1 = nn.Linear(32, 16)
        self.actorfc2 = nn.Linear(16, 4)
        # Critic FC
        self.criticfc1 = nn.Linear(32, 8)
        self.criticfc2 = nn.Linear(8, 1)

        self.relu = nn.LeakyReLU()
    
    def forward1(self, input):
        x = self.relu(self.sfc1(input))
        x = self.relu(self.sfc2(x))
        x = self.relu(self.sfc3(x))
        x = self.relu(self.actorfc1(x))
        return F.softmax(self.actorfc2(x), dim=-1)

    def forward2(self, input):
        x = self.relu(self.sfc1(input))
        x = self.relu(self.sfc2(x))
        x = self.relu(self.sfc3(x))
        x = self.relu(self.criticfc1(x))
        return self.criticfc2(x)

class actorCritic():
    def __init__(self, network):
        self.network = network
        paramset_actor = list(self.network.sfc1.parameters()) + list(self.network.sfc2.parameters()) + list(self.network.sfc1.parameters()) + list(self.network.actorfc1.parameters()) + list(self.network.actorfc2.parameters())
        paramset_critic = list(self.network.sfc1.parameters()) + list(self.network.sfc2.parameters()) + list(self.network.sfc1.parameters()) + list(self.network.criticfc1.parameters()) + list(self.network.criticfc2.parameters())
        self.optimizer_actor = optim.SGD(paramset_actor, lr=0.001)
        self.optimizer_critic = optim.SGD(paramset_critic, lr=0.001)
        
    def sample(self, state): # Choose action
        action_prob = self.network.forward1(torch.FloatTensor(state).to(device))
        action_dist = Categorical(action_prob)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        return action.item(), log_prob

    def learn(self, state, state_next, rewards, log_probs, gamma):
        # Learn critic
        value = self.network.forward2(state)
        value_next = self.network.forward2(state_next)
        td = rewards + gamma * value_next - value
        loss_critic = torch.square(td).sum()
        self.optimizer_critic.zero_grad()
        loss_critic.backward(retain_graph=True)
        # Learn actor
        loss_actor = -log_probs * rewards
        self.optimizer_actor.zero_grad()
        loss_actor.backward()
        
        self.optimizer_critic.step()
        self.optimizer_actor.step()


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
   
    network = sharedNetwork().to(device)
    agent = actorCritic(network)