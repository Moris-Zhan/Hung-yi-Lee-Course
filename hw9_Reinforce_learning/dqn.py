import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random

device = "cuda" if torch.cuda.is_available() else "cpu"

class Action(nn.Module):
  def __init__(self):
    super().__init__()
    self.fc1 = nn.Linear(8, 64)
    self.fc2 = nn.Linear(64, 64)
    self.fc3 = nn.Linear(64, 4)

  def forward(self, state):
      hid = torch.tanh(self.fc1(state))
      hid = torch.tanh(self.fc2(hid))
      return F.softmax(self.fc3(hid), dim=-1)

class Critic(nn.Module):
  def __init__(self):
    super().__init__()
    self.fc1 = nn.Linear(8, 64)
    self.fc2 = nn.Linear(64, 64)
    self.fc3 = nn.Linear(64, 4)

  def forward(self, state):
      hid = torch.tanh(self.fc1(state))
      hid = torch.tanh(self.fc2(hid))
      return F.softmax(self.fc3(hid), dim=-1)


class DQN():
    def __init__(self,action,critic):
        self.action_net = action
        self.critic_net = critic
        self.optimizer = optim.Adam(self.action_net.parameters(), lr=5e-4)
        self.critic_net.load_state_dict(self.action_net.state_dict()) # 加载action_net的行为
        self.critic_net.eval() # 模型验证

    def forward(self, state):
        return self.action_net(state)

    def learn(self, state_action_values, expected_state_action_values,batch):
        loss = torch.zeros(1).to(device)
        for i in range(len(state_action_values)):
            loss += F.smooth_l1_loss(state_action_values[i], expected_state_action_values[i])
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # 每10个batch更新一次target_net
        if batch%10 == 9:
            self.critic_net.load_state_dict(self.action_net.state_dict())

    def sample(self, state):
        r = torch.randn(1)
        if r > 0.2:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                return self.action_net(state).argmax().item()
        else:
            return torch.tensor([[random.randrange(4)]], device=device, dtype=torch.long).item()


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    action = Action().to(device)
    critic = Critic().to(device)
    agent = DQN(action, critic)