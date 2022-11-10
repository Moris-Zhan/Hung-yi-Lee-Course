
import gym
from utils import fix
import torch
import numpy as np
import matplotlib.pyplot as plt

seed = 543
env = gym.make('LunarLander-v2')
fix(env, seed) # fix the environment Do not revise this !!!

EPISODE_PER_BATCH = 5  # 每蒐集 5 個 episodes 更新一次 agent
NUM_BATCH = 1000       # 總共更新 400 次
GAMMA = 0.99           # Dacay rate
NUM_OF_TEST = 5 # Do not revise this !!!


avg_total_rewards, avg_final_rewards = [], []

device = "cuda" if torch.cuda.is_available() else "cpu"   

# Medium
from script_actor_critic import train, test
from model import PolicyGradientNetwork, PolicyGradientAgent
network = PolicyGradientNetwork().to(device)
agent = PolicyGradientAgent(network)

# Actor Critic
from script_actor_critic import train, test
from actor_critic import sharedNetwork, actorCritic
network = sharedNetwork().to(device)
agent = actorCritic(network)

# Actor Critic
from script_dqn import train, test
from dqn import Action, Critic, DQN
action = Action().to(device)
critic = Critic().to(device)
agent = DQN(action, critic)


train(env, agent, avg_total_rewards, avg_final_rewards, EPISODE_PER_BATCH, NUM_BATCH, GAMMA)
test_total_reward, action_list = test(env, agent, NUM_OF_TEST)

# Saving the result of Model Testing
PATH = "Action_List.npy" # Can be modified into the name or path you want
np.save(PATH ,np.array(action_list)) 

# Server
action_list = np.load(PATH,allow_pickle=True) # The action list you upload
seed = 543 # Do not revise this
fix(env, seed)

# agent.network.eval()    # set network to evaluation mode
agent.action_net.eval()    # set network to evaluation mode (DQN)

test_total_reward = []
if len(action_list) != 5:
    print("Wrong format of file !!!")
    exit(0)
for actions in action_list:
    state = env.reset()
    img = plt.imshow(env.render(mode='rgb_array'))

    total_reward = 0

    done = False

    for action in actions:
    
            state, reward, done, _ = env.step(action)
            total_reward += reward
            if done:
                break

    print(f"Your reward is : %.2f"%total_reward)
    test_total_reward.append(total_reward)

print(f"Your final reward is : %.2f"%np.mean(test_total_reward))    