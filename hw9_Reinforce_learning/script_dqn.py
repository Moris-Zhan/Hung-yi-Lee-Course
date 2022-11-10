from tqdm.notebook import tqdm
import numpy as np
import torch
import matplotlib.pyplot as plt

from IPython import display

device = "cuda" if torch.cuda.is_available() else "cpu"

def train(env, agent, avg_total_rewards, avg_final_rewards, EPISODE_PER_BATCH, NUM_BATCH, GAMMA):
    agent.action_net.train()  # 訓練前，先確保 network 處在 training 模式
    prg_bar = tqdm(range(NUM_BATCH))
    for batch in prg_bar:
        state_action_values, expected_state_action_values = [], []
        total_rewards, final_rewards = [], []
        # 收集訓練資料
        for episode in range(EPISODE_PER_BATCH):
            state = env.reset()
            total_reward = 0
            while True:
                action = agent.sample(state)
                next_state, reward, done, _ = env.step(action)
                state_action_value = agent.action_net(torch.FloatTensor(state).to(device))[action]
                state_action_values.append(state_action_value.to(device))
                state = next_state
                #Double DQN
                if done:
                    next_state_value = 0
                else:
                    next_state_value = agent.critic_net(torch.FloatTensor(state).to(device))[agent.action_net(torch.FloatTensor(state).to(device)).argmax().item()].detach()
                # 使用Policy Gradient
                expected_state_action_value = (next_state_value * GAMMA) + reward
                expected_state_action_values.append(torch.FloatTensor([expected_state_action_value]).to(device))
                total_reward += reward
                if done:
                    final_rewards.append(reward)
                    total_rewards.append(total_reward)
                    break

        # 記錄訓練過程
        avg_total_reward = sum(total_rewards) / len(total_rewards)
        avg_final_reward = sum(final_rewards) / len(final_rewards)
        avg_total_rewards.append(avg_total_reward)
        avg_final_rewards.append(avg_final_reward)
        prg_bar.set_description(f"Total: {avg_total_reward: 4.1f}, Final: {avg_final_reward: 4.1f}")
        # print(expected_state_action_values)
        # 更新網絡
        agent.learn(state_action_values, expected_state_action_values, batch)



def test(env, agent, NUM_OF_TEST):      
    # fix(env, seed)
    agent.action_net.eval()    # 測試前先將 network 切換為 evaluation 模式
    # NUM_OF_TEST = 5 # Do not revise this !!!
    test_total_reward = []
    action_list = []
    for i in range(NUM_OF_TEST):
        actions = []
        state = env.reset()

        img = plt.imshow(env.render(mode='rgb_array'))

        total_reward = 0

        done = False
        while not done:
                action = agent.sample(state)
                actions.append(action)
                state, reward, done, _ = env.step(action)

                total_reward += reward

                img.set_data(env.render(mode='rgb_array'))
                display.display(plt.gcf())
                display.clear_output(wait=True)
                
        print(total_reward)
        test_total_reward.append(total_reward)

        action_list.append(actions) # save the result of testing 

    return test_total_reward, action_list
          
