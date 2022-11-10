from tqdm.notebook import tqdm
import numpy as np
import torch
import matplotlib.pyplot as plt

from IPython import display

device = "cuda" if torch.cuda.is_available() else "cpu"

def train(env, agent, avg_total_rewards, avg_final_rewards, EPISODE_PER_BATCH, NUM_BATCH, GAMMA):
    agent.network.train()  # 訓練前，先確保 network 處在 training 模式
    prg_bar = tqdm(range(NUM_BATCH))
    for batch in prg_bar:

        log_probs, rewards, advantages = [], [], []
        total_rewards, final_rewards = [], []
        
        # collect trajectory 蒐集訓練資料
        for episode in range(EPISODE_PER_BATCH):
            
            state = env.reset()
            total_reward, total_step = 0, 0
            seq_rewards = []
            while True:

                action, log_prob = agent.sample(state) # at, log(at|st)
                next_state, reward, done, _ = env.step(action)

                log_probs.append(log_prob) # [log(a1|s1), log(a2|s2), ...., log(at|st)]
                # seq_rewards.append(reward)
                state = next_state # s_t
                total_reward += reward
                total_step += 1
                rewards.append(reward) # change here
                # ! IMPORTANT !
                # Current reward implementation: immediate reward,  given action_list : a1, a2, a3 ......
                #                                                         rewards :     r1, r2 ,r3 ......
                # medium：change "rewards" to accumulative decaying reward, given action_list : a1,                           a2,                           a3, ......
                #                                                           rewards :           r1+0.99*r2+0.99^2*r3+......, r2+0.99*r3+0.99^2*r4+...... ,  r3+0.99*r4+0.99^2*r5+ ......         
                for i in range(len(rewards) - 1):
                    rewards[i] += pow(GAMMA, len(rewards) - 1 - i) * rewards[-1]
                total_reward = sum(rewards)
                # boss : implement DQN
                # agent.learn_critic(torch.Tensor(state), torch.Tensor(next_state), reward, gamma)
                # agent.learn_actor(log_prob, reward)
                agent.learn(torch.Tensor(state).to(device), torch.Tensor(next_state).to(device), reward, log_prob, GAMMA)
                state = next_state               
                if done:
                    final_rewards.append(reward)
                    total_rewards.append(total_reward)                      
                    break

        # print(f"rewards looks like ", np.shape(rewards))  
        #print(f"log_probs looks like ", np.shape(log_probs))     
        # 紀錄訓練過程
        avg_total_reward = sum(total_rewards) / len(total_rewards)
        avg_final_reward = sum(final_rewards) / len(final_rewards)
        avg_total_rewards.append(avg_total_reward)
        avg_final_rewards.append(avg_final_reward)
        prg_bar.set_description(f"Total: {avg_total_reward: 4.1f}, Final: {avg_final_reward: 4.1f}")

        # # 更新網路
        # # rewards = np.concatenate(rewards, axis=0)
        # rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-9)  # normalize the reward 
        # agent.learn(torch.stack(log_probs), torch.from_numpy(rewards))
        # # print("logs prob looks like ", torch.stack(log_probs).size())
        # # print("torch.from_numpy(rewards) looks like ", torch.from_numpy(rewards).size())

def test(env, agent, NUM_OF_TEST):      
    # fix(env, seed)
    agent.network.eval()    # 測試前先將 network 切換為 evaluation 模式
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
                action, _ = agent.sample(state)
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
  