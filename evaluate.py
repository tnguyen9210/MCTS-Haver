
import numpy as np
import random

from tqdm import tqdm


def greedy_policy(action_means):
    action = np.argmax(action_means)
    return action


def evaluate(env, Q_table, num_episodes_eval, ep_max_steps):
    
    ep_reward_ary = []
    for i_ep in tqdm(range(num_episodes_eval)):
        np.random.seed(1000+i_ep)
        random.seed(1000+i_ep)
        state, info = env.reset(seed=1000+i_ep)
        # state = f"{state}"
            
        ep_reward = 0
        for i_step in range(ep_max_steps):
            # print(f"\n-> i_step={i_step}")
            action = greedy_policy(Q_table[state])
            next_state, reward, terminated, truncated, info = env.step(action)
            # print(f"state, action, next_state, terminated = {state, action, next_state, terminated}")
            # new_state = f"{new_state}"
            
            ep_reward += reward

            if terminated:
                break

            state = next_state

        ep_reward_ary.append(ep_reward)

    #     if (i_ep+1) % 1 == 0:
    #         print(f"ep={i_ep+1}, reward={ep_reward:0.4f}, avg_reward = {np.sum(ep_reward_ary)/(i_ep+1):0.4f}")
            
    # print(f"avg_reward = {np.sum(ep_reward_ary)/num_episodes_eval:0.4f}")

    return ep_reward_ary    
