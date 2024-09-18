
from collections import defaultdict
import numpy as np
import random

import time

import gym
from mcts import MCTS

import logging
logging.basicConfig(level=logging.WARNING)


class FrozenLakeSimulator:
    def __init__(self, trans_probs, num_actions):
        self.trans_probs = trans_probs
        self.num_actions = num_actions

    def step(self, state, action):
        transitions = self.trans_probs[state][action]
        trans_p = np.array([t[0] for t in transitions])
        # logging.info(f"action_probs = {action_probs}")
        idx = np.random.choice(len(trans_p), 1, p=trans_p)[0]
        p, next_state, reward, terminated = transitions[idx]
        # logging.info(f"state, action, next_state, terminated = {state, action, next_state, terminated}")
        return (int(next_state), reward, terminated, False, {"prob": p})

def main():
    np.random.seed(0)
    random.seed(0)
    
    # params
    env_id = "FrozenLake-v1"
    gamma = 0.95
    num_episodes_train = 100
    num_episodes_eval = 50
    eps_max_steps = 100
    
    mcts_max_steps = 2000
    mcts_max_depth = 3
    mcts_rollout_max_depth = 100

    # create gym env
    env = gym.make(
        env_id, map_name="4x4", is_slippery=False)
    
    num_states = env.observation_space.n
    num_actions = env.action_space.n
    logging.info(f"num_states = {num_states}")
    logging.info(f"sample_state = {env.observation_space.sample()}")
    logging.info(f"num_actions = {num_actions}")
    logging.info(f"sample_action = {env.action_space.sample()}")
    # logging.info(f"trans_probs = {env.P}")

    simulator = FrozenLakeSimulator(env.P, num_actions)
    # state, info = env.reset()
    # logging.info(f"state = {state}")
    # actions_ary = [2, 0, 1, 1, 2, 2, 3, 1, 1, 2]
    # for action in actions_ary:
    #     new_state, reward, terminated, truncated, info = simulator.step(state, action)
    #     logging.info(f"state, action, new_state, terminated = {state, action, new_state, terminated}")
    #     state = new_state
    
    # run q_learning
    all_ep_reward = 0
    start_time = time.time()
    for i_ep in range(num_episodes_train):
        state, info = env.reset()
        # state = f"{state}"

        mcts = MCTS(simulator, num_actions, gamma,
                    mcts_max_steps, mcts_max_depth, mcts_rollout_max_depth)

        ep_reward = 0
        for i_step in range(eps_max_steps):
            action = mcts.run(int(state))
            next_state, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward

            if terminated or truncated:
                break

            state = next_state
            
        # logging.warn(f"i_step = {i_step}, eps_reward={ep_reward:0.2f}, {ep_reward/(i_step+1):0.2f}")

        end_time = time.time()
        all_ep_reward += ep_reward
        if (i_ep+1) % 10 == 0:
            print(f"ep={i_ep+1}, avg_reward = {all_ep_reward/(i_ep+1):0.4f}, run_time={(end_time-start_time)/(i_ep+1):0.4f}")
        
    print(f"avg_reward = {all_ep_reward/num_episodes_train:0.4f}")
    # stats = mcts_learning(
    #     env, num_episodes_train, eps_max_steps, mcts_max_steps, tdqm_disable=False)
    # print(stats)

    # # evaluate
    # episode_reward_ary = evaluate(env, Q_table, num_episodes_eval, max_steps)
    # reward_mean = np.mean(episode_reward_ary)
    # reward_std = np.std(episode_reward_ary)
    # print(f"reward = {reward_mean:.2f} +/- {reward_std:.2f}")

    # env = gym.make(
    #     env_id, map_name="4x4", is_slippery=True, render_mode="human")
    # episode_reward_ary = evaluate(env, Q_table, num_episodes_eval, max_steps)
    # reward_mean = np.mean(episode_reward_ary)
    # reward_std = np.std(episode_reward_ary)
    # print(f"reward = {reward_mean:.2f} +/- {reward_std:.2f}")

if __name__ == '__main__':
    main()
