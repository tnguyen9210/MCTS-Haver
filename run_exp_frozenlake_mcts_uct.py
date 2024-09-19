
from collections import defaultdict
import numpy as np
import random

import time

import gym
from mcts import MCTS
from utils import FrozenLakeSimulator

import logging
logging.basicConfig(level=logging.WARNING)


def main():
    np.random.seed(0)
    random.seed(0)
    
    # params
    env_id = "FrozenLake-v1"
    action_multi = 1
    gamma = 0.95
    num_episodes_train = 100
    num_episodes_eval = 50
    eps_max_steps = 100
    
    mcts_max_iterations = 3000
    mcts_max_depth = 3
    mcts_rollout_max_depth = 100

    ucb_param = 1.0

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

        mcts = MCTS(simulator, num_actions, gamma, action_multi,
                    mcts_max_iterations, mcts_max_depth, mcts_rollout_max_depth,
                    ucb_param)

        ep_reward = 0
        for i_step in range(eps_max_steps):
            action = mcts.run(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward

            if terminated:
                break

            state = next_state
            
        # logging.warn(f"i_step = {i_step}, eps_reward={ep_reward:0.2f}, {ep_reward/(i_step+1):0.2f}")

        end_time = time.time()
        all_ep_reward += ep_reward
        if (i_ep+1) % 10 == 0:
            print(f"ep={i_ep+1}, avg_reward = {all_ep_reward/(i_ep+1):0.4f}, run_time={(end_time-start_time)/(i_ep+1):0.4f}")
        
    print(f"avg_reward = {all_ep_reward/num_episodes_train:0.4f}")

if __name__ == '__main__':
    main()
