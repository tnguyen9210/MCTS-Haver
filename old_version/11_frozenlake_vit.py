
from collections import defaultdict
import numpy as np
import random
np.set_printoptions(precision=4, suppress=True)

import time

import gym
from env import FrozenLakeCustom, FrozenLakeSimulator

from value_iteration import value_iteration
from evaluate import evaluate

import logging
logging.basicConfig(level=logging.WARNING)


def main():
    np.random.seed(0)
    random.seed(0)
    
    # params
    env_id = "FrozenLake-v1"
    action_multi = 1
    gamma = 0.95
    
    num_episodes_eval = 20
    ep_max_steps = 100

    threshold = 0.00001

    # create gym env
    env = FrozenLakeCustom(map_name="4x4", is_slippery=False, render_mode=None)
    simulator = FrozenLakeSimulator(env.P)
    
    num_states = env.observation_space.n
    num_actions = env.action_space.n
    logging.info(f"num_states = {num_states}")
    logging.info(f"sample_state = {env.observation_space.sample()}")
    logging.info(f"num_actions = {num_actions}")
    logging.info(f"sample_action = {env.action_space.sample()}")
    # logging.info(f"trans_probs = {env.P}")

    # run value iteration
    V_table, Q_table = value_iteration(simulator, num_actions, num_states, gamma, threshold)

    for state in range(num_states):
        print(f"\n-> state = {state}")
        print(f"V[state] = {V_table[state]:0.4f}")
        for action in range(num_actions):
            print(f"Q[state][action] = {Q_table[state][action]:0.4f}")
        print(f"best_action={np.argmax(Q_table[state])}")

    # evaluate
    ep_reward_ary = evaluate(env, Q_table, num_episodes_eval, ep_max_steps)
    reward_mean = np.mean(ep_reward_ary)
    reward_std = np.std(ep_reward_ary, ddof=1)
    print(f"reward = {reward_mean:.2f} +/- {reward_std:.2f}")

    # env = FrozenLakeCustom(map_name="4x4", is_slippery=True, render_mode="human")
    # episode_reward_ary = evaluate(env, Q_table, 3, 10)


if __name__ == '__main__':
    main()
