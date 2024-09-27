
from collections import defaultdict
import numpy as np
import random

import time

import gym
from env import FrozenLakeCustom, FrozenLakeSimulator

from mcts_haver import MCTS
from value_iteration import value_iteration

import logging
logging.basicConfig(level=logging.ERROR)


def main():
    np.random.seed(0)
    random.seed(0)
    
    
    env_id = "FrozenLake-v1"
    env = FrozenLakeCustom(map_name="4x4", is_slippery=False, render_mode=None)
    simulator = FrozenLakeSimulator(env.P)
    
    num_states = env.observation_space.n
    num_actions = env.action_space.n
    logging.info(f"num_states = {num_states}")
    logging.info(f"sample_state = {env.observation_space.sample()}")
    logging.info(f"num_actions = {num_actions}")
    logging.info(f"sample_action = {env.action_space.sample()}")
    # logging.info(f"trans_probs = {env.P}")

    # params
    action_multi = 1
    gamma = 0.95
    num_trial_episodes = 20
    ep_max_steps = 100
    
    mcts_max_iterations = 2000
    mcts_max_depth = 3
    mcts_rollout_max_depth = 100

    hparam_ucb_scale = 1.0
    hparam_haver_var = 1.0

    rollout_method = "vit"

    value_iteration_thres = 0.00001
    
    # get Q table from value iteration algo
    V_table, Q_table = value_iteration(
        simulator, num_actions, num_states, gamma, value_iteration_thres)

    for state in range(num_states):
        logging.warning(f"\n-> state = {state}")
        logging.warning(f"V[state] = {V_table[state]:0.4f}")
        for action in range(num_actions):
            logging.warning(f"Q[state][action] = {Q_table[state][action]:0.4f}")
        logging.warning(f"best_action={np.argmax(Q_table[state])}")

    
    # run mcts trials
    update_method = "haver"
    # hparam_ucb_scale_ary = [30, 40, 50, 60, 100]
    # hparam_ucb_scale_ary = np.arange(26, 40, 2)
    # for hparam_ucb_scale in hparam_ucb_scale_ary:
    #     print(f"\n-> hparam_ucb_scale = {hparam_ucb_scale}")
    #     run_mcts_trials(
    #         env, simulator, num_trial_episodes, ep_max_steps, gamma, action_multi,
    #         mcts_max_iterations, mcts_max_depth, mcts_rollout_max_depth,
    #         hparam_ucb_scale, hparam_haver_var, update_method,
    #         rollout_method, Q_table)
        

    hparam_ucb_scale = 30
    
    run_mcts_trials(
        env, simulator, num_trial_episodes, ep_max_steps, gamma, action_multi,
        mcts_max_iterations, mcts_max_depth, mcts_rollout_max_depth,
        hparam_ucb_scale, hparam_haver_var, update_method,
        rollout_method, Q_table)

    

    
    # reward_mean = np.mean(ep_reward_ary)
    # reward_std = np.std(ep_reward_ary, ddof=1)
    # print(f"reward = {reward_mean:.2f} +/- {reward_std:.2f}")

if __name__ == '__main__':
    main()
