
from collections import defaultdict
import numpy as np
import random

import time

import gym
from env import FrozenLakeCustom, FrozenLakeSimulator

from mcts_haver import MCTS
from value_iteration import value_iteration

import logging
logging.basicConfig(level=logging.WARNING)


def main():
    np.random.seed(0)
    random.seed(0)
    
    
    env_id = "FrozenLake-v1"
    env = FrozenLakeCustom(map_name="4x4", is_slippery=True, render_mode=None)
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
    num_trial_episodes = 100
    ep_max_steps = 100
    
    mcts_max_iterations = 2000
    mcts_max_depth = 3
    mcts_rollout_max_depth = 200

    hparam_ucb_scale = 1.0
    hparam_haver_var = 1.0

    rollout_method = "vit"

    value_iteration_thres = 0.00001
    
    # get Q table from value iteration algo
    V_table, Q_table = value_iteration(
        simulator, num_actions, num_states, gamma, value_iteration_thres)
    
    # run mcts trials
    update_method = "avg"
    # hparam_ucb_scale_ary = [10, 15, 20, 50, 1]
    # for hparam_ucb_scale in hparam_ucb_scale_ary:
    #     print(f"\n-> hparam_ucb_scale = {hparam_ucb_scale}")
    #     run_mcts_trials(
    #         env, simulator, num_trial_episodes, ep_max_steps, gamma, action_multi,
    #         mcts_max_iterations, mcts_max_depth, mcts_rollout_max_depth,
    #         hparam_ucb_scale, hparam_haver_var, update_method,
    #         rollout_method, Q_table)
        

    hparam_ucb_scale = 50
    
    run_mcts_trials(
        env, simulator, num_trial_episodes, ep_max_steps, gamma, action_multi,
        mcts_max_iterations, mcts_max_depth, mcts_rollout_max_depth,
        hparam_ucb_scale, hparam_haver_var, update_method,
        rollout_method, Q_table)

    
def run_mcts_trials(
        env, simulator, num_trial_episodes, ep_max_steps, gamma, action_multi,
        mcts_max_iterations, mcts_max_depth, mcts_rollout_max_depth,
        hparam_ucb_scale, hparam_haver_var, update_method,
        rollout_method, Q_table):
    
    # run trials
    ep_reward_ary = []
    start_time = time.time()
    for i_ep in range(num_trial_episodes):
        np.random.seed(1000+i_ep)
        random.seed(1000+i_ep)
        state, info = env.reset(seed=1000+i_ep)
        # state = f"{state}"

        mcts = MCTS(simulator, gamma, action_multi,
                    mcts_max_iterations, mcts_max_depth, mcts_rollout_max_depth,
                    hparam_ucb_scale, hparam_haver_var, update_method,
                    rollout_method, Q_table)

        ep_reward = 0
        for i_step in range(ep_max_steps):
            # logging.warning(f"\n-> i_step={i_step}")
            action = mcts.run(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            # logging.warning(f"state, action, next_state, terminated = {state, action, next_state, terminated}")

            if terminated:
                break

            state = next_state
            
        # logging.warning(f"i_step = {i_step}, eps_reward={ep_reward:0.2f}, {ep_reward/(i_step+1):0.2f}")
        # stop
        end_time = time.time()
        ep_reward_ary.append(ep_reward)
        if (i_ep+1) % 10 == 0:
            print(f"ep={i_ep+1}, reward={ep_reward:0.4f}, avg_reward = {np.sum(ep_reward_ary)/(i_ep+1):0.4f}, run_time={(end_time-start_time)/(i_ep+1):0.4f}")
            
    print(f"avg_reward = {np.sum(ep_reward_ary)/num_episodes_eval:0.4f}")
            
    reward_mean = np.mean(ep_reward_ary)
    reward_std = np.std(ep_reward_ary, ddof=1)
    print(f"reward = {reward_mean:.2f} +/- {reward_std:.2f}")

if __name__ == '__main__':
    main()
