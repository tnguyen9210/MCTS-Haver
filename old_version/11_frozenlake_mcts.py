
from collections import defaultdict
import numpy as np
import random

import time

import gym
from env import FrozenLakeCustom, FrozenLakeSimulator

from mcts_haver import MCTS

import logging
logging.basicConfig(level=logging.WARNING)


def main():
    np.random.seed(0)
    random.seed(0)
    
    # params
    env_id = "FrozenLake-v1"
    action_multi = 1
    gamma = 0.95
    num_trial_episodes = 40
    ep_max_steps = 100
    
    mcts_max_iterations = 2000
    mcts_max_depth = 3
    mcts_rollout_max_depth = 100

    hparam_ucb_scale = 1.0
    hparam_haver_var = 1.0

    # create gym env
    # env = gym.make(
    #     env_id, map_name="4x4", is_slippery=False)
    env = FrozenLakeCustom(map_name="4x4", is_slippery=False, render_mode=None)
    simulator = FrozenLakeSimulator(env.P)
    
    num_states = env.observation_space.n
    num_actions = env.action_space.n
    logging.info(f"num_states = {num_states}")
    logging.info(f"sample_state = {env.observation_space.sample()}")
    logging.info(f"num_actions = {num_actions}")
    logging.info(f"sample_action = {env.action_space.sample()}")
    # logging.info(f"trans_probs = {env.P}")
    
    # run mcts trials
    update_method = ""
    # hparam_ucb_scale_ary = [7, 8, 10, 15, 20]
    # for hparam_ucb_scale in hparam_ucb_scale_ary:
    #     print(f"\n-> hparam_ucb_scale = {hparam_ucb_scale}")
    #     run_mcts_trials(
    #         env, simulator, num_trial_episodes, ep_max_steps, gamma, action_multi,
    #         mcts_max_iterations, mcts_max_depth, mcts_rollout_max_depth,
    #         hparam_ucb_scale, hparam_haver_var, update_method)

    hparam_ucb_scale = 15
    
    run_mcts_trials(
        env, simulator, num_trial_episodes, ep_max_steps, gamma, action_multi,
        mcts_max_iterations, mcts_max_depth, mcts_rollout_max_depth,
        hparam_ucb_scale, hparam_haver_var, update_method)

def run_mcts_trials(
        env, simulator, num_trial_episodes, ep_max_steps, gamma, action_multi,
        mcts_max_iterations, mcts_max_depth, mcts_rollout_max_depth,
        hparam_ucb_scale, hparam_haver_var, update_method):

    num_states = env.observation_space.n
    num_actions = env.action_space.n
    
    # run trials
    all_ep_reward = 0
    start_time = time.time()
    for i_ep in range(num_trial_episodes):
        state, info = env.reset()
        np.random.seed(1000+i_ep)
        random.seed(1000+i_ep)
        # state = f"{state}"

        mcts = MCTS(simulator, gamma, action_multi,
                    mcts_max_iterations, mcts_max_depth, mcts_rollout_max_depth,
                    hparam_ucb_scale, hparam_haver_var, update_method)

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
        all_ep_reward += ep_reward
        if (i_ep+1) % 10 == 0:
            print(f"ep={i_ep+1}, reward={ep_reward:0.4f}, avg_reward = {all_ep_reward/(i_ep+1):0.4f}, run_time={(end_time-start_time)/(i_ep+1):0.4f}")
        
    print(f"avg_reward = {all_ep_reward/num_trial_episodes:0.4f}")


if __name__ == '__main__':
    main()
