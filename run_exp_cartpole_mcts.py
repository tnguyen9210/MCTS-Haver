
from collections import defaultdict
import numpy as np

import gym

from mcts import mcts_learning

def main():
    # params
    env_id = "FrozenLake-v1"
    eps_max_steps = 100
    mcts_max_steps = 100
    gamma = 0.95
    num_episodes_train = 200
    num_episodes_eval = 50

    # create gym env
    env = gym.make(
        env_id, map_name="4x4", is_slippery=False)
    
    num_states = env.observation_space.n
    num_actions = env.action_space.n
    print(f"num_states = {num_states}")
    print(f"sample_state = {env.observation_space.sample()}")
    print(f"num_actions = {num_actions}")
    print(f"sample_action = {env.action_space.sample()}")
    
    # run q_learning
    stats = mcts_learning(
        env, num_episodes_train, eps_max_steps, mcts_max_steps, tdqm_disable=False)
    print(stats)

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
