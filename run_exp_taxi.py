
from collections import defaultdict
import numpy as np

import gym

from algos import *

def main():
    # params
    env_id = "Taxi-v3"
    max_steps = 99
    gamma = 0.95
    num_episodes_train = 1000
    num_episodes_eval = 100

    args = dict()
    args["lr"] = 0.7
    lr_sched_type = "fixed"
    lr_sched_fn = create_lr_sched_fn(lr_sched_type, args)

    max_eps = 1.0
    min_eps = 0.05
    decay_rate = 0.0005
    eps_sched_type = "exp"
    eps_sched_fn = create_eps_sched_fn(eps_sched_type, min_eps, max_eps, decay_rate)
    
    # create gym env
    env = gym.make(env_id)
    
    num_states = env.observation_space.n
    num_actions = env.action_space.n
    print(f"num_states = {num_states}")
    print(f"sample_state = {env.observation_space.sample()}")
    print(f"num_actions = {num_actions}")
    print(f"sample_action = {env.action_space.sample()}")
    
    # init Qtable
    Q_table = defaultdict(lambda: np.zeros(num_actions))
    Q_nvisits = defaultdict(lambda: np.zeros(num_actions))
    
    # run q_learning
    Q_table, stats = q_learning(
        env, Q_table, Q_nvisits, num_episodes_train, max_steps,
        gamma, lr_sched_fn, eps_sched_fn, tdqm_disable=True)

    # evaluate
    episode_reward_ary = evaluate(env, Q_table, num_episodes_eval, max_steps)
    reward_mean = np.mean(episode_reward_ary)
    reward_std = np.std(episode_reward_ary)
    print(f"reward = {reward_mean:.2f} +/- {reward_std:.2f}")

    env = gym.make(env_id, render_mode="human")
    episode_reward_ary = evaluate(env, Q_table, 5, 50)

if __name__ == '__main__':
    main()
