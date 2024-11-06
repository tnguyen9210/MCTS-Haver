
import cProfile
import pstats

from collections import defaultdict
import random
import numpy as np
np.set_printoptions(precision=4, suppress=True)

import time
import copy 
import multiprocess as mp

import gym
from env import FrozenLakeCustom, FrozenLakeSimulator

from mcts_haver import run_mcts_trial
from value_iteration import value_iteration

from config import parse_args

import logging
# logger = logging.getLogger()
# logger.setLevel(logging.FATAL)
# logging.basicConfig(level=logging.DEBUG)
# logging.basicConfig(level=logging.WARNING)
logging.basicConfig(level=logging.FATAL)

# import ipdb

# ipdb.set_trace()

def main():
    np.random.seed(0)
    random.seed(0)

    # params
    args = parse_args()
    args["update_method"] = "haver"
    args["rollout_method"] = ""
    args["render_mode"] = ""
    args["action_multi"] = 1

    #
    env_id = "FrozenLake-v1"
    args["map_name"] = '4x4'
    env = FrozenLakeCustom(
        map_name=args["map_name"], 
        is_state_slippery=args["is_state_slippery"],
        is_slippery=args["is_slippery"], slippery_mode=args["slippery_mode"], 
        render_mode=args["render_mode"])

    simulator = FrozenLakeSimulator(env.P)

    V_vit, Q_vit = value_iteration(
        simulator, args["gamma"], args["vit_thres"])
    # global Q_vit_g = Q_vit

    args["hparam_ucb_scale"] = 64
    args["hparam_haver_var"] = 1

    ep_reward_ary = []
    Q_mcts_avg = defaultdict(lambda: np.zeros(simulator.num_actions))
    start_time = time.time()
    for i_trial in range(args["num_trials"]):
        # run mcts trial
        Q_mcts, ep_reward =  run_mcts_trial(env, simulator, Q_vit, i_trial, args)

        # aggregate results
        ep_reward_ary.append(ep_reward)

        end_time = time.time()
        if (i_trial+1) % 10 == 0:
            print(f"ep={i_trial+1}, reward={ep_reward:0.4f}, avg_reward={np.sum(ep_reward_ary)/(i_trial+1):0.4f}, run_time={(end_time-start_time)/(i_trial+1):0.4f}")

    num_trials = args["num_trials"]
    print(f"avg_reward = {np.sum(ep_reward_ary)/num_trials:0.4f}")

    
if __name__ == '__main__':
    with cProfile.Profile() as profile:
        main()

    results = pstats.Stats(profile)
    results.sort_stats(pstats.SortKey.TIME)
    results.print_stats()
