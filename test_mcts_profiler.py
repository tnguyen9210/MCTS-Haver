
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

from mcts_haver_profiler import run_mcts_trial
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

np.random.seed(0)
random.seed(0)

# params
args = parse_args()
args["update_method"] = "haver"
args["rollout_method"] = ""
args["render_mode"] = ""
args["action_multi"] = 1
args["num_trials"] = 2

m = args["num_trials"]
random_seeds = np.loadtxt("random_seeds.txt").astype("int64")
env_seeds = random_seeds[:m]
simulator_seeds = random_seeds[m:2*m]
mcts_seeds = random_seeds[2*m:]

#
env_id = "FrozenLake-v1"
args["map_name"] = '4x4'
env = FrozenLakeCustom(
    map_name=args["map_name"], 
    is_state_slippery=args["is_state_slippery"],
    is_slippery=args["is_slippery"], slippery_mode=args["slippery_mode"], 
    render_mode=args["render_mode"])

simulator = FrozenLakeSimulator(env.P, simulator_seed=0)

V_vit, Q_vit = value_iteration(
    simulator, args["gamma"], args["vit_thres"])
# global Q_vit_g = Q_vit

args["update_method"] = "haver"
args["rollout_method"] = ""

args["hparam_ucb_scale"] = 64
args["hparam_haver_var"] = 1

ep_reward_ary = []
Q_mcts_avg = defaultdict(lambda: np.zeros(simulator.num_actions))
start_time = time.time()
for i_trial in range(args["num_trials"]):
    # run mcts trial
    Q_mcts, ep_reward = run_mcts_trial(env, simulator, Q_vit, i_trial, 123, 123, args)

    # aggregate results
    ep_reward_ary.append(ep_reward)
    # for s in range(simulator.num_states):
    #     Q_mcts_avg[s] = (1-1/(i_trial+1))*Q_mcts_avg[s] + 1/(i_trial+1)*Q_mcts[s]
    
    end_time = time.time()
    # if (i_trial+1) % 10 == 0:
    #     print(f"ep={i_trial+1}, reward={ep_reward:0.4f}, avg_reward={np.sum(ep_reward_ary)/(i_trial+1):0.4f}, run_time={(end_time-start_time)/(i_trial+1):0.4f}")

# num_trials = args["num_trials"]
# print(f"avg_reward = {np.sum(ep_reward_ary)/num_trials:0.4f}")

