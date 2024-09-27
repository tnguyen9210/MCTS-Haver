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
# logger.setLevel(logging.WARNING)
logging.basicConfig(level=logging.WARNING)

np.random.seed(0)
random.seed(0)

# params
args = parse_args()
args["update_method"] = "haver"
args["rollout_method"] = ""


#
env_id = "FrozenLake-v1"
env = FrozenLakeCustom(
    map_name=args["map_name"], is_slippery=args["is_slippery"],
    render_mode=args["render_mode"])

simulator = FrozenLakeSimulator(env.P)

V_vit, Q_vit = value_iteration(
    simulator, args["gamma"], args["vit_thres"])
# global Q_vit_g = Q_vit
        
# for state in range(simulator.num_states):
#     logging.warning(f"\n-> state = {state}")
#     logging.warning(f"V[state] = {V_vit[state]:0.4f}")
#     for action in range(simulator.num_actions):
#         logging.warning(f"Q[state][action] = {Q_vit[state][action]:0.4f}")
#     logging.warning(f"best_action={np.argmax(Q_vit[state])}")
    
manager = mp.Manager()
ep_reward_list = manager.list()
Q_mcts_list = manager.list()

def run_trial(i_trial, Q_vit, args):

    random.seed(10000+i_trial)
    np.random.seed(10000+i_trial)

#     env = FrozenLakeCustom(
#         map_name=args["map_name"], is_slippery=args["is_slippery"],
#         render_mode=args["render_mode"])

#     simulator = FrozenLakeSimulator(env.P)

    Q_mcts, ep_reward = run_mcts_trial(env, simulator, Q_vit, i_trial, args)

    ep_reward_list.append(ep_reward)
    Q_mcts_list.append(Q_mcts)


Q_mcts_dict = defaultdict()

hparam_ucb_scale_ary = np.arange(26, 40, 2)
hparam_ucb_scale_ary = [30]
best_param = None
max_reward_mean = -np.inf
for hparam_ucb_scale in hparam_ucb_scale_ary:
    start_time = time.time()
    
    print(f"\n-> hparam_ucb_scale = {hparam_ucb_scale}")
    args["hparam_ucb_scale"] = hparam_ucb_scale
    
    pool = mp.Pool()
    pool.starmap(run_trial, [(i, Q_vit, args) for i in range(args["num_trials"])])

    reward_mean = np.mean(ep_reward_list)
    reward_std = np.std(ep_reward_list, ddof=1)
    print(f"reward = {reward_mean:.2f} +/- {reward_std:.2f}")
    
    Q_mcts_dict[f"{hparam_ucb_scale}"] = copy.deepcopy(Q_mcts_list)
    
    if reward_mean > max_reward_mean:
        max_reward_mean = reward_mean 
        best_param = hparam_ucb_scale
    
    ep_reward_list[:] = []
    Q_mcts_list[:] = []
    
    end_time = time.time()
    print(f"it takes {end_time-start_time:0.4f}")
