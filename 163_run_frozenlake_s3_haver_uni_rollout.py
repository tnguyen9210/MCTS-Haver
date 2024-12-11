
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

from mcts_haver_stochastic import run_mcts_trial
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
print(f"num_trials = {args['num_trials']}")

m = args["num_trials"]
random_seeds = np.loadtxt("random_seeds.txt").astype("int64")
env_seeds = random_seeds[:m]
simulator_seeds = random_seeds[m:2*m]
mcts_seeds = random_seeds[2*m:]

#
env_id = "FrozenLake-v1"
args["ep_max_steps"] = 20
args["map_name"] = "4x4X"
args["is_state_slippery"] = True
args["is_slippery"] = False
args["slippery_mode"] = "mild"
env = FrozenLakeCustom(
    map_name=args["map_name"], 
    is_state_slippery=args["is_state_slippery"],
    is_slippery=args["is_slippery"], slippery_mode=args["slippery_mode"], 
    render_mode=args["render_mode"])


simulator = FrozenLakeSimulator(env.P, simulator_seed=0)

V_vit, Q_vit = value_iteration(
    simulator, args["gamma"], args["vit_thres"])
# global Q_vit_g = Q_vit

manager = mp.Manager()
ep_reward_list = manager.list()
Q_mcts_list = manager.list()

def run_trial(i_trial, Q_vit, env_seed, simulator_seed, mcts_seed, args):

    # random.seed(random_seeds[i_trial])
    # np.random.seed(random_seeds[i_trial])

    env = FrozenLakeCustom(
        map_name=args["map_name"], 
        is_state_slippery=args["is_state_slippery"],
        is_slippery=args["is_slippery"], slippery_mode=args["slippery_mode"], 
        render_mode=args["render_mode"])

    simulator = FrozenLakeSimulator(env.P, simulator_seed)

    Q_mcts, ep_reward = run_mcts_trial(env, simulator, Q_vit, i_trial, env_seed, mcts_seed, args)

    ep_reward_list.append(ep_reward)
    Q_mcts_list.append(Q_mcts)
    return ep_reward

# hparam_ucb_scale_list = np.arange(10, 100, 10)
# hparam_ucb_scale_list = [32, 64, 128, 256, 512, 1024]
hparam_ucb_scale_list = [np.sqrt(100)**(i/2) for i in range(2,8)]
args["hparam_ucb_scale"] = 64

# hparam_haver_std_list = np.arange(10, 100, 10)
# hparam_haver_std_list = [0, 1/16, 1/8, 1/4, 1, 4, 8, 16]
hparam_haver_std_list = [0] + [np.sqrt(100)**(i/2) for i in range(-3,3)]

# num_trajectories_list = [200, 500, 1000, 1500, 2000, 2500, 3000]
# num_trajectories_list = [400, 600, 800]
num_trajectories_list = [int(np.sqrt(100)**(i/2)) for i in range(4,7)]

# num_trajectories_list = [2]

best_param_list = []
max_reward_mean_list = []
log_text = ""
res_text1 = ""
res_text2 = ""
for num_trajectories in num_trajectories_list:
    print(f"\n-> num_trajectories = {num_trajectories}")
    log_text += f"\n-> num_trajectories = {num_trajectories} \n"
    args["mcts_num_trajectories"] = num_trajectories
    
    best_param = None
    max_reward_mean = -np.inf
    start_time = time.time()
    res_text1 += f"{num_trajectories} "
    res_text2 += f"{num_trajectories} "
    for hparam_ucb_scale in hparam_ucb_scale_list: 
        
        args["hparam_ucb_scale"] = hparam_ucb_scale
        print(f"\n-> hparam_ucb_scale = {hparam_ucb_scale}")
        log_text += f"\n-> hparam_ucb_scale = {hparam_ucb_scale} \n"
        
        max_reward_mean = -np.inf
        best_param = None
        max_reward_error = None
        
        for hparam_haver_std in hparam_haver_std_list:
            # start_time = time.time()

            args["hparam_haver_var"] = hparam_haver_std**2
            # print(f"hparam_haver_var = {args['hparam_haver_var']}")
            # print(f"hparam_ucb_scale = {args['hparam_ucb_scale']}")

            pool = mp.Pool()
            pool.starmap(
                run_trial, 
                [(i, Q_vit, env_seeds[i], simulator_seeds[i], mcts_seeds[i], args) for i in range(args["num_trials"])])
            pool.close()
            
            reward_mean = np.mean(ep_reward_list)
            reward_std = np.std(ep_reward_list, ddof=1) if len(ep_reward_list) > 1 else 0
            reward_error = reward_std/np.sqrt(args["num_trials"])
            # if hparam_haver_std <= 8:
            #     res_text1 += f"& {reward_mean:0.2f} (\u00B1{reward_error:0.2f}) "
            # else:
            #     res_text2 += f"& {reward_mean:0.2f} (\u00B1{reward_error:0.2f}) "
            print(f"reward = {reward_mean:0.2f} \u00B1 {reward_error:0.2f}")
            log_text += f"reward = {reward_mean:0.2f} \u00B1 {reward_error:0.2f} \n"

            if reward_mean > max_reward_mean:
                max_reward_mean = reward_mean 
                max_reward_error = reward_error
                best_param = hparam_haver_std
                
            ep_reward_list[:] = []
            Q_mcts_list[:] = []

            end_time = time.time()
            # print(f"it takes {end_time-start_time:0.4f}")
        
        if hparam_ucb_scale <= 128:
            res_text1 += f"& {max_reward_mean:0.2f} (\u00B1{max_reward_error:0.2f}) "
        else:
            res_text2 += f"& {max_reward_mean:0.2f} (\u00B1{max_reward_error:0.2f}) "
            
        print(f"max_reward = {max_reward_mean:0.2f} \u00B1 {max_reward_error:0.2f}")
        print(f"best_param = {best_param}")
        log_text += f"max_reward = {max_reward_mean:0.2f} \u00B1 {max_reward_error:0.2f} \n"
        log_text += f"best_param = {best_param} \n"
            
    res_text1 += "\\\\ \n \hline \n"
    res_text2 += "\\\\ \n \hline \n"

    # print(f"max_reward_mean = {max_reward_mean:0.2f}")
    print(f"it takes {end_time-start_time:0.4f}")
    log_text += f"it takes {end_time-start_time:0.4f} \n"

    max_reward_mean_list.append(max_reward_mean)
    best_param_list.append(best_param)
    

print(res_text1)
print(res_text2)

tmp = f"num_trials = {m} \n"
with open("./results/63_frozenlake_s3_haver_uni_rollout_v1.txt", 'w+') as f:
    f.write(tmp)
    f.write(log_text)
    f.write("\n")
    f.write(res_text1)
    f.write(res_text2)