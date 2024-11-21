
import numpy as np

import gym
from env import FrozenLakeCustom, FrozenLakeSimulator

from config import parse_args
from mcts_haver import haver21count

import logging
logging.basicConfig(level=logging.DEBUG)

a = 9
b = a % 4
print(b)
stop

# create gym env
env = FrozenLakeCustom(map_name="4x4", is_slippery=True, render_mode=None)
simulator = FrozenLakeSimulator(env.P)

next_state, reward, terminate, _, _ = simulator.step(0, 1)
print(next_state)
stop
    

# def get_action_max_ucb(action_values, action_nvisits, debug=False):


#     total_nvisits = np.sum(action_nvisits)
#     action_bonuses = np.sqrt(2*np.log(total_nvisits)/action_nvisits)
#     action_ucbs = action_values + action_bonuses*30

#     best_actions = np.where(action_ucbs == np.max(action_ucbs))[0]
#     action = np.random.choice(best_actions)
#     print(best_actions)
#     print(action)
#     # best_action = np.argmax(action_ucbs)
#     return action


param = 10000
values = np.array([-109.,     -100.,      -88.5385, -104.25  ])
nvisits = np.array(  [ 9., 16., 13., 12.])
var = np.array([  33.3333,    0.,     1340.4024,   18.1875])

# action = get_action_max_ucb(values, nvisits)
# # print(best_actions)
# stop
q_est = haver21count(values, nvisits, var, param, debug=True)
print(f"{q_est:0.4f}")
stop

x = np.array([-103, -103, -103, -103, -103, -103, -103, -103, -103, -103, -103, -103, -0, -0])
x = np.array([-118.0, -102.0, -103.0])
x_std0 = np.std(x, ddof=0)
x_std1 = np.std(x, ddof=0)
print(f"x_std0={x_std0}")
print(f"x_std1={x_std1}")


x_avg = 0
x2_avg = 0
for i in range(1,len(x)+1):
    w = 1/i
    x_avg = (1-w)*x_avg + w*x[i-1]
    x2_avg = (1-w)*x2_avg + w*x[i-1]**2
    
print(f"x_avg={x_avg}")
print(f"x2_avg={x2_avg}")

x_std_emp = np.sqrt(x2_avg - x_avg**2)
print(f"x_std_emp={x_std_emp}")
print(f"x_var_emp={x_std_emp**2}")

# stop
# env_id = "FrozenLake-v1"
# env = FrozenLakeCustom(map_name="4x4", is_slippery=True, render_mode=None)

# simulator = FrozenLakeSimulator(env.P, num_actions)
# state, info = env.reset()
# logging.info(f"state = {state}")
# actions_ary = [2, 1, 2, 0, 1, 1, 2, 2, 3, 1, 1, 2]
# for action in actions_ary:
#     new_state, reward, terminated, truncated, info = simulator.step(state, action)
#     logging.info(f"state, action, new_state, reward, terminated = {state, action, new_state, reward, terminated}")
#     if terminated:
#         state = 0
#     else:
#         state = new_state
