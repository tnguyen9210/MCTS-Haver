
import numpy as np

import gym
from env import FrozenLakeCustom, FrozenLakeSimulator

from config import parse_args
from mcts_haver import haver21count

a = [2, 4, 3]
b = [2, 3, 3]

if np.any(a != b):
    print("yes")

param = 0.1
values = np.array([-101.,    0., -100.,    0.])
nvisits = np.array([1., 0., 1., 0.])

q_est = haver21count(values, nvisits,  param)
print(q_est)

stop
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
