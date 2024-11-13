
from typing import List, Optional

import numpy as np 

import gym
from gym import Env, logger, spaces, utils
from gym.envs.toy_text import FrozenLakeEnv

import logging

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

MAPS = {
    "4x4": ["SFFF",
            "FHFH",
            "FFFH",
            "HFFG"],
    "4x4X": ["SFFF",
            "FHFH",
            "XFFH",
            "HFFG"],
    "8x8": [
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FFFFFHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFG",
    ],
}

class FrozenLakeCustom(FrozenLakeEnv):
    def __init__(
        self,
        render_mode: Optional[str] = None,
        desc=None,
        map_name="4x4",
        is_state_slippery=False,
        is_slippery=True,
        slippery_mode="extreme",
    ):
        # super().__init__()

        if desc is None and map_name is None:
            desc = generate_random_map()
        elif desc is None:
            desc = MAPS[map_name]
        self.desc = desc = np.asarray(desc, dtype="c")
        self.nrow, self.ncol = nrow, ncol = desc.shape
        self.reward_range = (0, 1)

        nA = 4
        nS = nrow * ncol

        self.initial_state_distrib = np.array(desc == b"S").astype("float64").ravel()
        self.initial_state_distrib /= self.initial_state_distrib.sum()

        self.P = {s: {a: [] for a in range(nA)} for s in range(nS)}

        def to_s(row, col):
            return row * ncol + col

        def inc(row, col, a):
            if a == LEFT:
                col = max(col - 1, 0)
            elif a == DOWN:
                row = min(row + 1, nrow - 1)
            elif a == RIGHT:
                col = min(col + 1, ncol - 1)
            elif a == UP:
                row = max(row - 1, 0)
            return (row, col)

        def update_probability_matrix(row, col, action):
            newrow, newcol = inc(row, col, action)
            newstate = to_s(newrow, newcol)
            newletter = desc[newrow, newcol]
            terminated = bytes(newletter) in b"GH"
            # reward = float(newletter == b"G")
            reward = -1
            if newletter == b"G":
                reward = 1
            elif newletter == b"H":
                reward = -100
            return newstate, reward, terminated

        for row in range(nrow):
            for col in range(ncol):
                s = to_s(row, col)
                for a in range(4):
                    li = self.P[s][a]
                    letter = desc[row, col]
                    if letter in b"GH":
                        li.append((1.0, s, 0, True))
                    elif letter in b"X":
                        if is_state_slippery:
                            for b in range(4):
                                li.append(
                                    (1.0 / 4.0, *update_probability_matrix(row, col, b))
                                )
                    else:
                        if is_slippery:
                            stop
                            if slippery_mode == "extreme":
                                for b in [(a - 1) % 4, a, (a + 1) % 4]:
                                    li.append(
                                        (1.0 / 3.0, *update_probability_matrix(row, col, b))
                                )
                            else:
                                li.append((0.8, *update_probability_matrix(row, col, a)))
                                for b in [(a - 1) % 4, (a + 1) % 4]:
                                    li.append(
                                        (0.1, *update_probability_matrix(row, col, b))
                                    )
                            
                        else:
                            li.append((1.0, *update_probability_matrix(row, col, a)))

        self.observation_space = spaces.Discrete(nS)
        self.action_space = spaces.Discrete(nA)

        self.render_mode = render_mode

        # pygame utils
        self.window_size = (min(64 * ncol, 512), min(64 * nrow, 512))
        self.cell_size = (
            self.window_size[0] // self.ncol,
            self.window_size[1] // self.nrow,
        )
        self.window_surface = None
        self.clock = None
        self.hole_img = None
        self.cracked_hole_img = None
        self.ice_img = None
        self.elf_images = None
        self.goal_img = None
        self.start_img = None
    

class FrozenLakeSimulator:
    def __init__(self, trans_probs, simulator_seed):
        self.trans_probs = trans_probs
        self.num_states = len(trans_probs)
        self.num_actions = len(trans_probs[0])
        self.rng = np.random.Generator(np.random.PCG64(simulator_seed))

    def step(self, state, action):
        transitions = self.trans_probs[int(state)][action]
        trans_p = np.array([t[0] for t in transitions])
        # logging.warn(f"state={state}, trans_p = {trans_p}")
        idx = np.random.choice(len(trans_p), 1, p=trans_p)[0]
        p, next_state, reward, terminated = transitions[idx]
        # logging.info(f"state, action, next_state, terminated = {state, action, next_state, terminated}")
        return (next_state, reward, terminated, False, {"prob": p})
