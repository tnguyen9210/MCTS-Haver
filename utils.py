
import numpy as np

import logging


class FrozenLakeSimulator:
    def __init__(self, trans_probs, num_actions):
        self.trans_probs = trans_probs
        self.num_actions = num_actions

    def step(self, state, action):
        transitions = self.trans_probs[int(state)][action]
        trans_p = np.array([t[0] for t in transitions])
        # logging.info(f"action_probs = {action_probs}")
        idx = np.random.choice(len(trans_p), 1, p=trans_p)[0]
        p, next_state, reward, terminated = transitions[idx]
        # logging.info(f"state, action, next_state, terminated = {state, action, next_state, terminated}")
        return (next_state, reward, terminated, False, {"prob": p})
