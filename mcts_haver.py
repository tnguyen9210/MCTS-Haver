
import math
import numpy as np
import random

import time 

import copy
from tqdm import tqdm
from collections import defaultdict

import logging
# logging.basicConfig(level=logging.INFO)


class MCTS:
    def __init__(self, simulator, num_actions, gamma, action_multi,
                 max_iterations, max_depth, rollout_max_depth,
                 hparam_ucb_scale, hparam_haver_var, update_method):
        self.simulator = simulator
        self.action_multi = action_multi
        self.num_actions = num_actions*action_multi
        self.gamma = gamma
        self.max_iterations = max_iterations
        self.max_depth = max_depth
        self.rollout_max_depth = rollout_max_depth
        self.hparam_ucb_scale = hparam_ucb_scale
        self.hparam_haver_var = hparam_haver_var

        self.update_method = update_method
        
        self.Q = defaultdict(lambda: np.zeros(num_actions))
        self.QN = defaultdict(lambda: np.zeros(num_actions))
        self.QH = defaultdict(lambda: np.ones(num_actions))  # Q-table for Haver
        self.QR = defaultdict(lambda: np.ones(num_actions))  # Q-table for avg reward

        logging.debug(f"\n-> init")
        logging.debug(f"num_actions={self.num_actions}")
        logging.debug(f"max_iterations={self.max_iterations}")
        logging.debug(f"max_depth={self.max_depth}")
        logging.debug(f"rollout_max_depth={self.rollout_max_depth}")

    def run(self, cur_state):
        # cur_node = Node()
        for it in range(self.max_iterations):
            logging.info(f"\n\n-> it={it}")
            self.search(cur_state, 0, False, debug=False)

        action = np.argmax(self.Q[cur_state])
        # action = self.get_action_max_ucb(cur_state, debug=True)
        # action = action % self.action_multi
        return action
    
    def search(self, cur_state, depth, terminated, debug):
        logging.info(f"\n-> search")
        logging.info(f"cur_state={cur_state}, depth={depth}, terminated={terminated}")
        
        if terminated:
            return 0
        
        if depth > self.max_depth:
            logging.info(f"case: depth > max_depth")
            return self.rollout(cur_state)
        
        elif depth <= self.max_depth:
            logging.info(f"case: depth <= max_depth")
            action = self.select_action(cur_state, depth, debug)
            # action = action % self.action_multi
            logging.info(f"action={action}")
            
            next_state, reward, terminated, _, _ = \
                self.simulator.step(cur_state, action)
            
            q = reward + self.gamma*self.search(next_state, depth+1, terminated, debug)

            logging.info(f"after search")
            logging.info(f"cur_state={cur_state}, action={action}, next_state={next_state}, reward={reward}")    

            self.QN[cur_state][action] += 1
            total_nvisits = np.sum(self.QN[cur_state])
            
            w = 1/total_nvisits            
            self.Q[cur_state][action] = \
                  (1-w)*self.Q[cur_state][action] + w*q

            self.QR[cur_state][action] = (1-w)*self.QR[cur_state][action] + w*reward

            if depth == 0:
                self.QH[cur_state][action] = \
                    self.QR[cur_state][action] + haver21count(
                        self.Q[cur_state], self.QN[cur_state], self.num_actions, self.hparam_haver_var)
                        
            return q

        
    def select_action(self, cur_state, depth, debug=False):
        # find unvisited_actions
        unvisited_actions = []
        for action in range(self.num_actions):
            if self.QN[cur_state][action] == 0:
                unvisited_actions.append(action)
        logging.debug(f"unvisited_actions={unvisited_actions}")
                
        if len(unvisited_actions) != 0:  # some nodes are not visited
            # choose a random action 
            action = np.random.choice(unvisited_actions)
        elif len(unvisited_actions) == 0:
            # choose an action that maximizes ucb
            if self.update_method == "haver" and depth == 0:
                action_values = self.QH[cur_state]
            else:
                action_values = self.Q[cur_state]

            action_nvisits = self.QN[cur_state]
            action = self.get_action_max_ucb(action_values, action_nvisits, debug)

        return action

    
    def get_action_max_ucb(self, action_values, action_nvisits, debug=False):
        if np.sum(action_nvisits > 0) < self.num_actions:
            print("get_action_max_ucb, Q[cur_state] does not have enough children")
            stop

        best_action = None
        max_ucb = float("-inf")

        total_nvisits = np.sum(action_nvisits)
        for a in range(self.num_actions):
            a_nvisits = action_nvisits[a]
            a_value = action_values[a]
            if a_nvisits != 0:
                bonus = math.sqrt(2*math.log(total_nvisits)/a_nvisits)
                a_ucb = a_value + bonus*self.hparam_ucb_scale
                
            if a_ucb > max_ucb:
                max_ucb = a_ucb
                best_action = a

            if debug is True:
                logging.warning(f"action={a}")
                logging.warning(f"total_nvisits={total_nvisits}")
                logging.warning(f"action.value={a_value:0.4f}")
                logging.warning(f"action.nvisits={a_nvisits}")
                logging.warning(f"action.bonus={bonus:0.4f}")
                logging.warning(f"action.ucb={action_ucb:0.4f}")
                logging.warning(f"best_action={best_action},max_ucb={max_ucb:0.4f},")

        return best_action

    
    def rollout(self, cur_state):
        # logging.info(f"\n-> rollout")
        total_reward = 0
        for i_depth in range(self.rollout_max_depth):
            # logging.info(f"i_depth={i_depth}")
            action = np.random.choice(range(self.num_actions))
            # action = action % self.action_multi
            # logging.info(f"action={action}")
            next_state, reward, terminated, _, _ = \
                self.simulator.step(cur_state, action)
            total_reward += reward
            
            # logging.info(f"cur_state={cur_state}, action={action}, next_state={next_state}, reward={reward}")

            if terminated:
                break

            cur_state = next_state
            
        return total_reward

    
def haver21count(action_values, action_nvisits, num_actions, hparam_haver_var, debug=False):
    rhat_idx = None
    rhat_gam = None
    rhat_muhat = None
    max_lcb = -np.inf
    total_nvisits = np.sum(action_nvisits)
    num_actions_visited = np.sum(action_nvisits > 0)
    for a in range(num_actions):
        a_nvisits = action_nvisits[a]
        a_value = action_values[a]
        if a_nvisits != 0:
            gam_log = (num_actions_visited*total_nvisits/a_nvisits)**4
            a_gam = np.sqrt(18/(a_nvisits*hparam_haver_var)*np.log(gam_log))
            a_lcb = a_value - a_gam
            if a_lcb > max_lcb:
                max_lcb = a_lcb
                rhat_idx = a
                rhat_gam = a_gam
                # rhat_muhat = child.q_value

    # print(max_lcb)
    Bset_idxes = []
    Bset_muhats = np.zeros(num_actions)
    Bset_nvisits = np.zeros(num_actions)
    for a in range(num_actions):
        a_nvisits = action_nvisits[a]
        a_value = action_values[a]
        if a_nvisits != 0:
            gam_log = (num_actions_visited*total_nvisits/a_nvisits)**4
            a_gam = np.sqrt(18/(a_nvisits*hparam_haver_var)*np.log(gam_log))
            # if debug:
            #     logging.warn(f"rhat_gam = {rhat_gam:0.4f}")
            #     logging.warn(f"child_gam = {child_gam:0.4f}")
            if a_value >= max_lcb and a_gam <= 3.0/2*rhat_gam:
                Bset_muhats[a] = a_value
                Bset_nvisits[a] = a_nvisits
                Bset_idxes.append(a)
            

    Bset_probs = Bset_nvisits/np.sum(Bset_nvisits)
    haver_est = np.dot(Bset_muhats, Bset_probs)
    if debug:
        logging.warn(f"action.nvisits = {action_nvisits}")
        logging.warn(f"action_values = {action_values}")
        logging.warn(f"max_lcb = {max_lcb:0.4f}")
        logging.warn(f"Bset_idxes = {Bset_idxes}")
        logging.warn(f"Bset_nvisits = {Bset_nvisits}")
        logging.warn(f"Bset_probs = {Bset_probs}")
        logging.warn(f"Bset_muhats = {Bset_muhats}")
        # loging.warn(tmp)
        logging.warn(f"Q_est = {q_est:.2f}")

    return haver_est
    
