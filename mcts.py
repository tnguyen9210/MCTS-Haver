
import math
import numpy as np

import copy
from tqdm import tqdm
from collections import defaultdict

import logging
# logging.basicConfig(level=logging.INFO)




class MCTS:
    def __init__(self, simulator, num_actions, gamma, action_multi,
                 max_iterations, max_depth, rollout_max_depth,
                 ucb_param):
        self.simulator = simulator
        self.action_multi = action_multi
        self.num_actions = num_actions*action_multi
        self.gamma = gamma
        self.max_iterations = max_iterations
        self.max_depth = max_depth
        self.rollout_max_depth = rollout_max_depth
        self.ucb_param = ucb_param
        
        self.Q_table = defaultdict(lambda: np.zeros(num_actions))
        self.Q_nvisits = defaultdict(lambda: np.zeros(num_actions))
        self.Q_table_d1 = defaultdict(lambda: np.ones(num_actions)) 

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

        action = self.get_action_max_ucb(cur_state, debug=False)
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
            action = self.select_action(cur_state, debug)
            # action = action % self.action_multi
            logging.info(f"action={action}")
            
            next_state, reward, terminated, _, _ = \
                self.simulator.step(cur_state, action)
            
            q = reward + self.gamma*self.search(next_state, depth+1, terminated, debug)

            logging.info(f"after search")
            logging.info(f"cur_state={cur_state}, action={action}, next_state={next_state}, reward={reward}")    

            self.Q_nvisits[cur_state][action] += 1
            total_nvisits = np.sum(self.Q_nvisits[cur_state])
            
            w = 1/total_nvisits            
            self.Q_table[cur_state][action] = \
                (1-w)*self.Q_table[cur_state][action] + w*q
                        
            return q

        
    def select_action(self, cur_state, debug=False):
        # find unvisited_actions
        unvisited_actions = []
        for action in range(self.num_actions):
            if self.Q_nvisits[cur_state][action] == 0:
                unvisited_actions.append(action)
        logging.debug(f"unvisited_actions={unvisited_actions}")
                
        if len(unvisited_actions) != 0:  # some nodes are not visited
            # choose a random action 
            action = np.random.choice(unvisited_actions)
        elif len(unvisited_actions) == 0:
            # choose an action that maximizes ucb
            action = self.get_action_max_ucb(cur_state, debug)

        return action

    
    def get_action_max_ucb(self, cur_state, debug=False):
        if np.sum(self.Q_nvisits[cur_state] > 0) < self.num_actions:
            print("get_action_max_ucb, Q[cur_state] does not have enough children")
            stop

        best_action = None
        max_ucb = float("-inf")

        total_nvisits  = np.sum(self.Q_nvisits[cur_state])
        for action in range(self.num_actions):
            bonus = math.sqrt(2*math.log(total_nvisits)/self.Q_nvisits[cur_state][action])
            action_ucb = self.Q_table[cur_state][action] + bonus*self.ucb_param
                
            if action_ucb > max_ucb:
                max_ucb = action_ucb
                best_action = action

            if debug is True:
                logging.warn(f"idx={idx}")
                logging.warn(f"total_nvisits={total_nvisits}")
                logging.warn(f"action.nvisits={self.Q_nvisits[cur_state][action]}")
                logging.warn(f"max_idx={max_idx},max_ucb={max_ucb:0.4f},")

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
            
