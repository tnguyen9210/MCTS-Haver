
import math
import numpy as np
np.set_printoptions(precision=4, suppress=True)

import copy
from tqdm import tqdm
from collections import defaultdict

import logging
# logging.basicConfig(level=logging.INFO)


class Node:
    def __init__(self, action=None, parent=None):
        # self.state = state
        self.children = []
        self.q_value = 0
        self.nvisits = 0
        
        self.action = action
        self.parent = parent

    def add_child(self, action):
        child = Node(action, self)
        self.children.append(child)
        return child
        

class MCTS_Haver:
    def __init__(self, simulator, num_actions, gamma, action_multi,
                 max_steps, max_depth, rollout_max_depth, update_method):
        self.simulator = simulator
        self.action_multi = action_multi
        self.num_actions = int(num_actions*action_multi)
        self.gamma = gamma
        self.max_steps = max_steps
        self.max_depth = max_depth
        self.rollout_max_depth = rollout_max_depth
        self.update_method = update_method

        logging.debug(f"\n-> init")
        logging.debug(f"num_actions={self.num_actions}")
        logging.debug(f"max_steps={self.max_steps}")
        logging.debug(f"max_depth={self.max_depth}")
        logging.debug(f"rollout_max_depth={self.rollout_max_depth}")

    def run(self, cur_state):
        cur_node = Node()
        for i_step in range(self.max_steps):
            # logging.warn(f"\n\n-> i_step={i_step}")
            action, child_node = self.select_action_ucb(cur_node)
            # action = action % self.action_multi
            next_state, reward, terminated, _, _ = self.simulator.step(cur_state, action)
            q = self.search(child_node, next_state, 1, terminated, debug=False)

            if self.update_method == "haver":
                q = haver21count(child_node, self.num_actions, debug=False)

            update = reward + self.gamma*q
            w = 1/child_node.nvisits
            child_node.q_value = (1-w)*child_node.q_value + w*update

        # if (ep_step+1) % 1 == 0:
        #     print(f"ep_step = {ep_step+1}")
        #     print(f"cur_node.q_value = {cur_node.q_value}")
        #     for idx, child in enumerate(cur_node.children):
        #         print(f"child={idx}, child.q_value={child.q_value:0.4f}")
        
        action = self.get_max_ucb_child(cur_node, debug=False)
        # action = action % self.action_multi
        return action
    
    def search(self, cur_node, cur_state, depth, terminated, debug):
        logging.info(f"\n-> search")
        logging.info(f"cur_state={cur_state}, depth={depth}, terminated={terminated}")
        cur_node.nvisits += 1
        
        if terminated:
            return 0
        
        if depth > self.max_depth:
            logging.info(f"case: depth > max_depth")
            return self.rollout(cur_state)
        
        elif depth <= self.max_depth:
            logging.info(f"case: depth <= max_depth")
            action, child_node = self.select_action_ucb(cur_node, debug)
            # action = action % self.action_multi
            logging.info(f"action={action}")
            next_state, reward, terminated, _, _ = self.simulator.step(cur_state, action)
            q = reward + self.gamma*self.search(child_node, next_state, depth+1, terminated, debug)
            logging.info(f"after search")
            logging.info(f"cur_state={cur_state}, action={action}, next_state={next_state}, reward={reward}")    

            if depth > 1:
                w = 1/cur_node.nvisits
                cur_node.q_value = (1-w)*cur_node.q_value + w*q
            
            logging.info(f"cur_node(q_value,nvisits)={cur_node.q_value, cur_node.nvisits}")
            
            return q

        
    def select_action_ucb(self, cur_node, debug=False):
        visited_actions = set(child.action for child in cur_node.children)
        unvisited_actions = set(range(self.num_actions)) - visited_actions
        logging.debug(f"visited_actions={visited_actions}")
        logging.debug(f"unvisited_actions={unvisited_actions}")
        if len(unvisited_actions) != 0:  # some nodes are not visited
            # choose a random action 
            action = np.random.choice(list(unvisited_actions))
            child_node = cur_node.add_child(action)
        elif len(unvisited_actions) == 0:
            # choose an action that maximizes ucb
            action = self.get_max_ucb_child(cur_node, debug)
            child_node = cur_node.children[action]

        return action, child_node

    
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

    
    def get_max_ucb_child(self, cur_node, debug=False):
        if len(cur_node.children) < self.num_actions:
            print("get_max_ucb_child, cur_node does not have enough children")
            stop

        max_idx = 0
        max_ucb = float("-inf")

        total_nvisits  = np.sum(child.nvisits for child in cur_node.children)
        for idx, child in enumerate(cur_node.children):
            ucb = math.sqrt(2*math.log(total_nvisits)/child.nvisits)
            child_ucb = child.q_value + ucb
                
            if child_ucb > max_ucb:
                max_ucb = child_ucb
                max_idx = idx

            if debug is True:
                logging.warn(f"idx={idx}")
                logging.warn(f"cur_node.nvisits={cur_node.nvisits}")
                logging.warn(f"total_nvisits={total_nvisits}")
                logging.warn(f"child.nvisits={child.nvisits}")
                logging.warn(f"child_q_value={child.q_value:0.4f},ucb={ucb:0.4f},child_ucb={child_ucb:0.4f}")
                logging.warn(f"max_idx={max_idx},max_ucb={max_ucb:0.4f},")

        return max_idx
            
def haver21count(cur_node, num_actions, debug=False):
    rhat_idx = None
    rhat_gam = None
    rhat_muhat = None
    max_lcb = -np.inf
    num_children = len(cur_node.children)
    total_nvisits = np.sum([child.nvisits for child in cur_node.children])
    for idx, child in enumerate(cur_node.children):
        gam_log = (num_children*total_nvisits/child.nvisits)**4
        child_gam = np.sqrt(18/child.nvisits*np.log(gam_log))
        child_lcb = child.q_value - child_gam
        if child_lcb > max_lcb:
            max_lcb = child_lcb
            rhat_idx = idx
            rhat_gam = child_gam
            # rhat_muhat = child.q_value

    # print(max_lcb)
    Bset_idxes = []
    Bset_muhats = np.zeros(len(cur_node.children))
    Bset_nvisits = np.zeros(len(cur_node.children))
    for idx, child in enumerate(cur_node.children):
        gam_log = (num_children*total_nvisits/child.nvisits)**4
        child_gam = np.sqrt(18/child.nvisits*np.log(gam_log))
        # if debug:
        #     logging.warn(f"rhat_gam = {rhat_gam:0.4f}")
        #     logging.warn(f"child_gam = {child_gam:0.4f}")
        if child.q_value >= max_lcb and child_gam <= 3.0/2*rhat_gam:
            Bset_muhats[idx] = child.q_value
            Bset_nvisits[idx] = child.nvisits
            Bset_idxes.append(idx)
            

    Bset_probs = Bset_nvisits/np.sum(Bset_nvisits)
    q_est = np.dot(Bset_muhats, Bset_probs)
    if debug:
        logging.warn(f"children.nvisits = {[child.nvisits for child in cur_node.children]}")
        logging.warn(f"children.q_value = {np.array([child.q_value for child in cur_node.children])}")
        logging.warn(f"max_lcb = {max_lcb:0.4f}")
        logging.warn(f"Bset_idxes = {Bset_idxes}")
        logging.warn(f"Bset_nvisits = {Bset_nvisits}")
        logging.warn(f"Bset_probs = {Bset_probs}")
        logging.warn(f"Bset_muhats = {Bset_muhats}")
        # loging.warn(tmp)
        logging.warn(f"Q_est = {q_est:.2f}")

    return q_est
    
