
import math
import numpy as np

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
        
    def get_max_ucb_child(self):
        if len(self.children) == 0:  # no children
            return

        max_idx = 0
        max_ucb = float("-inf")

        for idx, child in enumerate(self.children):
            if child.nvisits == 0:
                child_ucb = float("inf")
            else:
                avg_reward = child.cum_reward/child.nvisits
                ucb = math.sqrt(2*math.log(self.nvisits)/child.nvisits)
                child_ucb = avg_reward + ucb
                
            if child_ucb > max_ucb:
                max_ucb = child_ucb
                max_idx = idx

        return max_idx


class MCTS:
    def __init__(self, simulator, num_actions, gamma,
                 max_steps, max_depth, rollout_max_depth):
        self.simulator = simulator
        self.num_actions = num_actions
        self.gamma = gamma
        self.max_steps = max_steps
        self.max_depth = max_depth
        self.rollout_max_depth = rollout_max_depth

        logging.debug(f"\n-> init")
        logging.debug(f"num_actions={self.num_actions}")
        logging.debug(f"max_steps={self.max_steps}")
        logging.debug(f"max_depth={self.max_depth}")
        logging.debug(f"rollout_max_depth={self.rollout_max_depth}")

    def run(self, cur_state):
        cur_node = Node()
        for i_step in range(self.max_steps):
            logging.info(f"\n\n-> i_step={i_step}")
            self.search(cur_node, cur_state, 0, False, debug=False)

        # logging.warn(f"done")
        # logging.warn(f"cur_node.children={cur_node.children}")
        return self.get_max_ucb_child(cur_node, debug=False)
    
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
            logging.info(f"action={action}")
            next_state, reward, terminated, _, _ = self.simulator.step(cur_state, action)
            q = reward + self.gamma*self.search(child_node, next_state, depth+1, terminated, debug)
            logging.info(f"after search")
            logging.info(f"cur_state={cur_state}, action={action}, next_state={next_state}, reward={reward}")    
            
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
            

    
# def mcts_learning(env, num_episodes_train, eps_max_steps, mcts_max_steps, tdqm_disable):

#     stats = []
#     for i_eps in tqdm(
#             range(num_episodes_train), desc="train q_learning", disable=tdqm_disable):

#         state, info = env.reset()
#         state = f"{state}"
#         start_state = copy.deepcopy(state)

#         eps_reward = 0.0
#         for i_step in range(eps_max_steps):
#             mcts = MCTS(copy.deepcopy(env))
#             action = mcts.search(mcts_max_steps)

#             _, reward, terminated, truncated, _ = env.step(action)
#             eps_reward += reward

#             # env.render()
#             if terminated or truncated:
#                 break

#         print(reward)
#         stats.append((i_step+1, eps_reward/(i_step+1)))
            
#     return stats
