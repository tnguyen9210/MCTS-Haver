
import math
import numpy as np
import random

import time 

import copy
from tqdm import tqdm
from collections import defaultdict

import logging
# logging.basicConfig(level=logging.INFO)

import ipdb

class MCTS:
    def __init__(self, simulator, rollout_Q, args):

        self.tol = 1e-7
        self.simulator = simulator
        self.num_actions = simulator.num_actions*args["action_multi"]
        # self.action_multi = args["action_multi"]
        self.gamma = args["gamma"]
        self.num_trajectories = args["mcts_num_trajectories"]
        self.max_depth = args["mcts_max_depth"]
        self.rollout_max_depth = args["mcts_rollout_max_depth"]
        self.hparam_ucb_scale = args["hparam_ucb_scale"]
        self.hparam_haver_var = args["hparam_haver_var"]

        self.update_method = args["update_method"]
        
        self.rollout_method = args["rollout_method"]
        self.rollout_Q = rollout_Q
        
        logging.debug(f"\n-> init")
        logging.debug(f"num_actions={self.num_actions}")
        logging.debug(f"num_trajectories={self.num_trajectories}")
        logging.debug(f"max_depth={self.max_depth}")
        logging.debug(f"rollout_max_depth={self.rollout_max_depth}")

    def run(self, cur_state):
        # cur_node = Node()
        self.N = defaultdict(lambda: np.zeros(self.num_actions))
        self.NH = defaultdict(lambda: np.zeros(self.num_actions))
        self.NM = defaultdict(lambda: np.zeros(self.num_actions))

        self.Q = defaultdict(lambda: np.zeros(self.num_actions))
        self.QH = defaultdict(lambda: -np.inf*np.ones(self.num_actions))  # Q-table for Haver
        self.QM = defaultdict(lambda: -np.inf*np.ones(self.num_actions))  # Q-table for Haver

        self.R = defaultdict(lambda: np.zeros(self.num_actions))  # Q-table for avg reward
        
        self.Q2 = defaultdict(lambda: np.zeros(self.num_actions))
        self.var = defaultdict(lambda: np.zeros(self.num_actions))

        # self.Q_list = defaultdict(lambda: defaultdict(list))
        
        
        # ipdb.set_trace()
        
        for it in range(self.num_trajectories):
            # ipdb.set_trace()
            logging.info(f"\n\n-> it={it}")
            self.search(cur_state, 0, False, debug=False)

        if self.update_method == "haver":
            action = np.argmax(self.QH[cur_state])
            logging.warn(f"action={action}")
        elif self.update_method == "max":
            action = np.argmax(self.QM[cur_state])
        else:
            action = np.argmax(self.Q[cur_state])
        # action = self.get_action_max_ucb(cur_state, debug=True)
        # action = action % self.action_multi
        return action
    
    def search(self, cur_state, depth, terminated, debug):
        logging.debug(f"\n-> search")
        logging.debug(f"cur_state={cur_state}, depth={depth}, terminated={terminated}")
        
        if terminated:
            return 0
        
        if depth > self.max_depth:
            logging.debug(f"case: depth > max_depth")
            return self.rollout(cur_state)
        
        elif depth <= self.max_depth:
            logging.debug(f"case: depth <= max_depth")
            action = self.select_action(cur_state, depth, debug)
            action_t = action % 4
            logging.debug(f"action={action}")
            
            next_state, reward, terminated, _, _ = \
                self.simulator.step(cur_state, action_t)
            
            q = reward + self.gamma*self.search(next_state, depth+1, terminated, debug)

            logging.debug(f"after search")
            logging.debug(f"cur_state={cur_state}, action={action}, next_state={next_state}, reward={reward}")    

            self.N[cur_state][action] += 1
            
            w = 1/self.N[cur_state][action]            
            self.Q[cur_state][action] = \
                  (1-w)*self.Q[cur_state][action] + w*q
            logging.debug(f"Q[cur_state][action] = {self.Q[cur_state][action]:0.4f}")    

            self.R[cur_state][action] = (1-w)*self.R[cur_state][action] + w*reward

            # self.Q_list[cur_state][action].append(q)
            
            self.Q2[cur_state][action] = \
                  (1-w)*self.Q2[cur_state][action] + w*q**2
            
            self.var[cur_state][action] = \
                self.Q2[cur_state][action] - self.Q[cur_state][action]**2

            if depth == 0:
                logging.info(f"\n-> depth-0, cur_state={cur_state}, action={action}, {self.N[next_state]}")
                # pdb.set_trace()
                # ipdb.set_trace()

                self.NH[cur_state][action] += 1
                self.NM[cur_state][action] += 1
                
                if terminated:
                    self.QH[cur_state][action] = self.R[cur_state][action]
                    self.QM[cur_state][action] = self.R[cur_state][action]
                        
                elif np.sum(self.N[next_state]) > 0:
                    if self.update_method == "haver":
                        # if np.sum(self.N[next_state] <= 10) > 1:
                        #     self.QH[cur_state][action] = copy.deepcopy(self.Q[cur_state][action])
                        # else:
                        self.QH[cur_state][action] = \
                            self.R[cur_state][action] + haver21count(
                                self.Q[next_state], self.N[next_state],
                                self.var[next_state],
                                self.hparam_haver_var, debug)
                        
                        logging.info(f"Q[next_state]= {self.Q[next_state]}")
                        logging.info(f"N[next_state]= {self.N[next_state]}")
                        logging.info(f"var[next_state]= {self.var[next_state]}")
                        # logging.info(f"Q_list[next_state]= {self.Q_list[next_state]}")
                        logging.info(f"QH[cur_state]= {self.QH[cur_state]}")
                        logging.info(f"QH[cur_state][action]= {self.QH[cur_state][action]:0.4f}")
                        logging.info(f"Q[cur_state]= {self.Q[cur_state]}")
                        logging.info(f"Q[cur_state][action]= {self.Q[cur_state][action]:0.4f}")
                        a = 0
                        
                    elif self.update_method == "max":
                        self.QM[cur_state][action] = \
                            self.R[cur_state][action] + np.max(
                                self.Q[next_state][self.N[next_state] > 0])

                        # self.QH[cur_state][action] = \
                        #     self.R[cur_state][action] + haver21count(
                        #         self.Q[next_state], self.N[next_state],
                        #         self.hparam_haver_var, debug)

                        # if np.any(self.QM[cur_state][action] != self.QH[cur_state][action]):
                        #     ipdb.set_trace()
                        
                        logging.info(f"Q[next_state]= {self.Q[next_state]}")
                        logging.info(f"N[next_state]= {self.N[next_state]}")
                        # logging.info(f"var[next_state]= {self.var[next_state]}")
                        # logging.info(f"Q_list[next_state]= {self.Q_list[next_state]}")
                        logging.info(f"QM[cur_state]= {self.QM[cur_state]}")
                        logging.info(f"QM[cur_state][action]= {self.QM[cur_state][action]:0.4f}")
                        # logging.info(f"Q[cur_state]= {self.Q[cur_state]}")
                        # logging.info(f"Q[cur_state][action]= {self.Q[cur_state][action]:0.4f}")
                        
                        # logging.info(f"QM[cur_state][action]= {self.QM[cur_state][action]}")
                        # logging.info(f"QH[cur_state][action]= {self.QH[cur_state][action]}")

                        a = 0
                # ipdb.set_trace()
                        
            return q

    def select_action(self, cur_state, depth, debug=False):
        logging.debug(f"\n-> select_action")
        if self.update_method == "haver" and depth == 0:
            # find unvisited_actions
            unvisited_actions = []
            for action in range(self.num_actions):
                if self.NH[cur_state][action] == 0:
                    unvisited_actions.append(action)
                    
            logging.info(f"unvisited_actions={unvisited_actions}")

            if len(unvisited_actions) != 0:  # some nodes are not visited
                # choose a random action 
                # ipdb.set_trace()
                # action = np.random.choice(unvisited_actions)
                action_values = self.Q[cur_state]
                action_nvisits = self.N[cur_state]
                action = self.get_action_max_ucb(action_values, action_nvisits, debug)
            else:
                action_values = self.QH[cur_state]
                action_nvisits = self.NH[cur_state]
                action = self.get_action_max_ucb(action_values, action_nvisits, debug)

        elif self.update_method == "max" and depth == 0:
            # find unvisited_actions
            unvisited_actions = []
            for action in range(self.num_actions):
                if self.NM[cur_state][action] == 0:
                    unvisited_actions.append(action)
            logging.info(f"unvisited_actions={unvisited_actions}")

            if len(unvisited_actions) != 0:  # some nodes are not visited
                # choose a random action 
                # ipdb.set_trace()
                action_values = self.Q[cur_state]
                action_nvisits = self.N[cur_state]
                action = self.get_action_max_ucb(action_values, action_nvisits, debug)
                # action = np.random.choice(unvisited_actions)
            else:
                action_values = self.QM[cur_state]
                action_nvisits = self.NM[cur_state]
                action = self.get_action_max_ucb(action_values, action_nvisits, debug)
                
        else:
            # find unvisited_actions
            unvisited_actions = []
            for action in range(self.num_actions):
                if self.N[cur_state][action] == 0:
                    unvisited_actions.append(action)
            logging.info(f"unvisited_actions={unvisited_actions}")

            if len(unvisited_actions) != 0:  # some nodes are not visited
                # choose a random action 
                action = np.random.choice(unvisited_actions)
            elif len(unvisited_actions) == 0:
                # choose an action that maximizes ucb
                action_values = self.Q[cur_state]
                action_nvisits = self.N[cur_state]
                action = self.get_action_max_ucb(action_values, action_nvisits, debug)

        return action
    
    def get_action_max_ucb(self, action_values, action_nvisits, debug=False):
        # if np.sum(action_nvisits > 0) < self.num_actions:
        #     print("get_action_max_ucb, Q[cur_state] does not have enough children")
        #     stop


        idx_action_visited = (action_nvisits > 0)
        n_action_visited = np.sum(action_nvisits > 0)

        action_ucbs = np.inf * np.ones(len(action_values))
        if n_action_visited != 0:
            my_action_values = action_values[idx_action_visited]
            my_action_nvisits = action_nvisits[idx_action_visited]
            
            total_nvisits = np.sum(my_action_nvisits)
            action_bonuses = np.sqrt(2*np.log(total_nvisits)/my_action_nvisits)
            my_action_ucbs = my_action_values + action_bonuses*self.hparam_ucb_scale

            action_ucbs[idx_action_visited] = my_action_ucbs

        maxval = np.max(action_ucbs)

        if maxval == np.inf:
            best_actions = np.where(action_ucbs == np.inf)[0]
        else:
            best_actions = np.where(np.abs(action_ucbs - maxval) < self.tol)[0]

        # best_actions = np.where(action_ucbs == np.max(action_ucbs))[0]
        if len(best_actions) == 0:
            print(action_values)
            print(action_nvisits)
            print(action_ucbs)
            print(best_actions)
            
        action = np.random.choice(best_actions)
        
        # best_actions = [i if action_ucbs[i] == np.max(action_ucbs) for i in range(self.num_actions)]
        # action = np.argmax(action_ucbs)
    

        # best_action = None
        # max_ucb = float("-inf")
        
        # for a in range(self.num_actions):
        #     a_nvisits = action_nvisits[a]
        #     a_value = action_values[a]
        #     if a_nvisits != 0:
        #         bonus = math.sqrt(2*math.log(total_nvisits)/a_nvisits)
        #         a_ucb = a_value + bonus*self.hparam_ucb_scale
                
        #     if a_ucb > max_ucb:
        #         max_ucb = a_ucb
        #         best_action = a

        #     if debug is True:
        #         logging.warning(f"action={a}")
        #         logging.warning(f"total_nvisits={total_nvisits}")
        #         logging.warning(f"action.value={a_value:0.4f}")
        #         logging.warning(f"action.nvisits={a_nvisits}")
        #         logging.warning(f"action.bonus={bonus:0.4f}")
        #         logging.warning(f"action.ucb={action_ucb:0.4f}")
        #         logging.warning(f"best_action={best_action},max_ucb={max_ucb:0.4f},")

        return action

    
    def rollout(self, cur_state):
        # logging.info(f"\n-> rollout")
        total_reward = 0
        for i_depth in range(self.rollout_max_depth):
            # logging.info(f"i_depth={i_depth}")
            if self.rollout_method == "vit":
                action = np.argmax(self.rollout_Q[cur_state])
                # logging.warning(f"cur_state={cur_state}, action={action}")
            else:
                action = np.random.choice(range(self.num_actions))
            # action = action % self.action_multi
            action_t = action % 4
            # logging.info(f"action={action}")
            next_state, reward, terminated, _, _ = \
                self.simulator.step(cur_state, action_t)
            total_reward += reward
            
            # logging.info(f"cur_state={cur_state}, action={action}, next_state={next_state}, reward={reward}")

            if terminated:
                break

            cur_state = next_state
            
        return total_reward

    
def haver21count(
        action_values, action_nvisits, action_vars, hparam_haver_var, debug=False):
    
    # num_actions = len(action_values)  
    # num_actions_visited = np.sum(action_nvisits > 0)
    # total_nvisits = np.sum(action_nvisits)

    # visited_idxes = action_nvisits != 0

    # # compute rhat
    # gam_log = np.zeros(num_actions)
    # action_gams = np.inf*np.ones(num_actions)
    # gam_log[visited_idxes] = \
    #     (num_actions_visited*total_nvisits/action_nvisits[visited_idxes])**4
    # action_gams[visited_idxes] = \
    #     hparam_haver_var*np.sqrt(18/action_nvisits[visited_idxes]*np.log(gam_log[visited_idxes]))
    # action_lcbs = action_values - action_gams
    # rhat_idx = np.argmax(action_lcbs)
    # rhat_gam = action_gams[rhat_idx]
    # max_lcb = action_lcbs[rhat_idx]

    # Bset_muhats = np.zeros(num_actions)
    # Bset_nvisits = np.zeros(num_actions)

    # Bset_conds = visited_idxes & (action_values >= max_lcb) & (action_gams <= 3.0/2*rhat_gam)
    # Bset_muhats[Bset_conds] = action_values[Bset_conds]
    # Bset_nvisits[Bset_conds] = action_nvisits[Bset_conds]
    
    # Bset_probs = Bset_nvisits/np.sum(Bset_nvisits)
    # haver_est = np.dot(Bset_muhats, Bset_probs)

    # if debug:
    #     logging.warn(f"action.nvisits = {action_nvisits}")
    #     logging.warn(f"action_values = {action_values}")
    #     logging.warn(f"max_lcb = {max_lcb:0.4f}")
    #     # logging.warn(f"Bset_idxes = {Bset_idxes}")
    #     logging.warn(f"Bset_conds = {Bset_conds}")
    #     logging.warn(f"Bset_nvisits = {Bset_nvisits}")
    #     logging.warn(f"Bset_probs = {Bset_probs}")
    #     logging.warn(f"Bset_muhats = {Bset_muhats}")
    #     # loging.warn(tmp)
    #     logging.warn(f"haver_est = {haver_est:.2f}")

    num_actions = len(action_values)  
    num_actions_visited = np.sum(action_nvisits > 0)
    total_nvisits = np.sum(action_nvisits)

    rhat_idx = None
    # rhat_idx = None
    # rhat_gam = None
    # rhat_muhat = None
    # rhat_nvisits = None
    # max_lcb = -np.inf

    valid_idx = np.where(action_nvisits > 0)[0]
    my_action_values = action_values[valid_idx]
    my_action_nvisits = action_nvisits[valid_idx]
    my_action_vars = action_vars[valid_idx]

    my_vars = np.maximum(my_action_vars, hparam_haver_var/my_action_nvisits)
    my_gams_log = np.log(my_action_nvisits*total_nvisits/0.05)
    my_gams = np.sqrt(my_vars)*np.sqrt(2/my_action_nvisits*my_gams_log)

    my_lcbs = my_action_values - my_gams

    rhat_idx = np.argmax(my_lcbs)
    maxlcb = my_lcbs[rhat_idx]
    rhat_nvisits = my_action_nvisits[rhat_idx]

    my_Bset_idx = np.where((my_action_values >= maxlcb) & (my_action_nvisits >= 4.0/9*rhat_nvisits))[0]
    my_Bset_values = my_action_values[my_Bset_idx]
    my_Bset_nvisits = my_action_nvisits[my_Bset_idx]
    my_Bset_probs = my_Bset_nvisits/np.sum(my_Bset_nvisits)
    haver_est = np.dot(my_Bset_values, my_Bset_probs)
    
    
    # num_actions = len(action_values)  
    # num_actions_visited = np.sum(action_nvisits > 0)
    # total_nvisits = np.sum(action_nvisits)
    
    # visited_idxes = action_nvisits != 0
    # action_vs = -10*np.ones(num_actions)
    # action_vs[visited_idxes] = np.maximum(
    #     action_vars[visited_idxes], hparam_haver_var/action_nvisits[visited_idxes])
    
    # rhat_idx = None
    # rhat_gam = None
    # rhat_muhat = None
    # rhat_nvisits = None
    # max_lcb = -np.inf
    # for a in range(num_actions):
    #     a_nvisits = action_nvisits[a]
    #     a_value = action_values[a]
    #     if a_nvisits != 0:
    #         a_var = max(action_vars[a], hparam_haver_var/a_nvisits)
    #         # a_var = action_vars[a] if action_vars[a] != 0 else hparam_haver_var
            
    #         gam_log = (num_actions_visited*total_nvisits/a_nvisits)**4
    #         a_gam = np.sqrt(a_var)*np.sqrt(18/a_nvisits*np.log(gam_log))
    #         # a_gam = hparam_haver_var*np.sqrt(18/a_nvisits*np.log(gam_log))
            
    #         a_lcb = a_value - a_gam
    #         if a_lcb > max_lcb:
    #             max_lcb = a_lcb
    #             rhat_idx = a
    #             rhat_gam = a_gam
    #             rhat_nvisits = a_nvisits

    # # print(max_lcb)
    # Bset_idxes = []
    # Bset_muhats = np.zeros(num_actions)
    # Bset_nvisits = np.zeros(num_actions)
    
    # for a in range(num_actions):
    #     a_nvisits = action_nvisits[a]
    #     a_value = action_values[a]
    #     if a_nvisits != 0:
    #         a_var = max(action_vars[a], hparam_haver_var/a_nvisits)
    #         # a_var = action_vars[a] if action_vars[a] != 0 else hparam_haver_var
    #         # a_var = action_vars[a]
            
    #         gam_log = (num_actions_visited*total_nvisits/a_nvisits)**4
    #         a_gam = np.sqrt(a_var)*np.sqrt(18/a_nvisits*np.log(gam_log))
    #         # a_gam = hparam_haver_var*np.sqrt(18/a_nvisits*np.log(gam_log))

    #         # if a_value >= max_lcb and a_gam <= 3.0/2*rhat_gam:
    #         if a_value >= max_lcb and a_nvisits >= 4.0/9*rhat_nvisits:
    #         # if a_value >= max_lcb:
    #             Bset_muhats[a] = a_value
    #             Bset_nvisits[a] = a_nvisits
    #             # Bset_nvisits[a] = a_nvisits
    #             Bset_idxes.append(a)

    # Bset_probs = Bset_nvisits/np.sum(Bset_nvisits)
    # haver_est = np.dot(Bset_muhats, Bset_probs)

    # if debug:
    #     logging.info(f"action.nvisits = {action_nvisits}")
    #     logging.info(f"action_values = {action_values}")
    #     logging.info(f"action_vars = {action_vars}")
    #     logging.info(f"action_vs = {action_vs}")
    #     logging.info(f"max_lcb = {max_lcb:0.4f}")
    #     # logging.warn(f"Bset_idxes = {Bset_idxes}")
    #     logging.info(f"Bset_nvisits = {Bset_nvisits}")
    #     logging.info(f"Bset_probs = {Bset_probs}")
    #     logging.info(f"Bset_muhats = {Bset_muhats}")
    #     # loging.warn(tmp)
    #     logging.info(f"haver_est = {haver_est:.2f}")
    

    return haver_est
    

def run_mcts_trial(env, simulator, Q_vit, i_trial, args):

    np.random.seed(1000+i_trial)
    random.seed(1000+i_trial)
    state, info = env.reset(seed=1000+i_trial)
    
    # run trials
    mcts = MCTS(simulator, Q_vit, args)

    ep_reward = 0
    for i_step in range(args["ep_max_steps"]):
        logging.warn(f"\n-> i_step={i_step}")
        action = mcts.run(state)
        action_t = action % 4
        next_state, reward, terminated, truncated, info = env.step(action_t)
        ep_reward += reward
        logging.warn(f"state, action, next_state, terminated = {state, action, next_state, terminated}")
        logging.warn(f"Q[state] = {mcts.Q[state]}")
        logging.warn(f"QH[state] = {mcts.QH[state]}")
        # logging.warn(f"QM[state] = {mcts.QM[state]}")
        # logging.warn(f"Qvit[state] = {Q_vit[state]}")

        # logging.warn(f"NH[state] = {mcts.NH[state]}")

        if terminated:
            break

        state = next_state

    return mcts.Q, ep_reward


