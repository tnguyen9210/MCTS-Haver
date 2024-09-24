
from collections import defaultdict
import numpy as np
np.set_printoptions(precision=4, suppress=True)

import copy 


def value_iteration(simulator, num_actions, num_states, gamma, threshold):
    V = defaultdict(float)
    Q = defaultdict(lambda: np.zeros(num_actions))
    
    while True:
        delta = 0
        for state in range(num_states):
            old_v = copy.deepcopy(V[state])
            # update
            best_action = None
            max_value = -np.inf
            for action in range(num_actions):
                Q[state][action] = 0
                transitions = simulator.trans_probs[state][action]
                for idx in range(len(transitions)):
                    p, next_state, reward, terminated = transitions[idx]
                    Q[state][action] += p*(reward + gamma*V[next_state])

            V[state] = max(Q[state])
            delta = max(delta, abs(V[state] - old_v))
            
        if delta < threshold:
            break

    # for state in range(num_states):
    #     print(f"\n-> state = {state}")
    #     print(f"V[state] = {V[state]:0.4f}")
    #     for action in range(num_actions):
    #         print(f"Q[state][action] = {Q[state][action]:0.4f}")
    #     print(f"best_action={np.argmax(Q[state])}")

    # stop
    return V, Q
                
                
            # for next_state in range() 
                

