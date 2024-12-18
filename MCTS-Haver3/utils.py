
import random
import numpy as np
np.set_printoptions(precision=2, suppress=True)

from mcts_haver import run_mcts_trial
import multiprocess as mp


class MultiProcess():
    def __init__(self):
        self.pool = mp.Pool()

    def callback(self, result):
        if result < -6:
            # print(f"ep_reward = {result:0.2f} is too small")
            self.pool.terminate()

    def run(self, func, args_list):
        # ep_reward_list = []
        # Q_mcts_list = []
        for args in args_list:
            try:
                async_process = self.pool.apply_async(
                    func, args=args, callback=self.callback)
                # ep_reward, Q_mcts = async_process.get()
                # ep_reward_list.append(ep_reward)
                # Q_mcts_list.append(Q_mcts)
            except ValueError:
                pass

        self.pool.close()
        self.pool.join()
        return 
