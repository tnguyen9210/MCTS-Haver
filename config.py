
import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    # General
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--env_id', type=str, default="FrozenLake-v1")
    parser.add_argument('--map_name', type=str, default="4x4")
    parser.add_argument('--is_state_slippery', type=bool, default=False)
    parser.add_argument('--is_slippery', type=bool, default=False)
    parser.add_argument('--slippery_mode', type=str, default="extreme")
    parser.add_argument('--render_mode', type=str, default="")

    # 
    parser.add_argument('--update_method', type=str, default="avg")
    
    # General params
    parser.add_argument('--num_trials', type=int, default=500)
    parser.add_argument('--ep_max_steps', type=int, default=20)
    
    parser.add_argument('--mcts_num_trajectories', type=int, default=1500)
    parser.add_argument('--mcts_max_depth', type=int, default=3)
    parser.add_argument('--mcts_rollout_max_depth', type=int, default=100)

    parser.add_argument('--gamma', type=float, default=1.0)
    
    parser.add_argument('--rollout_method', type=str, default="")
    
    #
    parser.add_argument('--action_multi', type=int, default=1)
    parser.add_argument('--hparam_ucb_scale', type=float, default=1.0)
    parser.add_argument('--hparam_haver_var', type=float, default=1.0)

    # value iteration
    parser.add_argument('--vit_thres', type=float, default=0.00001)
    
    args = vars(parser.parse_args(args=[]))
    return args

