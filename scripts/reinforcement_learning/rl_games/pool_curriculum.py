import subprocess
import time
import os
from pathlib import Path
import json
import copy
import re
import argparse

parser = argparse.ArgumentParser(description="Pool training")

parser.add_argument("--pool_name", type=str, default="action_1.0", help="Name of the user")
#parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--config", type=str, default=None, help="config.yaml file, rl_games_cfg_entry_point used when not provided.")
parser.add_argument("--opponent_update_steps", type=int, default=1000000, help="Path to model checkpoint.")
args = parser.parse_args()
# List of config files to train with

# Python executable and training script
PYTHON = "python3"
TRAIN_SCRIPT = "scripts/reinforcement_learning/rl_games/train_klask.py"  # your main train script


MODE = 0 #either 0 --> opponent is chosen from a pool of players and periodically changed or 1 --> always the best opponent across 4 instances is chosen
project_folder = Path("/home/student/klask_rl/IsaacLab/logs/rl_games/klask/pool_of_players") / Path(args.pool_name)




# Start training with different config files
cmd = [
    PYTHON, TRAIN_SCRIPT,
    "--config", args.config,
    "--headless",
    "--num_envs", "4096",
    "--wandb-project-name", f"Training_from_pool_{args.pool_name}", 
    "--training_curriculum", "--mode", str(MODE), "--project_folder", project_folder
]

# Start process without redirecting output
proc = subprocess.Popen(cmd)




proc.wait()



