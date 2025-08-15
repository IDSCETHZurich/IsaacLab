import subprocess
import time
import os
from pathlib import Path
import json
import copy
import re

# List of config files to train with
configs = [
    "planned_runs/training_curriculum/klask_config_1.yaml",
    "planned_runs/training_curriculum/klask_config_2.yaml",
    "planned_runs/training_curriculum/klask_config_3.yaml",
    "planned_runs/training_curriculum/klask_config_4.yaml"
]

checkpoint = ["/home/student/klask_rl/IsaacLab/logs/rl_games/klask/pretrained_agent_action_1.0/nn/last_klask_ep_70_rew_2.950554.pth",
              "/home/student/klask_rl/IsaacLab/logs/rl_games/klask/pretrained_agent_action_1.0/nn/last_klask_ep_70_rew_2.950554.pth",
              "/home/student/klask_rl/IsaacLab/logs/rl_games/klask/pretrained_agent_action_1.0/nn/last_klask_ep_35_rew_3.9405801.pth",
              "/home/student/klask_rl/IsaacLab/logs/rl_games/klask/pretrained_agent_action_1.0/nn/last_klask_ep_35_rew_3.9405801.pth"
]

# Python executable and training script
PYTHON = "python3"
TRAIN_SCRIPT = "scripts/reinforcement_learning/rl_games/train_klask.py"  # your main train script


MODE = 2 #either 0 --> opponent is chosen from a pool of players and periodically changed or 1 --> always the best opponent across 4 instances is chosen
# Store process handles
processes = []

project_folder = Path("/home/student/klask_rl/IsaacLab/logs/rl_games/klask/training_curriculum") 

# Start training with different config files
for i, cfg in enumerate(configs):
    print(f"Launching training #{i+1} with config: {cfg}")
    cmd = [
        PYTHON, TRAIN_SCRIPT,
        "--config", cfg,
        "--device", f"cuda:{i}",
        "--headless",
        "--num_envs", "4096",
        "--checkpoint", checkpoint[i],
        "--wandb-project-name", f"Training_curriculum_agent_{i+1}",
        "--training_curriculum", "--mode", str(MODE),
        "--project_folder", str(project_folder / f"agent_{i+1}/nn"),

    ]

    # Start process without redirecting output
    proc = subprocess.Popen(cmd)

    processes.append(proc)


# Ensure all complete
for proc in processes:
    proc.wait()



