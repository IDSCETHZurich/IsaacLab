import subprocess
import time
import os
from pathlib import Path
import json
import copy
import re

# List of config files to train with
configs = [
    "/home/student/klask_rl/IsaacLab/planned_runs/rl_games_self_play_sparse_own_half_horizon128.yaml"
]

checkpoint = [
    "/home/student/ros2_ws/klask_imitation_learning/trained_models/converted_to_pth/pretrained_agent_ilppo_from_sb3.pth",
    "/home/student/ros2_ws/klask_imitation_learning/trained_models/converted_to_pth/pretrained_agent_ilppo_from_sb3.pth",
    "/home/student/ros2_ws/klask_imitation_learning/trained_models/converted_to_pth/pretrained_agent_ilppo_from_sb3.pth",
    "/home/student/ros2_ws/klask_imitation_learning/trained_models/converted_to_pth/pretrained_agent_ilppo_from_sb3.pth",
]

# Python executable and training script
PYTHON = "python3"
TRAIN_SCRIPT = "scripts/reinforcement_learning/rl_games/train_klask.py"  # your main train script


MODE = 0 #either 0 --> opponent is chosen from a pool of players and periodically changed or 1 --> always the best opponent across 4 instances is chosen
# Store process handles
processes = []

project_folder = Path("/home/student/klask_rl/IsaacLab/logs/rl_games/klask/training_curriculum") 
num_gpus=1
num_envs_per_gpu = 4096


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



