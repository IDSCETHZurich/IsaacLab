import subprocess
from pathlib import Path

# Train DAPG agent against fixed Maurus opponent for 30M timesteps
configs = [
    "/home/student/klask_rl/IsaacLab/planned_runs/rl_games_self_play_sparse_own_half_horizon128.yaml"
]

# Your DAPG starting checkpoint
your_dapg_model = "/home/student/ros2_ws/klask_imitation_learning/trained_models/converted_to_pth/pretrained_agent_ilppo_from_sb3.pth"

# Maurus's fixed opponent
maurus_opponent = "/home/student/klask_rl/IsaacLab/logs/rl_games/klask/pretrained_agent_action_1.0/nn/last_klask_ep_70_rew_2.950554.pth"

checkpoint = [your_dapg_model]  # Only 1 agent training

PYTHON = "python3"
TRAIN_SCRIPT = "scripts/reinforcement_learning/rl_games/train_klask.py"

MODE = 3  # NEW: Fixed opponent mode
processes = []

project_folder = Path("/home/student/klask_rl/IsaacLab/logs/rl_games/klask/dapg_vs_maurus_fixed")

for i, cfg in enumerate(configs):
    print(f"Launching DAPG vs Maurus (fixed opponent)")
    cmd = [
        PYTHON, TRAIN_SCRIPT,
        "--config", cfg,
        "--device", "cuda:0",
        "--headless",
        "--num_envs", "4096",
        "--checkpoint", checkpoint[i],
        "--fixed_opponent_checkpoint", maurus_opponent,  # NEW
        "--wandb-project-name", "DAPG_vs_Maurus_Fixed",
        "--training_curriculum",
        "--mode", str(MODE),
        "--project_folder", str(project_folder / "nn"),
        "--max_iterations", "30000",  # Adjust for 30M timesteps
    ]
    
    proc = subprocess.Popen(cmd)
    processes.append(proc)

for proc in processes:
    proc.wait()

print("\n🎉 Training completed!")



