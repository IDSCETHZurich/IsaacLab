import torch, os
from stable_baselines3 import PPO

sb3_path = "/home/student/ros2_ws/klask_imitation_learning/trained_models/klask_airl_20251103_123055_v2/ppo_best_round2.zip"
rlg_path = "/path/to/rlg_init.pth"

model = PPO.load(sb3_path, device="cpu")
sd = model.policy.state_dict()

# Inspect rl_games model to see expected keys
from rl_games.torch_runner import Runner
import yaml

with open("/home/student/klask_rl/IsaacLab/planned_runs/rl_games_self_play_sparse.yaml") as f:
    cfg = yaml.safe_load(f)
runner = Runner()
runner.load(cfg)
player = runner.create_player()

print("\n=== NETWORK STRUCTURE ===")
print(player.network)

print("\n=== STATE DICT KEYS ===")
for k, v in player.network.state_dict().items():
    print(k, v.shape)

target_sd = player.network.state_dict().copy()

# Simple mapping (example; adjust if names differ)
mapping = {
    'mlp_extractor.policy_net.0.weight': 'mlp.0.weight',
    'mlp_extractor.policy_net.0.bias':   'mlp.0.bias',
    'mlp_extractor.policy_net.1.weight': 'mlp.2.weight',
    'mlp_extractor.policy_net.1.bias':   'mlp.2.bias',
    'action_net.weight': 'policy_mu.weight',
    'action_net.bias':   'policy_mu.bias',
    'mlp_extractor.value_net.0.weight':  'critic_mlp.0.weight',
    'mlp_extractor.value_net.0.bias':    'critic_mlp.0.bias',
    'mlp_extractor.value_net.1.weight':  'critic_mlp.2.weight',
    'mlp_extractor.value_net.1.bias':    'critic_mlp.2.bias',
    'value_net.weight': 'value.weight',
    'value_net.bias':   'value.bias',
}

for sb3_k, rlg_k in mapping.items():
    if sb3_k in sd and rlg_k in target_sd and sd[sb3_k].shape == target_sd[rlg_k].shape:
        target_sd[rlg_k] = sd[sb3_k]
    else:
        print("Skip:", sb3_k, "->", rlg_k)

torch.save(target_sd, rlg_path)
print("Wrote converted checkpoint:", rlg_path)