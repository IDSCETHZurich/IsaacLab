import gymnasium as gym

from . import agents
from .klask_env_cfg import KlaskEnvCfg, KlaskGoalEnvCfg
from .klask_env_wrapper import *
from .actuator_model import ActuatorModelWrapper

##
# Register Gym environment.
##

gym.register(
    id="Isaac-Klask-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": KlaskEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml"
    },
)

