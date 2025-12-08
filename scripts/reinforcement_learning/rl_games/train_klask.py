# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RL-Games."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RL-Games.")
parser.add_argument(
    "--video", action="store_true", default=False, help="Record videos during training."
)
parser.add_argument(
    "--video_length",
    type=int,
    default=200,
    help="Length of the recorded video (in steps).",
)
parser.add_argument(
    "--video_interval",
    type=int,
    default=2000,
    help="Interval between video recordings (in steps).",
)
parser.add_argument(
    "--num_envs", type=int, default=None, help="Number of environments to simulate."
)
parser.add_argument(
    "--task", type=str, default="Isaac-Klask-v0", help="Name of the task."
)
parser.add_argument(
    "--seed", type=int, default=None, help="Seed used for the environment"
)
parser.add_argument(
    "--distributed",
    action="store_true",
    default=False,
    help="Run training with multiple GPUs or nodes.",
)
parser.add_argument(
    "--checkpoint", type=str, default=None, help="Path to model checkpoint."
)
parser.add_argument(
    "--sigma", type=str, default=None, help="The policy's initial standard deviation."
)
parser.add_argument(
    "--max_iterations", type=int, default=None, help="RL Policy training iterations."
)

parser.add_argument(
    "--config",
    type=str,
    default=None,
    help="config.yaml file, rl_games_cfg_entry_point used when not provided.",
)
parser.add_argument(
    "--full_experiment_name",
    type=str,
    default=None,
    help="Experiment name used for logs.",
)
parser.add_argument(
    "--wandb-project-name", type=str, default=None, help="the wandb's project name"
)
parser.add_argument(
    "--wandb-entity",
    type=str,
    default=None,
    help="the entity (team) of wandb's project",
)
parser.add_argument("--training_curriculum", action="store_true", default=False)
parser.add_argument(
    "--mode", type=int, default=None, help="mode for training curriculum"
)
parser.add_argument(
    "--project_folder", type=str, default=None, help="mode for training curriculum"
)
parser.add_argument(
    "--fixed_opponent_checkpoint", 
    type=str, 
    default=None, 
    help="Path to fixed opponent checkpoint (for MODE 3)"
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import math
import os
import random
import time
from datetime import datetime

import gymnasium as gym
import isaaclab_tasks  # noqa: F401
import yaml
from isaaclab.envs import (
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
from isaaclab_tasks.manager_based.klask.actuator_model import ActuatorModelWrapper
from isaaclab_tasks.manager_based.klask.config import KLASK_PARAMS
from isaaclab_tasks.manager_based.klask.env_wrapper import (
    ActionHistoryWrapper,
    KlaskCollisionAvoidanceWrapper,
    KlaskRandomOpponentWrapper,
    OpponentObservationWrapper,
    RlGamesGpuEnvSelfPlay,
)
from isaaclab_tasks.utils.hydra import hydra_task_config
from klask_rl_games import KlaskAlgoObserver, KlaskRunner
from rl_games.common import env_configurations, vecenv
from utils import set_terminations


@hydra_task_config(args_cli.task, "rl_games_cfg_entry_point")
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict
):
    """Train with RL-Games agent."""
    # override configurations with non-hydra CLI arguments
    if args_cli.config is not None:
        with open(args_cli.config, "r") as file:
            config = yaml.safe_load(file)
        agent_cfg.update(config)
    env_cfg.scene.num_envs = (
        args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    )
    env_cfg.sim.device = (
        args_cli.device if args_cli.device is not None else env_cfg.sim.device
    )
    if args_cli.full_experiment_name is not None:
        agent_cfg["params"]["config"]["full_experiment_name"] = (
            args_cli.full_experiment_name
        )

    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    agent_cfg["params"]["seed"] = (
        args_cli.seed if args_cli.seed is not None else agent_cfg["params"]["seed"]
    )
    agent_cfg["params"]["config"]["max_epochs"] = (
        args_cli.max_iterations
        if args_cli.max_iterations is not None
        else agent_cfg["params"]["config"]["max_epochs"]
    )
    if args_cli.checkpoint is not None:
        resume_path = retrieve_file_path(args_cli.checkpoint)
        agent_cfg["params"]["load_checkpoint"] = True
        agent_cfg["params"]["load_path"] = resume_path
        print(
            f"[INFO]: Loading model checkpoint from: {agent_cfg['params']['load_path']}"
        )
    train_sigma = float(args_cli.sigma) if args_cli.sigma is not None else None

    # multi-gpu training config
    if args_cli.distributed:
        agent_cfg["params"]["seed"] += app_launcher.global_rank
        agent_cfg["params"]["config"]["device"] = f"cuda:{app_launcher.local_rank}"
        agent_cfg["params"]["config"]["device_name"] = f"cuda:{app_launcher.local_rank}"
        agent_cfg["params"]["config"]["multi_gpu"] = True
        # update env config device
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
    else:
        # SINGLE-GPU: Ensure device is set from CLI
        device = args_cli.device if args_cli.device is not None else "cuda:0"
        agent_cfg["params"]["config"]["device"] = device
        agent_cfg["params"]["config"]["device_name"] = device
        env_cfg.sim.device = device

    # set the environment seed (after multi-gpu config for updated rank from agent seed)
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg["params"]["seed"]

    # specify directory for logging experiments
    log_root_path = os.path.join(
        "logs", "rl_games", agent_cfg["params"]["config"]["name"]
    )
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs
    log_dir = agent_cfg["params"]["config"].get(
        "full_experiment_name", datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    # set directory into agent config
    # logging directory path: <train_dir>/<full_experiment_name>
    agent_cfg["params"]["config"]["train_dir"] = log_root_path
    agent_cfg["params"]["config"]["full_experiment_name"] = log_dir

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_root_path, log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_root_path, log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_root_path, log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_root_path, log_dir, "params", "agent.pkl"), agent_cfg)

    # read configurations about the agent-training
    rl_device = agent_cfg["params"]["config"]["device"]

    # ADD DEBUG PRINT BEFORE ENV CREATION
    print(f"[INFO] 🔍 Device configuration:")
    print(f"   env_cfg.sim.device: {env_cfg.sim.device}")
    print(f"   agent device: {agent_cfg['params']['config']['device']}")
    print(f"   rl_device: {rl_device}")

    # create isaac environment
    env = gym.make(
        args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None
    )
    obs, info = env.reset()
    print("type(obs):", type(obs))
    print("Obs keys:", obs.keys())              # since it's a Dict space
    print("policy obs shape:", obs["policy"].shape)
    print("opponent obs shape:", obs["opponent"].shape)
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_root_path, log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    print(agent_cfg.keys())
    if agent_cfg["env"].get("actuator_model", True):
        env = ActuatorModelWrapper(env)
        
    # Get clip values from config
    clip_obs = agent_cfg["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["env"].get("clip_actions", 0.2)  # ADD THIS LINE

    if agent_cfg["env"].get("collision_avoidance", True):
        env = KlaskCollisionAvoidanceWrapper(env, max_vel=clip_actions)

    if KLASK_PARAMS["observations"]["action_history"] > 0:
        env = ActionHistoryWrapper(
            env, history_length=KLASK_PARAMS["observations"]["action_history"]
        )

    # if self-play, use opponent observation wrapper to get access to opponent player's observations:
    if agent_cfg["params"]["config"].get("self_play", False):
        env = OpponentObservationWrapper(env)
        print("[DEBUG] Applied OpponentObservationWrapper for self-play")

    # if no self-play, pick random actions for the opponent:
    else:
        env = KlaskRandomOpponentWrapper(env)
        print("[DEBUG] Applied KlaskRandomOpponentWrapper")

    # wrap around environment for rl-games
    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions)

        # DEBUG: Print final action space seen by RL-Games
    print("[DEBUG train_klask] Final env action space before runner:")
    print(f"  env.action_space: {env.action_space}")
    print(f"  env.action_space.shape: {env.action_space.shape}")

    # register the environment to rl-games registry
    # note: in agents configuration: environment name must be "rlgpu"
    if agent_cfg["params"]["config"].get("self_play", False):
        vecenv.register(
            "IsaacRlgWrapper",
            lambda config_name, num_actors, **kwargs: RlGamesGpuEnvSelfPlay(
                config_name,
                num_actors,
                agent_cfg.copy(), 
                training_curriculum=args_cli.training_curriculum,
                mode=args_cli.mode,
                folder=args_cli.project_folder,
                fixed_opponent_checkpoint=args_cli.fixed_opponent_checkpoint,  # NEW
                **kwargs,
            ),
        )
        env_configurations.register(
            "rlgpu",
            {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env},
        )

    else:
        vecenv.register(
            "IsaacRlgWrapper",
            lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(
                config_name, num_actors, **kwargs
            ),
        )
        env_configurations.register(
            "rlgpu",
            {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env},
        )


    # set active termination terms specified in agent_cfg:
    if "terminations" in agent_cfg.keys():
        set_terminations(env, agent_cfg["terminations"])

    # set number of actors into agent config
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
    # create runner from rl-games
    runner = KlaskRunner(KlaskAlgoObserver())


        # CRITICAL FIX: Set env_info before runner.load()
    #from rl_games.common.env_configurations import get_env_info
    #env_info = get_env_info(env)
    #agent_cfg["params"]["config"]["env_info"] = env_info

        # DEBUG: Check what's in agent_cfg before load
    print("[DEBUG] agent_cfg['params']['network']['space']:", agent_cfg['params']['network']['space'])
    print("[DEBUG] agent_cfg['params']['config'].get('env_info'):", agent_cfg['params']['config'].get('env_info'))
    
    runner.load(agent_cfg)

    # create complete config and log to wandb:
    if "env" in agent_cfg.keys():
        agent_cfg["env"].update(KLASK_PARAMS)
    else:
        agent_cfg["env"] = KLASK_PARAMS

    if args_cli.wandb_project_name is not None:
        import wandb

        config = {"agent": agent_cfg, "env": env_cfg.to_dict()}
        wandb.init(
            project=args_cli.wandb_project_name,
            entity=args_cli.wandb_entity,
            sync_tensorboard=True,
            config=config,
            monitor_gym=True,
            save_code=True,
        )
        # --- DEBUG: inspect rl_games network before training ---
    player = runner.create_player()

    print("=== RL-Games Player model ===")
    print(player.model)

    import torch
    net = player.model.a2c_network    # <- MOST LIKELY correct

    print("=== A2C network ===")
    print(net)

    print("=== State dict keys & shapes ===")
    for k, v in net.state_dict().items():
        print(k, v.shape)
    # --- END DEBUG ---


    # reset the agent and env
    runner.reset()
    start_time = time.time()
    # train the agent
    if args_cli.checkpoint is not None:
        runner.run(
            {
                "train": True,
                "play": False,
                "sigma": train_sigma,
                "checkpoint": resume_path,
            }
        )
    else:
        runner.run({"train": True, "play": False, "sigma": train_sigma})
    print(f"Total training time: {time.time() - start_time}")

    # log model checkpoint to v:
    if args_cli.wandb_project_name is not None:
        model = wandb.Artifact("model", type="model")
        model.add_file(
            os.path.join(
                log_root_path,
                log_dir,
                "nn",
                f"{agent_cfg['params']['config']['name']}.pth",
            )
        )
        wandb.log_artifact(model)
        wandb.finish()

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
