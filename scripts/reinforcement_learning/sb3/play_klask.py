# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from Stable-Baselines3."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from Stable-Baselines3.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument(
    "--use_last_checkpoint",
    action="store_true",
    help="When no checkpoint provided, use the last saved model. Otherwise use the best saved model.",
)
parser.add_argument("--opponent", type=str, default="random", help="Opponent player (random or agent)")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import numpy as np
import os
import torch
import matplotlib.pyplot as plt
import time

from stable_baselines3 import PPO, SAC
#from sbx import PPO, SAC
from stable_baselines3.common.vec_env import VecNormalize

from isaaclab.utils.dict import print_dict

import isaac_lab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import get_checkpoint_path, load_cfg_from_registry, parse_env_cfg
from isaaclab_rl.sb3 import Sb3VecEnvWrapper, process_sb3_cfg

from isaaclab_tasks.manager_based.klask import (
    KlaskEnvCfg,
    KlaskGoalEnvWrapper, 
    KlaskSimpleEnvWrapper,
    KlaskRandomOpponentWrapper,
    OpponentObservationWrapper,
    KlaskSb3VecEnvWrapper, 
    CurriculumWrapper,
    SubtaskHerReplayBuffer, 
    TwoPlayerPPO,
    TwoPlayerSAC
)

available_algorithms = {
    "PPO": PPO, 
    "SAC": SAC, 
    "TwoPlayerPPO": TwoPlayerPPO, 
    "TwoPlayerSAC": TwoPlayerSAC
}


def main():
    """Play with stable-baselines agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg = load_cfg_from_registry(args_cli.task, "sb3_cfg_entry_point")

    # directory for logging into
    log_root_path = os.path.join("logs", "sb3", args_cli.task)
    log_root_path = os.path.abspath(log_root_path)
    # check checkpoint is valid
    if args_cli.checkpoint is None:
        if args_cli.use_last_checkpoint:
            checkpoint = "model_.*.zip"
        else:
            checkpoint = "model.zip"
        checkpoint_path = get_checkpoint_path(log_root_path, ".*", checkpoint)
    else:
        checkpoint_path = args_cli.checkpoint
    log_dir = os.path.dirname(checkpoint_path)

    # post-process agent configuration
    agent_cfg = process_sb3_cfg(agent_cfg)

    algorithm = agent_cfg.pop("algorithm")
    algorithm = available_algorithms[algorithm]
    use_her = agent_cfg.pop("her")

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    # wrap around environment for stable baselines
    if use_her:
        env = KlaskGoalEnvWrapper(env)
    if args_cli.opponent == "random":
        env = KlaskRandomOpponentWrapper(env)
    elif args_cli.opponent == "agent":
        env = OpponentObservationWrapper(env)
    if "rewards" in agent_cfg.keys():
        env = CurriculumWrapper(env, agent_cfg["rewards"])
    if use_her:
        env = KlaskSb3VecEnvWrapper(env)
    else:
        env = Sb3VecEnvWrapper(env)

    # normalize environment (if needed)
    if "normalize_input" in agent_cfg:
        env = VecNormalize(
            env,
            training=True,
            norm_obs="normalize_input" in agent_cfg and agent_cfg.pop("normalize_input"),
            norm_obs_keys=["observation", "desired_goal", "achieved_goal"],
            norm_reward="normalize_value" in agent_cfg and agent_cfg.pop("normalize_value"),
            clip_obs="clip_obs" in agent_cfg and agent_cfg.pop("clip_obs"),
            gamma=agent_cfg["gamma"],
            clip_reward=np.inf,
        )

    # create agent from stable baselines
    print(f"Loading checkpoint from: {checkpoint_path}")
    agent = algorithm.load(checkpoint_path, env, print_system_info=True)
    print(agent.policy)

    # reset environment
    rewards = []
    obs = env.reset()
    timestep = 0
    start_time = time.time()
    # simulate environment
    while simulation_app.is_running() and time.time() - start_time < 100.0:
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            if args_cli.opponent == "agent":
                # obs_opponent = obs.copy()
                # if use_her:
                #     obs_opponent["observation"] *= -1
                #     obs_opponent["achieved_goal"] *= -1
                # else:
                #     obs_opponent *= -1.0
                actions_player, _ = agent.predict(obs["player"], deterministic=True)
                actions_opponent, _ = agent.predict(obs["opponent"], deterministic=True)
                #actions_opponent = agent.get_bootstrap_action(obs_opponent, player="opponent")
                actions = np.concatenate((actions_player[:, :2], -1.0 * actions_opponent[:, :2]), axis=-1)
            else:
                actions, _ = agent.predict(obs, deterministic=True)
            #print(actions_player[0, :])
            # env stepping
            obs, rew, _, _ = env.step(actions)
            rewards.append(rew)

        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

    # close the simulator
    env.close()
    plt.plot(rewards)
    plt.show()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
