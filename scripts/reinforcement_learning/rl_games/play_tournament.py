# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RL-Games."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a tournament between RL agents from RL-Games.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Klask-v0", help="Name of the task.")
parser.add_argument("--dir", type=str, default=None, help="Path to tournament directory containing checkpoints and config directories.")
parser.add_argument("--config", type=str, default=None, help="config.yaml file, rl_games_cfg_entry_point used when not provided")
parser.add_argument("--num_rounds", type=int, default=10, help="Number of rounds in the tournament.")
parser.add_argument("--num_games_per_round", type=int, default=100, help="Number of games per round.")
parser.add_argument("--tournament_name", type=str, default="tournament", help="Name used for logging videos.")


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
import math
import os
import torch
import yaml
import time
import matplotlib.pyplot as plt
import itertools
import numpy as np

from rl_games.common import env_configurations, vecenv
from rl_games.common.player import BasePlayer
from rl_games.torch_runner import Runner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, load_cfg_from_registry, parse_env_cfg
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

from isaaclab_tasks.manager_based.klask import (
    OpponentObservationWrapper,
    CurriculumWrapper,
    RlGamesGpuEnvSelfPlay,
    ObservationNoiseWrapper,
    find_wrapper
)
from isaaclab_tasks.manager_based.klask.utils_manager_based import set_terminations


def update_elo(p1, p2, score_1, score_2, k=10.0):
    elo_1 = p1.elo + k * (score_1 - 1 / (1 + 10 ** ((p2.elo - p1.elo) / 400)))
    elo_2 = p2.elo + k * (score_2 - 1 / (1 + 10 ** ((p1.elo - p2.elo) / 400)))
    p1.elo = elo_1.item()
    p2.elo = elo_2.item()


def compute_scores(info):
    average_score_1 = (info["episode"]["Episode_Termination/goal_scored"]
                     + info["episode"]["Episode_Termination/opponent_in_goal"]
                     + 0.5 * info["episode"]["Episode_Termination/time_out"])
    average_score_2 = (info["episode"]["Episode_Termination/goal_conceded"]
                     + info["episode"]["Episode_Termination/player_in_goal"]
                     + 0.5 * info["episode"]["Episode_Termination/time_out"])
    return average_score_1, average_score_2


def main():
    """Play with RL-Games agent."""
    # parse env configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg = load_cfg_from_registry(args_cli.task, "rl_games_cfg_entry_point")

    if args_cli.config is not None:
        with open(args_cli.config, 'r') as file:
            config = yaml.safe_load(file)
        agent_cfg.update(config)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rl_games", agent_cfg["params"]["config"]["name"])
    log_root_path = os.path.abspath(log_root_path)
    log_dir = os.path.dirname(args_cli.tournament_name)

    # wrap around environment for rl-games
    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
    
    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_root_path, log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    obs_noise = agent_cfg["env"].get("obs_noise", 0.0)
    env = ObservationNoiseWrapper(env, obs_noise)
    env = OpponentObservationWrapper(env)
    if "rewards" in agent_cfg.keys():
        env = CurriculumWrapper(env, agent_cfg["rewards"], mode="test")
    
    # wrap around environment for rl-games
    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions)

    # register the environment to rl-games registry
    # note: in agents configuration: environment name must be "rlgpu"
    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnvSelfPlay(
            config_name, num_actors, agent_cfg.copy(), **kwargs)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})
 
    # set active termination terms specified in agent_cfg:
    if "terminations" in agent_cfg.keys():
        set_terminations(env, agent_cfg["terminations"])
    
    # set number of actors into agent config
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
    
    # reset environment
    obs = env.reset()
    timestep = 0
    
    # create runner from rl-games
    runner = Runner()
    players = []
    checkpoints_dir = os.path.join(args_cli.dir, "checkpoints")
    for i, model_file in enumerate(os.listdir(checkpoints_dir)):
        player_name = model_file.split(".")[0]
        with open(os.path.join(args_cli.dir, "configs", f"{player_name}.yaml"), 'r') as file:
            config = yaml.safe_load(file)
        runner.load(config)
        player = runner.create_player()
        player.restore(os.path.join(checkpoints_dir, model_file))
        player.reset()
        # initialize RNN states if used
        if player.is_rnn:
            player.init_rnn()
        _ = player.get_batch_size(obs, 1)
        player.elo = 1200.0
        player.name = player_name
        players.append(player)

    player_elos = []
            
    # simulate environment
    # note: We simplified the logic in rl-games player.py (:func:`BasePlayer.run()`) function in an
    #   attempt to have complete control over environment stepping. However, this removes other
    #   operations such as masking that is used for multi-agent learning by RL-Games.
    for i, _ in enumerate(range(args_cli.num_rounds)):
        for p1, p2 in itertools.combinations(players, 2):
            num_games = 0
            average_score_1 = 0.0
            average_score_2 = 0.0
            while num_games < args_cli.num_games_per_round:
                # run everything in inference mode
                with torch.inference_mode():
                    # convert obs to agent format
                    obs_1 = p1.obs_to_torch(obs)
                    opponent_obs = find_wrapper(env, OpponentObservationWrapper).opponent_obs
                    obs_2 = p2.obs_to_torch(opponent_obs)
                    # agent stepping
                    actions = p1.get_action(obs_1, is_deterministic=True)
                    actions[:, 2:] = -p2.get_action(obs_2, is_deterministic=True)[:, :2]
                    # env stepping
                    obs, rew, dones, info = env.step(actions)
                    # perform operations for terminated episodes
                    if len(dones) > 0:
                        num_games += dones.sum()
                        score_1, score_2 = compute_scores(info)
                        average_score_1 += score_1
                        average_score_2 += score_2
                        # reset rnn state for terminated episodes
                        if p1.is_rnn and p1.states is not None:
                            for s in p1.states:
                                s[:, dones, :] = 0.0
                        if p2.is_rnn and p2.states is not None:
                            for s in p2.states:
                                s[:, dones, :] = 0.0
                if args_cli.video:
                    timestep += 1
                    # Exit the play loop after recording one video
                    if timestep == args_cli.video_length:
                        break
            
            update_elo(p1, p2, average_score_1 / num_games, average_score_2 / num_games)
            player_elos.append([p.elo for p in players])
        
        print(f"Round {i} completed. ELO scores:")
        for p in players:
            print(f"{p.name}: {p.elo}")

    # Plot ELO scores:
    player_elos = np.array(player_elos)
    for i, p in enumerate(players):
        plt.plot(player_elos[:, i], label=p.name)
    plt.xlabel("game")
    plt.ylabel("ELO")
    plt.legend()
    plt.grid()
    plt.show()

    # close the simulator
    env.close()



if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
