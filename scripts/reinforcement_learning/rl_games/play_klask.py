# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RL-Games."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Play a checkpoint of an RL agent from RL-Games."
)
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
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument(
    "--num_envs", type=int, default=1, help="Number of environments to simulate."
)
parser.add_argument(
    "--task", type=str, default="Isaac-Klask-v0", help="Name of the task."
)
parser.add_argument(
    "--checkpoint", type=str, default="/home/student/klask_rl/IsaacLab/logs/rl_games/klask/self_play_sparse_own_half_horizon128/nn/last_klask_ep_50_rew_0.67127305.pth", help="Path to model checkpoint."
)
parser.add_argument(
    "--use_last_checkpoint",
    action="store_true",
    help="When no checkpoint provided, use the last saved model. Otherwise use the best saved model.",
)
parser.add_argument(
    "--config",
    type=str,
    default="/home/student/klask_rl/IsaacLab/planned_runs/rl_games_self_play_sparse_own_half_horizon128.yaml",
    help="config.yaml file, rl_games_cfg_entry_point used when not provided",
)


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

import math
import os
import time

import gymnasium as gym
import isaaclab_tasks  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
from isaaclab_tasks.manager_based.klask.actuator_model import ActuatorModelWrapper
from isaaclab_tasks.manager_based.klask.config import KLASK_PARAMS
from isaaclab_tasks.manager_based.klask.env_wrapper import (
    ActionHistoryWrapper,
    KlaskAgentOpponentWrapper,
    KlaskCollisionAvoidanceWrapper,
    KlaskRandomOpponentWrapper,
    RlGamesGpuEnvSelfPlay,
    find_wrapper,
)
from isaaclab_tasks.utils import (
    get_checkpoint_path,
    load_cfg_from_registry,
    parse_env_cfg,
)
from rl_games.common import env_configurations, vecenv
from rl_games.common.player import BasePlayer
from rl_games.torch_runner import Runner
from utils import set_terminations


def main():
    """Play with RL-Games agent."""
    # parse env configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    
    # Load agent config: skip registry if custom config provided
    if args_cli.config is not None:
        print(f"[INFO]: Loading configuration from: {args_cli.config}")
        with open(args_cli.config, "r") as file:
            agent_cfg = yaml.safe_load(file)
    else:
        agent_cfg = load_cfg_from_registry(args_cli.task, "rl_games_cfg_entry_point")

    # Fix: Override num_actors with CLI argument
    if args_cli.num_envs is not None:
        if "params" not in agent_cfg:
            agent_cfg["params"] = {}
        if "config" not in agent_cfg["params"]:
            agent_cfg["params"]["config"] = {}
        agent_cfg["params"]["config"]["num_actors"] = args_cli.num_envs
    
    # Debug
    print(f"[DEBUG] Network units: {agent_cfg.get('params', {}).get('network', {}).get('mlp', {}).get('units', 'NOT SET')}")
    print(f"[DEBUG] Action space: {agent_cfg['params']['network']['space']}")
    print()
    # specify directory for logging experiments
    log_root_path = os.path.join(
        "logs", "rl_games", agent_cfg["params"]["config"]["name"]
    )
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    # find checkpoint
    if args_cli.checkpoint is None:
        # specify directory for logging runs
        run_dir = agent_cfg["params"]["config"].get("full_experiment_name", ".*")
        # specify name of checkpoint
        if args_cli.use_last_checkpoint:
            checkpoint_file = ".*"
        else:
            # this loads the best checkpoint
            checkpoint_file = f"{agent_cfg['params']['config']['name']}.pth"
        # get path to previous checkpoint
        resume_path = get_checkpoint_path(
            log_root_path, run_dir, checkpoint_file, other_dirs=["nn"]
        )
    else:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    log_dir = os.path.dirname(os.path.dirname(resume_path))

    # wrap around environment for rl-games

    rl_device = agent_cfg["params"]["config"]["device"]
    rl_device = args_cli.device

    # Fix: these also need to read from params.env
    env_params = agent_cfg.get("params", {}).get("env", {})
    clip_obs = env_params.get("clip_observations", math.inf)
    clip_actions = env_params.get("clip_actions", math.inf)

    # create isaac environment
    env = gym.make(
        args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None
    )

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

    # Remove duplicate env_params line if you added it earlier
    # env_params = agent_cfg.get("params", {}).get("env", {})
    
    if env_params.get("actuator_model", False):
        env = ActuatorModelWrapper(env, device=args_cli.device)

    if env_params.get("collision_avoidance", False):
        env = KlaskCollisionAvoidanceWrapper(env)

    if KLASK_PARAMS["observations"]["action_history"] > 0:
        env = ActionHistoryWrapper(
            env, history_length=KLASK_PARAMS["observations"]["action_history"]
        )

    obs_noise = env_params.get("obs_noise", 0.0)
    if obs_noise > 0.0:
        env = ObservationNoiseWrapper(env, obs_noise)

    if agent_cfg["params"]["config"].get("self_play", False):
        # env = KlaskAgentOpponentWrapper(env)
        env = env
    else:
        env = KlaskRandomOpponentWrapper(env)

    # wrap around environment for rl-games
    env = RlGamesVecEnvWrapper(
        env, rl_device, clip_obs=clip_obs, clip_actions=clip_actions
    )

    # set active termination terms specified in agent_cfg:
    if "terminations" in agent_cfg.keys():
        set_terminations(env, agent_cfg["terminations"])

    # IMPORTANT: Set num_actors BEFORE registering vecenv
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs

    # register the environment to rl-games registry
    # note: in agents configuration: environment name must be "rlgpu"
    # Use RlGamesGpuEnvSelfPlay for both training and play when self_play=True
    if agent_cfg["params"]["config"].get("self_play", False):
        vecenv.register(
            "IsaacRlgWrapper",
            lambda config_name, num_actors, **kwargs: RlGamesGpuEnvSelfPlay(
                config_name, 
                num_actors, 
                agent_cfg.copy(),  # This captures agent_cfg at registration time
                is_deterministic=True,
                **kwargs
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

    # load previously trained model
    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = resume_path
    print(f"[INFO]: Loading model checkpoint from: {agent_cfg['params']['load_path']}")

    # DON'T set num_actors here - already set above before registration
    # agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
    
    # create runner from rl-games
    runner = Runner()
    runner.load(agent_cfg)
    # obtain the agent from the runner
    agent: BasePlayer = runner.create_player()
    agent.restore(resume_path)
    agent.reset()

    agent.device = torch.device(args_cli.device)
    agent.model.to(args_cli.device)
    agent.actions_low = agent.actions_low.to(args_cli.device)
    agent.actions_high = agent.actions_high.to(args_cli.device)
    # reset environment
    obs = env.reset()
    rewards = []
    if isinstance(obs, dict):
        obs = obs["obs"]
    timestep = 0
    # required: enables the flag for batched observations
    _ = agent.get_batch_size(obs, 1)
    # initialize RNN states if used
    if agent.is_rnn:
        agent.init_rnn()
    '''
    if agent_cfg["params"]["config"].get("self_play", False):
        opponent = runner.create_player()
        opponent.device = torch.device(args_cli.device)
        opponent.model.to(args_cli.device)
        opponent.actions_low = agent.actions_low.to(args_cli.device)
        opponent.actions_high = agent.actions_high.to(args_cli.device)
        opponent.set_weights(agent.get_weights())
        find_wrapper(env, KlaskAgentOpponentWrapper).add_opponent(opponent)
    '''
    # simulate environment
    # note: We simplified the logic in rl-games player.py (:func:`BasePlayer.run()`) function in an
    #   attempt to have complete control over environment stepping. However, this removes other
    #   operations such as masking that is used for multi-agent learning by RL-Games.
    start_time = time.time()
    while simulation_app.is_running() and time.time() - start_time < 1000.0:
        # run everything in inference mode

        with torch.inference_mode():
            # convert obs to agent format
            obs = agent.obs_to_torch(obs)
            # agent stepping
            actions = agent.get_action(obs, is_deterministic=True)
            # env stepping
            obs, rew, dones, _ = env.step(actions)
            rewards.append(rew.detach().cpu())
            # perform operations for terminated episodes
            if len(dones) > 0:
                # reset rnn state for terminated episodes
                if agent.is_rnn and agent.states is not None:
                    for s in agent.states:
                        s[:, dones, :] = 0.0
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

    # close the simulator
    env.close()
    plt.plot(rewards, label="Reward")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
