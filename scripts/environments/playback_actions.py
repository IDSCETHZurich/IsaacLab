# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run an environment with zero action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Zero agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument(
    "--num_envs", type=int, default=1, help="Number of environments to simulate."
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--file", type=str, help="File containing action and observation trajectories."
)
parser.add_argument(
    "--video", action="store_true", default=False, help="Record videos during training."
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os

import gymnasium as gym
import isaaclab_tasks  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import torch
from isaaclab.utils.dict import print_dict
from isaaclab_tasks.manager_based.klask.actuator_model import ActuatorModelWrapper
from isaaclab_tasks.utils import parse_env_cfg
from tqdm import tqdm


def initialize_sim_state(env_cfg, ball_state, peg_1_state, peg_2_state):
    env_cfg.events.reset_x_position_peg_1.params["position_range"] = (
        peg_1_state[0, 0],
        peg_1_state[0, 0],
    )
    env_cfg.events.reset_y_position_peg_1.params["position_range"] = (
        peg_1_state[0, 1],
        peg_1_state[0, 1],
    )
    env_cfg.events.reset_x_position_peg_2.params["position_range"] = (
        peg_2_state[0, 0],
        peg_2_state[0, 0],
    )
    env_cfg.events.reset_y_position_peg_2.params["position_range"] = (
        peg_2_state[0, 1],
        peg_2_state[0, 1],
    )
    env_cfg.events.reset_ball_position.params["pose_range"]["x"] = (
        ball_state[0, 0],
        ball_state[0, 0],
    )
    env_cfg.events.reset_ball_position.params["pose_range"]["y"] = (
        ball_state[0, 1],
        ball_state[0, 1],
    )

    env_cfg.events.reset_x_position_peg_1.params["velocity_range"] = (
        peg_1_state[0, 2],
        peg_1_state[0, 2],
    )
    env_cfg.events.reset_y_position_peg_1.params["velocity_range"] = (
        peg_1_state[0, 3],
        peg_1_state[0, 3],
    )
    env_cfg.events.reset_x_position_peg_2.params["velocity_range"] = (
        peg_2_state[0, 2],
        peg_2_state[0, 2],
    )
    env_cfg.events.reset_y_position_peg_2.params["velocity_range"] = (
        peg_2_state[0, 3],
        peg_2_state[0, 3],
    )
    env_cfg.events.reset_ball_position.params["velocity_range"]["x"] = (
        ball_state[0, 2],
        ball_state[0, 2],
    )
    env_cfg.events.reset_ball_position.params["velocity_range"]["y"] = (
        ball_state[0, 3],
        ball_state[0, 3],
    )


def main():
    """Zero actions agent with Isaac Lab environment."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )

    trajectories = np.load(args_cli.file)
    peg_1_state = trajectories["obs"][:, 0, :4]
    peg_2_state = trajectories["obs"][:, 0, 4:8]
    ball_state = trajectories["obs"][:, 0, 8:12]
    print(peg_1_state)
    print(peg_2_state)
    initialize_sim_state(env_cfg, ball_state, peg_1_state, peg_2_state)

    # create environment
    env = gym.make(
        args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None
    )

    log_dir = os.path.join("logs", "playback_actions")
    if args_cli.video:
        video_kwargs = {
            "video_folder": log_dir,
            "step_trigger": lambda step: step == 0,
            "video_length": len(trajectories["actions"]),
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = ActuatorModelWrapper(env)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment
    obs, _ = env.reset()
    # simulate environment
    obs_buffer = []
    for actions in tqdm(trajectories["actions"]):
        # run everything in inference mode
        with torch.inference_mode():
            actions = torch.from_numpy(actions).to(env.unwrapped.device)
            obs_buffer.append(obs["policy"][0].cpu().numpy())
            # apply actions
            obs, _, _, _, _ = env.step(actions.unsqueeze(0))

    actions = trajectories["actions"]
    observations_real = trajectories["obs"]
    observations_sim = np.array(obs_buffer)
    np.savez(
        os.path.join(log_dir, "sim2real_trajectories"),
        actions=actions,
        observations_real=observations_real,
        observations_sim=observations_sim,
    )

    fig, ax = plt.subplots(4)
    ax[0].plot(observations_real[:, 0, 0], label="Peg 1 x pos real")
    ax[1].plot(observations_real[:, 0, 1], label="Peg 1 y pos real")
    ax[0].plot(observations_sim[:, 0], label="Peg 1 x pos sim")
    ax[1].plot(observations_sim[:, 1], label="Peg 1 y pos sim")
    ax[2].plot(observations_real[:, 0, 2], label="Peg 1 x vel real")
    ax[3].plot(observations_real[:, 0, 3], label="Peg 1 y vel real")
    ax[2].plot(observations_sim[:, 2], label="Peg 1 x vel sim")
    ax[3].plot(observations_sim[:, 3], label="Peg 1 y vel sim")
    ax[0].legend()
    ax[1].legend()
    ax[2].legend()
    ax[3].legend()
    plt.show()

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
