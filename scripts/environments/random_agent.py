# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to an environment with random action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Random agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
import matplotlib.pyplot as plt
import time
import numpy as np

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

from isaaclab_tasks.manager_based.klask import ObservationNoiseWrapper


def main():
    """Random actions agent with Isaac Lab environment."""
    # create environment configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = ObservationNoiseWrapper(env, 0.01)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment
    obs, info = env.reset()

    peg_1_x_pos = []
    peg_1_x_vel = []
    peg_1_body_x_pos = []
    peg_1_body_x_vel = []
    obs_x_pos = []
    obs_x_vel = []
    obs_y_pos = []
    obs_y_vel = []
    resting = []

    start_time = time.time()
    try:
        # simulate environment
        dir = "left"
        stop = False
        stop_counter = 20
        step = 0
        max_steps = 20
        while simulation_app.is_running() and time.time() - start_time < 1000.0 and step < max_steps:
            # run everything in inference mode
            with torch.inference_mode():
                #peg_1_x_pos.append(env.unwrapped.scene.articulations["klask"].data.joint_pos[0, -1].item())
                #peg_1_x_vel.append(env.unwrapped.scene.articulations["klask"].data.joint_vel[0, -1].item())
                #peg_1_body_x_pos.append(env.unwrapped.scene.articulations["klask"].data.body_pos_w[0, -1, 0].item())
                #obs_x_vel.append(env.unwrapped.scene.articulations["klask"].data.body_lin_vel_w[0, -1, 0].item())
                            
                obs_x_pos.append(obs["policy"][0, 0].item())
                obs_x_vel.append(obs["policy"][0, 2].item())
                obs_y_pos.append(obs["policy"][0, 1].item())
                obs_y_vel.append(obs["policy"][0, 3].item())

                # sample actions from -1 to 1
                #actions = 2 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1
                actions = torch.zeros(*env.action_space.shape, device=env.unwrapped.device, dtype=float)
                actions[:, 0] = 1.0
                
                # apply actions
                obs, rew, terminated, truncated, info = env.step(actions)
                step += 1

    finally:
        # close the simulator
        env.close()
        fig, ax = plt.subplots()
        #ax[0].plot(obs_x_pos, label="Obs x")
        #ax[0].plot(obs_y_pos, label="Obs y")

        print("Velocity reached at step:")
        #print(np.argwhere(np.abs(np.array(obs_x_vel) - 1.0 ) < 0.01)[0])
        print(obs_x_vel)

        ax.axhline(y=0, color='red', linestyle='--')      
        ax.plot(obs_x_vel, label="Peg x vel")
        ax.plot(obs_y_vel, label="Peg y vel")
        #ax.plot(obs_y_vel, label="Peg y vel")
        #ax[0].eventplot(events)
        #ax[0].legend()
        ax.legend()
        plt.show()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
