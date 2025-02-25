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

from isaaclab_tasks.manager_based.klask import ObservationNoiseWrapper, ActuatorModelWrapper
from isaaclab.managers import SceneEntityCfg


def main():
    """Random actions agent with Isaac Lab environment."""
    # create environment configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = ActuatorModelWrapper(env)
    #env = ObservationNoiseWrapper(env, 0.01)

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

    start_index = 10000
    batch_size = 100
    data_file = "/home/student/Documents/ActuatorPolicy/actuator_model/data/data_history_10_interval_0.02_delay_0.0_with_states.npz"
    data = np.load(data_file)
    X_commands, Y, Y_prev, commands = data["X_commands"], data["Y"], data["Y_prev"], data["commands"]
    if "X_states" in data.keys():
        X_states = data["X_states"]
    else:
        X_states = None
    command_history = torch.from_numpy(X_commands[start_index-1]).to(env.unwrapped.device)
    X_commands, Y, Y_prev, commands = X_commands[start_index:start_index+batch_size], Y[start_index:start_index+batch_size], Y_prev[start_index:start_index+batch_size], commands[start_index:start_index+batch_size]
    if X_states is not None:
        X_states = X_states[start_index:start_index+batch_size]

    if X_states is not None:
        state_history = torch.from_numpy(X_states[0]).to(env.unwrapped.device)
    
    env.command_buffer_1[0, :] = command_history
    env.state_buffer_1[0, :] = state_history

    start_time = time.time()
    try:
        # simulate environment
        step = 0
        max_steps = batch_size
        while simulation_app.is_running() and time.time() - start_time < 1000.0 and step < max_steps:
            # run everything in inference mode
            with torch.inference_mode():
                #peg_1_x_pos.append(env.unwrapped.scene.articulations["klask"].data.joint_pos[0, -1].item())
                #peg_1_x_vel.append(env.unwrapped.scene.articulations["klask"].data.joint_vel[0, -1].item())
                #peg_1_body_x_pos.append(env.unwrapped.scene.articulations["klask"].data.body_pos_w[0, -1, 0].item())
                #obs_x_vel.append(env.unwrapped.scene.articulations["klask"].data.body_lin_vel_w[0, -1, 0].item())

                # sample actions from -1 to 1
                #actions = 2 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1
                actions = torch.zeros(*env.action_space.shape, device=env.unwrapped.device, dtype=torch.float32)
                #actions[:, 0] = 1.0
                actions[:, :2] = torch.from_numpy(X_commands[step, :2]).to(env.unwrapped.device)
                state_history = torch.from_numpy(X_states[step]).to(env.unwrapped.device)
                command_history = torch.from_numpy(X_commands[step]).to(env.unwrapped.device)
                #env.command_buffer_1[0, :] = command_history
                #env.state_buffer_1[0, ::4] = state_history[::4]
                #env.state_buffer_1[0, 1::4] = state_history[1::4]
                #env.state_buffer_1[0, 2::4] = state_history[2::4]
                #env.state_buffer_1[0, 3::4] = state_history[3::4]

                # apply actions
                obs, rew, terminated, truncated, info = env.step(actions)

                obs_x_pos.append(obs["policy"][0, 0].item())
                obs_x_vel.append(obs["policy"][0, 2].item())
                obs_y_pos.append(obs["policy"][0, 1].item())
                obs_y_vel.append(obs["policy"][0, 3].item())

                #print(f"y_dot gt: {Y[step, 1].item()}, y_dot sim: {obs_y_vel[-1]}")
                #print(env.state_buffer_1[0, 2::4])

                step += 1

    finally:
        actions_log = np.vstack(env.actions_log)
        # close the simulator
        env.close()
        fig, ax = plt.subplots(4)
        #ax[0].plot(obs_x_pos, label="Obs x")
        #ax[0].plot(obs_y_pos, label="Obs y")

        print("Velocity reached at step:")
        #print(np.argwhere(np.abs(np.array(obs_x_vel) - 1.0 ) < 0.01)[0])
        print(obs_x_vel)

        ax[0].plot(Y[:, 0], label="GT x_dot")
        ax[0].plot(obs_x_vel, label="Sim x_dot")
        ax[0].plot(actions_log[:, 0], label="Model x_dot")
        ax[0].legend()

        ax[1].plot(Y[:, 1], label="GT y_dot")
        ax[1].plot(obs_y_vel, label="Sim y_dot")
        ax[1].plot(actions_log[:, 1], label="Model y_dot")
        ax[1].legend()

        ax[2].plot(X_states[:, 0], label="GT x")
        ax[2].plot(obs_x_pos, label="Sim x")
        ax[2].legend()

        ax[3].plot(X_states[:, 1], label="GT y")
        ax[3].plot(obs_y_pos, label="Sim y")
        ax[3].legend()

        #ax.axhline(y=0, color='red', linestyle='--')      
        #ax.plot(obs_x_vel, label="Peg x vel")
        #ax.plot(obs_y_vel, label="Peg y vel")
        #ax.plot(obs_y_vel, label="Peg y vel")
        #ax[0].eventplot(events)
        #ax[0].legend()
        #ax.legend()
        plt.show()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
