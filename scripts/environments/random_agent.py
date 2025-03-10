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

from isaaclab_tasks.manager_based.klask import ObservationNoiseWrapper, ActuatorModelWrapper, KlaskCollisionAvoidanceWrapper
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
    env = KlaskCollisionAvoidanceWrapper(env)
    #env = ObservationNoiseWrapper(env, 0.01)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")

    peg_1_x_pos = []
    peg_1_x_vel = []
    peg_1_body_x_pos = []
    peg_1_body_x_vel = []
    obs_x_pos = []
    obs_x_vel = []
    obs_y_pos = []
    obs_y_vel = []
    resting = []

    start_index = 1000
    batch_size = 100
    data_file = "/home/student/Documents/ActuatorPolicy/actuator_model/data/data_history_10_interval_0.02_delay_0.0_horizon3_with_states.npz"
    data = np.load(data_file)
    X_commands, Y, Y_prev, commands = data["X_commands"], data["Y"], data["Y_prev"], data["commands"]
    if "X_states" in data.keys():
        X_states = data["X_states"]
    else:
        X_states = None
    X_commands, Y, Y_prev, commands = X_commands[start_index:start_index+batch_size], Y[start_index:start_index+batch_size], Y_prev[start_index:start_index+batch_size], commands[start_index:start_index+batch_size]
    if X_states is not None:
        X_states = X_states[start_index:start_index+batch_size]

    command_history = torch.from_numpy(X_commands[0]).to(env.unwrapped.device)
    if X_states is not None:
        state_history = torch.from_numpy(X_states[0]).to(env.unwrapped.device)
    
    preds = env.model(torch.from_numpy(X_commands).to("cuda"), torch.from_numpy(X_states).to("cuda"))

     # reset environment
    obs, info = env.reset()
    
    asset_cfg = SceneEntityCfg("klask", joint_names=["ground_to_slider_1"])
    asset = env.scene[asset_cfg.name]
    joint_pos = asset.data.default_joint_pos[0].clone()
    joint_vel = asset.data.default_joint_vel[0].clone()
    joint_vel[1] = torch.tensor(X_states[1, 3]).to(joint_pos.device)
    joint_vel[3] = torch.tensor(X_states[1, 2]).to(joint_pos.device)
    asset.write_joint_state_to_sim(joint_pos, joint_vel) 

    obs["policy"][0, 2] = joint_vel[3]
    obs["policy"][0, 3] = joint_vel[1]

    env.command_buffer_1[0, :] = command_history
    env.state_buffer_1[0, :] = state_history

    start_time = time.time()
    # simulate environment
    step = 0
    while simulation_app.is_running() and time.time() - start_time < 1000.0:
    #for t in range(commands.shape[1]):
        # run everything in inference mode
        with torch.inference_mode():
            #peg_1_x_pos.append(env.unwrapped.scene.articulations["klask"].data.joint_pos[0, -1].item())
            #peg_1_x_vel.append(env.unwrapped.scene.articulations["klask"].data.joint_vel[0, -1].item())
            #peg_1_body_x_pos.append(env.unwrapped.scene.articulations["klask"].data.body_pos_w[0, -1, 0].item())
            #obs_x_vel.append(env.unwrapped.scene.articulations["klask"].data.body_lin_vel_w[0, -1, 0].item())

            # sample actions from -1 to 1
            actions = 2 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1
            #actions *= 0.5
            actions[:, 2:] = 0.0
            actions[:, :2] = torch.from_numpy(X_commands[step+1, :2]).to(actions.device)
            #actions = torch.zeros(*env.action_space.shape, device=env.unwrapped.device, dtype=torch.float32)
            actions[:, 0] = 0.9
            actions[:, 1] = 0.0
            #actions[:, 1] = -0.7
            
            #print()
            #print(torch.from_numpy(X_states[step]).to(actions.device) - env.state_buffer_1[0])
            #env.state_buffer_1[0, :] = torch.from_numpy(X_states[step]).to(actions.device)

            # apply actions
            obs, rew, terminated, truncated, info = env.step(actions)
            #print(obs["policy"][0, 2:4], preds[step])
            #print(obs["policy"][0, 2:4], preds[step], X_states[step+1, :2])

            obs_x_pos.append(obs["policy"][0, 0].item())
            obs_x_vel.append(obs["policy"][0, 2].item())
            obs_y_pos.append(obs["policy"][0, 1].item())
            obs_y_vel.append(obs["policy"][0, 3].item())

            #print(f"y_dot gt: {Y[step, 1].item()}, y_dot sim: {obs_y_vel[-1]}")
            #print(env.state_buffer_1[0, 2::4])

            step += 1


    
    # close the simulator
    env.close()
    fig, ax = plt.subplots(2)
    #ax[0].plot(obs_x_pos, label="Obs x")
    #ax[0].plot(obs_y_pos, label="Obs y")

    #ax[0].plot(Y[0, :, 0], label="GT x_dot")
    ax[0].plot(obs_x_vel, label="Sim x_dot")
    ax[0].legend()

    #ax[1].plot(Y[0, :, 1], label="GT y_dot")
    ax[1].plot(obs_y_vel, label="Sim y_dot")
    ax[1].legend()

    #ax[2].plot(X_states[:, 0], label="GT x")
    #ax[2].plot(obs_x_pos, label="Sim x")
    #ax[2].legend()

    #ax[3].plot(X_states[:, 1], label="GT y")
    #ax[3].plot(obs_y_pos, label="Sim y")
    #ax[3].legend()

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
