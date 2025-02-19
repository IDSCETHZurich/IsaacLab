import torch

from isaaclab.assets import RigidObject, Articulation
import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs import ManagerBasedRLEnv


def reset_joints_by_offset(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    position_range: tuple[float, float],
    velocity_range: tuple[float, float],
    asset_cfg: SceneEntityCfg,
):
    """Reset the robot joints with offsets around the default position and velocity by the given ranges.

    This function samples random values from the given ranges and biases the default joint positions and velocities
    by these values. The biased values are then set into the physics simulation.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # get default joint state
    joint_pos = asset.data.default_joint_pos[env_ids].clone()[:, asset_cfg.joint_ids]
    joint_vel = asset.data.default_joint_vel[env_ids].clone()[:, asset_cfg.joint_ids]

    # bias these values randomly
    joint_pos += math_utils.sample_uniform(*position_range, joint_pos.shape, joint_pos.device)
    joint_vel += math_utils.sample_uniform(*velocity_range, joint_vel.shape, joint_vel.device)

    # clamp joint pos to limits
    joint_pos_limits = asset.data.soft_joint_pos_limits[env_ids][:, asset_cfg.joint_ids, :]
    joint_pos = joint_pos.clamp_(joint_pos_limits[...,  0], joint_pos_limits[..., 1])
    # clamp joint vel to limits
    joint_vel_limits = asset.data.soft_joint_vel_limits[env_ids][:, asset_cfg.joint_ids]
    joint_vel = joint_vel.clamp_(-joint_vel_limits, joint_vel_limits)

    # set into the physics simulation
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids, joint_ids=asset_cfg.joint_ids) 


def in_goal(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, goal: tuple[float, float, float], weight: float | None = None
) -> torch.Tensor:
    """
        Penalize asset being in goal.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    body_name = asset_cfg.body_names[0]

    # Check if asset located in circle
    cx, cy, r = goal
    asset_pos_rel = asset.data.body_pos_w[:, asset_cfg.body_ids, :].squeeze() - env.scene.env_origins
    bodies_in_goal = (asset_pos_rel[:, 0] - cx) ** 2 + (asset_pos_rel[:, 1] - cy) ** 2 <= r ** 2        

    if weight is not None:
        bodies_in_goal *= weight
    
    return bodies_in_goal


def ball_in_goal(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, goal: tuple[float, float, float], max_ball_vel: float = 0.5, weight: float | None = None
) -> torch.Tensor:
    """
        Penalize asset being in goal.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    body_name = asset_cfg.name

    # Check if ball located inside goal
    cx, cy, r = goal
    ball_pos_rel = asset.data.root_pos_w - env.scene.env_origins
    ball_in_goal = (ball_pos_rel[:, 0] - cx) ** 2 + (ball_pos_rel[:, 1] - cy) ** 2 <= r ** 2
    
    # Check if ball is slower than max_vel_ball
    ball_slow = (asset.data.root_lin_vel_w[:, 0] ** 2 + 
                 asset.data.root_lin_vel_w[:, 1] ** 2 <= max_ball_vel ** 2)
    
    if weight is None:
        ball_in_goal = ball_in_goal * ball_slow
    else:
        ball_in_goal = weight * ball_in_goal * ball_slow

    return ball_in_goal


def root_xy_pos_w(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Asset root position in the environment frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]


def root_lin_xy_vel_w(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Asset root linear velocity in the environment frame."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_lin_vel_w[:, :2]


def body_xy_pos_w(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Asset body position in the environment frame"""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.body_pos_w[:, asset_cfg.body_ids, :2].squeeze(dim=1) - env.scene.env_origins[:, :2]


def body_lin_xy_vel_w(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2].squeeze(dim=1)


def opponent_goal_obs(env: ManagerBasedRLEnv, goal: tuple[float, float]) -> torch.Tensor:
    return torch.Tensor([*goal, 0.0, 0.0]).repeat(env.num_envs, 1)


def distance_player_ball(env: ManagerBasedRLEnv, player_cfg: SceneEntityCfg, ball_cfg: SceneEntityCfg) -> torch.Tensor:
    return torch.sqrt(torch.sum((root_xy_pos_w(env, ball_cfg) - body_xy_pos_w(env, player_cfg)) ** 2, dim=1))


def speed(vel: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(vel[:, 0] ** 2 + vel[:, 1] ** 2)


def ball_speed(env: ManagerBasedRLEnv, ball_cfg: SceneEntityCfg) -> torch.Tensor:
    vel = root_lin_xy_vel_w(env, ball_cfg)
    return speed(vel)


def player_speed(env: ManagerBasedRLEnv, player_cfg: SceneEntityCfg) -> torch.Tensor:
    vel = body_lin_xy_vel_w(env, player_cfg)
    return speed(vel)


def distance_ball_goal(env: ManagerBasedRLEnv, ball_cfg: SceneEntityCfg, goal: tuple[float, float, float]) -> torch.Tensor:
    cx, cy, r = goal
    ball_pos = root_xy_pos_w(env, ball_cfg)
    return 1.0 - torch.sqrt((ball_pos[:, 0] - cx) ** 2 + (ball_pos[:, 1] - cy) ** 2)


def distance_player_ball_own_half(env: ManagerBasedRLEnv, player_cfg: SceneEntityCfg, ball_cfg: SceneEntityCfg) -> torch.Tensor:
    ball_pos = root_xy_pos_w(env, ball_cfg)
    ball_in_own_half = ball_pos[:, 1] < 0.0
    return ball_in_own_half * (1.0 - distance_player_ball(env, player_cfg, ball_cfg))


def ball_stationary(env: ManagerBasedRLEnv, ball_cfg: SceneEntityCfg, eps=5e-3) -> torch.Tensor:
    return ball_speed(env, ball_cfg) < eps


def collision_player_ball(env: ManagerBasedRLEnv, player_cfg: SceneEntityCfg, ball_cfg: SceneEntityCfg, eps=0.02) -> torch.Tensor:
    return (distance_player_ball(env, player_cfg, ball_cfg) < eps) * player_speed(env, player_cfg)


def ball_in_own_half(env: ManagerBasedRLEnv, ball_cfg: SceneEntityCfg):
    ball_pos = root_xy_pos_w(env, ball_cfg)
    return 1.0 * (ball_pos[:, 1] < 0.0)
