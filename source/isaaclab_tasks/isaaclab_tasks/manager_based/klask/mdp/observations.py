import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

from .utils import body_xy_pos_w, root_xy_pos_w


def angle_ball_goal(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg,
    player_cfg: SceneEntityCfg,
    goal: tuple[float, float, float],
) -> torch.Tensor:
    ball_pos = root_xy_pos_w(env, ball_cfg)
    player_pos = body_xy_pos_w(env, player_cfg)
    cx, cy, r = goal
    goal_pos = torch.tensor([cx, cy], device=player_pos.device)  # shape (2,)

    # Vectors from player to goal and ball
    vec_to_goal = goal_pos - player_pos  # shape (N, 2)
    vec_to_ball = ball_pos - player_pos  # shape (N, 2)

    # Dot product between the vectors
    dot = torch.sum(vec_to_goal * vec_to_ball, dim=1)  # shape (N,)

    # Norms (magnitudes)
    norm_goal = torch.norm(vec_to_goal, dim=1)  # shape (N,)
    norm_ball = torch.norm(vec_to_ball, dim=1)  # shape (N,)

    # Cosine of angle
    cos_theta = dot / (norm_goal * norm_ball + 1e-8)  # avoid division by zero
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)  # numerical stability

    # Angle in radians
    angle_rad = torch.acos(cos_theta)
    return angle_rad.unsqueeze(-1)


def angle_ball_opp(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg,
    player_1_cfg: SceneEntityCfg,
    player_2_cfg: SceneEntityCfg,
) -> torch.Tensor:
    ball_pos = root_xy_pos_w(env, ball_cfg)
    player_pos = body_xy_pos_w(env, player_1_cfg)
    opponent_pos = body_xy_pos_w(env, player_2_cfg)
    vec_opp_to_ball = ball_pos - player_pos  # (N, 2)

    # Vector from ball to player
    vec_ball_to_player = opponent_pos - player_pos  # (N, 2)

    # Dot product and norms
    dot = torch.sum(vec_opp_to_ball * vec_ball_to_player, dim=1)  # (N,)
    norm1 = torch.norm(vec_opp_to_ball, dim=1)  # (N,)
    norm2 = torch.norm(vec_ball_to_player, dim=1)  # (N,)

    # Cosine and angle
    cos_theta = dot / (norm1 * norm2 + 1e-8)  # Avoid division by zero
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)  # Clamp for numerical stability
    angle_rad = torch.acos(cos_theta)  # (N,)

    return angle_rad.unsqueeze(-1)


def distance_to_goal(
    env: ManagerBasedRLEnv, ball_cfg: SceneEntityCfg, goal: tuple
) -> torch.Tensor:
    """
    Compute Euclidean distance from ball to goal center.

    Args:
        env: simulation environment
        ball_cfg: config for ball entity
        goal: (cx, cy, r) goal center and radius

    Returns:
        torch.Tensor of shape (N,) with distances
    """
    ball_pos = root_xy_pos_w(env, ball_cfg)  # (N, 2)
    goal_pos = torch.tensor(goal[:2], device=ball_pos.device)  # (2,)
    dist = torch.norm(ball_pos - goal_pos, dim=1)  # (N,)
    return dist.unsqueeze(-1)


def distance_ball_to_player(
    env: ManagerBasedRLEnv, ball_cfg: SceneEntityCfg, player_cfg: SceneEntityCfg
) -> torch.Tensor:
    """
    Compute Euclidean distance from ball to player.

    Args:
        env: simulation environment
        ball_cfg: config for ball entity
        player_cfg: config for player entity

    Returns:
        torch.Tensor of shape (N,) with distances
    """
    ball_pos = root_xy_pos_w(env, ball_cfg)  # (N, 2)
    player_pos = body_xy_pos_w(env, player_cfg)  # (N, 2)
    dist = torch.norm(ball_pos - player_pos, dim=1)  # (N,)
    return dist.unsqueeze(-1)
