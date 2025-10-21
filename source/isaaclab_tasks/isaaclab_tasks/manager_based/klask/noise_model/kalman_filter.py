from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import ManagerTermBase, ObservationTermCfg, SceneEntityCfg

from ..mdp.utils import body_xy_pos_w, root_lin_xy_vel_w, root_xy_pos_w

if TYPE_CHECKING:
    from collections.abc import Sequence


class KalmanFilter(ManagerTermBase):
    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.ball_state = torch.zeros(env.num_envs, 4, device=env.device)

        # Process and measurement noise covariances
        self.Q = torch.tensor(
            [
                [cfg.params["process_variance_pos"], 0, 0, 0],
                [0, cfg.params["process_variance_pos"], 0, 0],
                [0, 0, cfg.params["process_variance_vel"], 0],
                [0, 0, 0, cfg.params["process_variance_vel"]],
            ],
            device=env.device,
        )
        self.R = cfg.params["measurement_variance_pos"] * torch.eye(
            2, device=env.device
        )

        # Process matrix
        self.F = torch.tensor(
            [
                [1, 0, env.step_dt, 0],
                [0, 1, 0, env.step_dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            device=env.device,
        )

        # Measurement matrix (we observe positions only)
        self.H = torch.tensor(
            [[1, 0, 0, 0], [0, 1, 0, 0]], device=env.device, dtype=torch.float
        )

        # Initial estimation error covariance
        P = torch.eye(4, device=env.device)
        self.P = P.unsqueeze(0).repeat(env.num_envs, 1, 1)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        x_limits: tuple[float, float],
        y_limits: tuple[float, float],
        distance: float,
        process_variance_pos: float,
        process_variance_vel: float,
        measurement_variance_pos: float,
        peg_1: SceneEntityCfg,
        peg_2: SceneEntityCfg,
        ball: SceneEntityCfg,
    ) -> torch.Tensor:
        ball_pos = root_xy_pos_w(env, asset_cfg=ball).clone()
        ball_pos += 0.014 * torch.rand_like(ball_pos) - 0.007
        ball_vel = root_lin_xy_vel_w(env, asset_cfg=ball).clone()
        ball_state = torch.cat([ball_pos, ball_vel], dim=1)
        peg_1_pos = body_xy_pos_w(env, asset_cfg=peg_1)
        peg_2_pos = body_xy_pos_w(env, asset_cfg=peg_2)
        x_collision, y_collision = self._check_collision(
            ball_state, x_limits, y_limits, distance, peg_1_pos, peg_2_pos
        )
        self._predict(x_collision, y_collision)
        self._update(ball_state)

        return self.ball_state

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self._env.num_envs, device=self.device)
        self.ball_state[env_ids] = 0.0
        self.P[env_ids] = torch.eye(4, device=self.device)

    def _check_collision(
        self,
        ball_state: torch.Tensor,
        x_limits: tuple[float, float],
        y_limits: tuple[float, float],
        distance: float,
        peg_1_pos: torch.Tensor,
        peg_2_pos: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = ball_state[:, 0], ball_state[:, 1]
        x_collision = (x < x_limits[0] + distance) | (x > x_limits[1])
        y_collision = (y < y_limits[0] + distance) | (y > y_limits[1])
        close_to_peg = (
            torch.linalg.norm(ball_state[:, :2] - peg_1_pos, dim=1) < distance
        ) | (torch.linalg.norm(ball_state[:, :2] - peg_2_pos, dim=1) < distance)
        x_collision = x_collision | close_to_peg
        y_collision = y_collision | close_to_peg

        return x_collision, y_collision

    def _predict(self, x_collision: torch.Tensor, y_collision: torch.Tensor):
        Q = self.Q.repeat(self._env.num_envs, 1, 1)
        Q[x_collision] += torch.diag(
            torch.tensor([4.0, 0.0, 10000.0, 0.0], device=Q.device)
        )
        Q[y_collision] += torch.diag(
            torch.tensor([0.0, 4.0, 0.0, 10000.0], device=Q.device)
        )

        self.ball_state = (self.F @ self.ball_state.unsqueeze(-1)).squeeze(-1)
        self.P = self.F @ self.P @ self.F.T + Q

    def _update(self, ball_state: torch.Tensor):
        pos_error = ball_state[:, :2] - (
            self.H @ self.ball_state.unsqueeze(-1)
        ).squeeze(-1)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ torch.linalg.inv(S)
        self.ball_state = self.ball_state + (K @ pos_error.unsqueeze(-1)).squeeze(-1)
        self.P = (torch.eye(4, device=ball_state.device) - K @ self.H) @ self.P
