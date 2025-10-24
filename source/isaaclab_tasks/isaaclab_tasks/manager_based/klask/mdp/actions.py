from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING
from collections.abc import Sequence

import torch
from isaaclab.assets import AssetBase
from isaaclab.envs.mdp.actions import JointVelocityAction, JointVelocityActionCfg
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from ..actuator_model import ActuatorNetwork
from .manager_base import ManagerBase, ManagerTermBase
from .manager_term_cfg import ActionTermCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class ActuatorModelAction(JointVelocityAction):
    def __init__(self, cfg: ActuatorModelActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.cfg = cfg

        num_envs = env.num_envs
        self.dT = env.step_dt
        input_dim = (
            cfg.num_history_steps * 2
            + cfg.include_states * (cfg.num_history_steps - 1) * 2
        )
        output_dim = 2
        self.model = ActuatorNetwork(
            input_dim, output_dim, hidden_dim=cfg.hidden_dim
        ).to(env.device)
        self.model.load_state_dict(torch.load(cfg.model_file, map_location=env.device))
        self.command_buffer_1 = torch.zeros(
            num_envs, 2 * cfg.num_history_steps, dtype=torch.float32
        ).to(env.device)
        self.command_buffer_2 = torch.zeros(
            num_envs, 2 * cfg.num_history_steps, dtype=torch.float32
        ).to(env.device)

        if cfg.include_states:
            self.state_buffer_1 = torch.zeros(
                num_envs, 2 * (cfg.num_history_steps - 1), dtype=torch.float32
            ).to(env.device)
            self.state_buffer_2 = torch.zeros(
                num_envs, 2 * (cfg.num_history_steps - 1), dtype=torch.float32
            ).to(env.device)
            self.position_player = torch.zeros(num_envs, 2, dtype=torch.float32).to(
                env.device
            )
            self.position_opponent = torch.zeros(num_envs, 2, dtype=torch.float32).to(
                env.device
            )

        self.model.eval()

    def process_actions(self, actions: torch.Tensor):
        # Peg 1 command history update:
        command_1 = actions[:, :2]
        prev_buffer = self.command_buffer_1.clone()
        self.command_buffer_1[:, 2:] = prev_buffer[:, :-2]
        self.command_buffer_1[:, :2] = command_1

        # Peg 2 command history update:
        command_2 = actions[:, 2:]
        self.command_buffer_2[:, 2:] = self.command_buffer_2.clone()[:, :-2]
        self.command_buffer_2[:, :2] = command_2

        # Map commands to velocities using the actuator model:
        if self.cfg.include_states:
            states_input_1, states_input_2 = self.state_buffer_1, self.state_buffer_2
        else:
            states_input_1, states_input_2 = None, None

        with torch.no_grad():
            actions_1 = self.model(self.command_buffer_1, states_input_1)
            actions_2 = self.model(self.command_buffer_2, states_input_2)

        actions[:, :2] = actions_1
        actions[:, 2:] = actions_2
        obs, rew, terminated, truncated, info = self.env.step(actions, *args, **kwargs)

        if self.cfg.include_states:
            # Peg 1 state history update:
            self.position_player = obs["policy"][:, :2]
            state_1 = obs["policy"][:, 2:4]

            self.state_buffer_1[:, 2:] = self.state_buffer_1.clone()[:, :-2]
            self.state_buffer_1[:, :2] = state_1

            # Peg 2 state history update:
            self.position_opponent = obs["opponent"][:, :2]
            state_2 = obs["opponent"][:, 2:4]

            self.state_buffer_2[:, 2:] = self.state_buffer_2.clone()[:, :-2]
            self.state_buffer_2[:, :2] = state_2

        # Reset buffers for terminated envs:
        done = terminated | truncated
        self.command_buffer_1[done, :] = 0.0
        self.command_buffer_2[done, :] = 0.0
        if self.include_states:
            self.state_buffer_1[done, :] = state_1[done].repeat(
                1, self.num_history_steps - 1
            )
            self.state_buffer_2[done, :] = state_2[done].repeat(
                1, self.num_history_steps - 1
            )

    def reset(self, env_ids: Sequence[int] | None = None):
        pass



@configclass
class ActuatorModelActionCfg(JointVelocityActionCfg):
    class_type = ActuatorModelAction

    model_file: str = MISSING
    """The path to the actuator model .pth file."""

    num_history_steps: int = 10
    """The number of past commands to include in input."""

    hidden_dim: int = 64
    """The hidden dimension of the MLP."""

    delay: int = 0
    """The delay (in env steps) between the most recent command included in the input 
    and the current time.
    """

    include_states: bool = True
    """Whether to include a history of peg states in the model input."""
