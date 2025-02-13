from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from gymnasium import spaces
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.noise import ActionNoise


class TwoPlayerSAC(SAC):
        
    def __init__(self, *args, bootstrap="random", opponent_action="model", **kwargs):
        self.bootstrap = bootstrap
        self.opponent_action = opponent_action
        super().__init__(*args, **kwargs)
    
    def _sample_action(
        self,
        learning_starts: int,
        action_noise: Optional[ActionNoise] = None,
        n_envs: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample an action according to the exploration policy.
        This is either done by sampling the probability distribution of the policy,
        or sampling a random action (from a uniform distribution over the action space)
        or by adding noise to the deterministic output.

        :param action_noise: Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :param n_envs:
        :return: action to take in the environment
            and scaled action that will be stored in the replay buffer.
            The two differs when the action space is not normalized (bounds are not [-1, 1]).
        """
        # Select action randomly or according to policy
        if self.num_timesteps < learning_starts and not (self.use_sde and self.use_sde_at_warmup):
            # Warmup phase
            if self.bootstrap == "random":
                unscaled_action_player = np.array([self.action_space.sample() for _ in range(n_envs)])
                unscaled_action_opponent = np.array([self.action_space.sample() for _ in range(n_envs)])
            elif self.bootstrap == "controlled":
                unscaled_action_player = self.get_bootstrap_action(self._last_obs, player="player")
                unscaled_action_player = self.get_bootstrap_action(self._last_obs, player="opponent")
            elif self.bootstrap == "model":
                unscaled_action_player, _ = self.predict(self._last_obs, deterministic=False)
                last_obs_opponent = self._last_obs.copy()
                last_obs_opponent["observation"] *= -1
                last_obs_opponent["achieved_goal"] *= -1
                unscaled_action_opponent, _ = self.predict(self._last_obs, deterministic=False)
            
        else:
            # Note: when using continuous actions,
            # we assume that the policy uses tanh to scale the action
            # We use non-deterministic action in the case of SAC, for TD3, it does not matter
            assert self._last_obs is not None, "self._last_obs was not set"
            unscaled_action_player, _ = self.predict(self._last_obs, deterministic=False)
            last_obs_opponent = self._last_obs.copy()
            last_obs_opponent["observation"] *= -1
            last_obs_opponent["achieved_goal"] *= -1
            if self.opponent_action == "model":
                unscaled_action_opponent, _ = self.predict(last_obs_opponent, deterministic=False)
            elif self.opponent_action == "controlled":
                unscaled_action_opponent = self.get_bootstrap_action(last_obs_opponent, player="opponent")

        # Rescale the action from [low, high] to [-1, 1]
        if isinstance(self.action_space, spaces.Box):
            scaled_action_player = self.policy.scale_action(unscaled_action_player)
            scaled_action_opponent = self.policy.scale_action(unscaled_action_opponent)

            # Add noise to the action (improve exploration)
            if action_noise is not None:
                scaled_action_player = np.clip(scaled_action_player + action_noise(), -1, 1)
                scaled_action_opponent = np.clip(scaled_action_opponent + action_noise(), -1, 1)

            # We store the scaled action in the buffer
            buffer_action_player = scaled_action_player
            buffer_action_opponent = scaled_action_opponent
            action_player = self.policy.unscale_action(scaled_action_player)
            action_opponent = self.policy.unscale_action(scaled_action_opponent)
        else:
            # Discrete case, no need to normalize or clip
            buffer_action_player = unscaled_action_player
            buffer_action_opponent = unscaled_action_opponent
            action_player = buffer_action_player
            action_opponent = buffer_action_opponent
        
        action = np.concatenate((action_player, -1.0 * action_opponent), axis=-1)
        return action, buffer_action_player
    
    def get_bootstrap_action(self, obs, player="player"):
        #target_vel = (self.env.unwrapped.scene.rigid_objects["ball"].data.root_pos_w
        #              - self.env.unwrapped.scene.articulations["klask"].data.body_pos_w[:, -1, :])[:, :2]
        player_indices = [0, 1] if player == "player" else [2, 3]
        target_vel = obs["observation"][:, 8:10] - obs["observation"][:, player_indices]
        #target_vel = target_vel.detach().cpu().numpy()
        target_vel /= np.linalg.norm(target_vel)
        return 10 * np.random.rand() * target_vel
    

class TwoPlayerPPO(PPO):
    
    def __init__(self, *args, bootstrap="random", opponent_action="model", **kwargs):
        self.bootstrap = bootstrap
        self.opponent_action = opponent_action
        super().__init__(*args, **kwargs)
    