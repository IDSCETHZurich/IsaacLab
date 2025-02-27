import numpy as np
from collections import defaultdict
import torch
import yaml
import os
import gymnasium as gym
from gymnasium import Wrapper, ActionWrapper, ObservationWrapper, spaces
from collections import OrderedDict
from stable_baselines3.common.vec_env.base_vec_env import VecEnv
from rl_games.torch_runner import Runner
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

from isaaclab_assets.robots.klask import KLASK_PARAMS
from isaaclab_rl.sb3 import Sb3VecEnvWrapper, process_sb3_cfg
from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv


def find_wrapper(env, wrapper_type):
    """Recursively searches for a wrapper of a given type."""
    while not isinstance(env, wrapper_type):
        env = env.env  # Move to the next layer
    if isinstance(env, wrapper_type):
            return env  # Found the wrapper
    return None  # Wrapper not found


class KlaskGoalEnvWrapper(Wrapper):

    def __init__(self, env):
        super().__init__(env)
        
        self.player_in_goal_weight = 1.0
        self.goal_conceded_weight = 1.0
        self.goal_scored_weight = 1.0
        self.ball_speed_weight = 1.0
        self.distance_player_ball_weight = 1.0
        self.distance_ball_opponent_goal_weight = 1.0
        self.dt = self.unwrapped.step_dt
        self.single_observation_space = self.unwrapped.single_observation_space
        self.single_action_space = spaces.Box(
            self.unwrapped.single_action_space.low[:2], 
            self.unwrapped.single_action_space.high[:2],
            shape=(2,), 
            dtype=self.unwrapped.single_action_space.dtype
        )
    
    def compute_reward(self, achieved_goal, desired_goal, info, observation, **kwargs):
        player_in_goal_reward = self.compute_player_in_goal_reward(observation)
        goal_conceded_reward = self.compute_goal_conceded_reward(observation)
        goal_reward = self.compute_goal_reward(achieved_goal, desired_goal)
        ball_speed_reward = self.compute_ball_speed_reward(observation)
        distance_player_ball_reward = self.compute_distance_player_ball_reward(observation)
        distance_ball_opponent_goal_reward = self.compute_distance_ball_opponent_goal_reward(observation)
        return [self.dt * (goal_reward + player_in_goal_reward + goal_conceded_reward + ball_speed_reward 
                           + distance_player_ball_reward)]
    
    def compute_goal_reward(self, achieved_goal, desired_goal):
        # TODO: possibly need to unnormalize achieved and desired goal OR don't normalize achieved
        # and desired goal in the first place
        r = KLASK_PARAMS["opponent_goal"][2]
        v = np.sqrt(desired_goal[:, 2] ** 2 + desired_goal[:, 3] ** 2)
        ball_in_goal = (achieved_goal[:, 0] - desired_goal[:, 0]) ** 2 + (achieved_goal[:, 1] - desired_goal[:, 1]) ** 2 <= r ** 2
        ball_slow = ((achieved_goal[:, 2] ** 2 + achieved_goal[:, 3] ** 2 >= v ** 2) & 
                     (achieved_goal[:, 2] ** 2 + achieved_goal[:, 3] ** 2 <= (v + KLASK_PARAMS["max_ball_vel"]) ** 2))
        return self.goal_scored_weight * ball_in_goal * ball_slow
    
    def compute_player_in_goal_reward(self, observation):
        cx, cy, r = KLASK_PARAMS["player_goal"]
        player_in_goal = (observation[:, 0] - cx) ** 2 + (observation[:, 1] - cy) ** 2 <= r ** 2
        return self.player_in_goal_weight * player_in_goal

    def compute_goal_conceded_reward(self, observation):
        cx, cy, r = KLASK_PARAMS["player_goal"]
        ball_in_goal = (observation[:, 8] - cx) ** 2 + (observation[:, 9] - cy) ** 2 <= r ** 2
        ball_slow = observation[:, 10] ** 2 + observation[:, 11] ** 2 <= KLASK_PARAMS["max_ball_vel"] ** 2
        return self.goal_conceded_weight * ball_in_goal * ball_slow

    def compute_ball_speed_reward(self, observation):
        return self.ball_speed_weight * np.sqrt((observation[:, 10] ** 2 + observation[:, 11] ** 2))   

    def compute_distance_player_ball_reward(self, observation):
        return  self.distance_player_ball_weight * np.sqrt((observation[:, 0] - observation[:, 8]) ** 2 + (observation[:, 1] - observation[:, 9]) ** 2)
    
    def compute_distance_ball_opponent_goal_reward(self, observation):
        cx, cy, r = KLASK_PARAMS["player_goal"]
        distance_ball_opponent_goal = torch.sqrt((observation[:, 8] - cx) ** 2 + (observation[:, 9] - cy) ** 2)
        return self.distance_ball_opponent_goal_weight * distance_ball_opponent_goal
    

class KlaskRandomOpponentWrapper(Wrapper):
    
    def step(self, actions, *args, **kwargs):
        actions[:, 2:] = 2 * torch.rand_like(actions)[:, :2] - 1
        return self.env.step(actions, *args, **kwargs)
    
    def reset(self, *args, **kwargs):
        return self.env.reset(*args, **kwargs)


class KlaskSb3VecEnvWrapper(Sb3VecEnvWrapper):
    
    def __init__(self, env: KlaskGoalEnvWrapper):
        """Initialize the wrapper.

        Args:
            env: The environment to wrap around.

        Raises:
            ValueError: When the environment is not an instance of :class:`ManagerBasedRLEnv` or :class:`DirectRLEnv`.
        """
        # check that input is valid
        if not isinstance(env.unwrapped, ManagerBasedRLEnv) and not isinstance(env.unwrapped, DirectRLEnv):
            raise ValueError(
                "The environment must be inherited from ManagerBasedRLEnv or DirectRLEnv. Environment type:"
                f" {type(env)}"
            )
        # initialize the wrapper
        self.env = env
        # collect common information
        self.num_envs = self.unwrapped.num_envs
        self.sim_device = self.unwrapped.device
        self.render_mode = self.unwrapped.render_mode

        # obtain gym spaces
        # note: stable-baselines3 does not like when we have unbounded action space so
        #   we set it to some high value here. Maybe this is not general but something to think about.
        #observation_space = self.unwrapped.single_observation_space
        observation_space = self.env.single_observation_space
        action_space = self.env.single_action_space
        if isinstance(action_space, gym.spaces.Box) and not action_space.is_bounded("both"):
            action_space = gym.spaces.Box(low=-100, high=100, shape=action_space.shape)

        # initialize vec-env
        VecEnv.__init__(self, self.num_envs, observation_space, action_space)
        # add buffer for logging episodic information
        self._ep_rew_buf = torch.zeros(self.num_envs, device=self.sim_device)
        self._ep_len_buf = torch.zeros(self.num_envs, device=self.sim_device)

    def _process_obs(self, obs_dict: torch.Tensor | dict[str, torch.Tensor]) -> np.ndarray | dict[str, np.ndarray]:
        """Convert observations into NumPy data type."""
        obs = obs_dict

        # note: ManagerBasedRLEnv uses torch backend (by default).
        if isinstance(obs, dict):
            for key, value in obs.items():
                obs[key] = value.detach().cpu().numpy()
        elif isinstance(obs, torch.Tensor):
            obs = obs.detach().cpu().numpy()
        else:
            raise NotImplementedError(f"Unsupported data type: {type(obs)}")
        return obs
    

class ObservationNoiseWrapper(ObservationWrapper):

    def __init__(self, env, noise_std, indices=None):
        super().__init__(env)
        self.noise_std = noise_std
        self.indices = indices
        if self.indices is None:
            self.indices = env.unwrapped.single_action_space.shape[-1]
    
    def observation(self, observation):
        if type(observation) is dict:
            for k, v in observation.items():
                noise = self.noise_std * torch.randn_like(v)
                observation[k][:, self.indices] += noise[:, self.indices]
        else:
            noise = self.noise_std * torch.randn_like(observation)
            observation[:, self.indices] += noise[:, self.indices]
        return observation
    

class CurriculumWrapper(Wrapper):

    def __init__(self, env, cfg, num_steps=None, mode="train"):
        super().__init__(env)
        self.cfg = cfg
        self.num_steps = num_steps
        self.mode = mode
        self._step = 0
        for term, weight in cfg.items():
            term_idx = self.env.unwrapped.reward_manager.active_terms.index(term)
            if type(weight) is dict:
                if type(weight["weight"]) is list:
                    _weight = weight["weight"][0]
                else:
                    _weight = weight["weight"]
                if not weight.get("per_second", False):
                    _weight /= KLASK_PARAMS["decimation"] * KLASK_PARAMS["physics_dt"]
            else:
                _weight = weight / (KLASK_PARAMS["decimation"] * KLASK_PARAMS["physics_dt"])
            self.env.unwrapped.reward_manager._term_cfgs[term_idx].weight = _weight

    def step(self, actions):
        if self.mode == "train":
            self._step += self.env.unwrapped.num_envs
            for term, weight in self.cfg.items():
                if type(weight) is dict and type(weight["weight"]) is list:
                    term_idx = self.env.unwrapped.reward_manager.active_terms.index(term)
                    weight_step = (weight["weight"][1] - weight["weight"][0]) / self.num_steps
                    if not weight.get("per_second", False):
                        weight_step /= KLASK_PARAMS["decimation"] * KLASK_PARAMS["physics_dt"]
                    self.env.unwrapped.reward_manager._term_cfgs[term_idx].weight += weight_step
        
        return self.env.step(actions)
                  

class OpponentObservationWrapper(Wrapper):

    def __init__(self, env, mode="train"):
        super().__init__(env)
        self.mode = mode
    
    def get_opponent_obs(self, obs):
        opponent_obs = obs.detach().clone()
        opponent_obs[:, :12] = -obs[:, :12]
        return opponent_obs
    
    def reset(self, *args, **kwargs):
        obs_dict, extras = self.env.reset(*args, **kwargs)
        if self.mode == "train":
            self.opponent_obs = self.get_opponent_obs(obs_dict["opponent"])
        else:
            obs_dict["opponent"] = self.get_opponent_obs(obs_dict["opponent"])
        return obs_dict, extras
    
    def step(self, actions, *args, **kwargs):
        obs_dict, rew, terminated, truncated, extras = self.env.step(actions, *args, **kwargs)
        if self.mode == "train":
            self.opponent_obs = self.get_opponent_obs(obs_dict["opponent"])
        else:
            obs_dict["opponent"] = self.get_opponent_obs(obs_dict["opponent"])
        return obs_dict, rew, terminated, truncated, extras 


class RlGamesGpuEnvSelfPlay(RlGamesGpuEnv):

    def __init__(self, config_name, num_actors, config, is_deterministic=False, **kwargs):
        self.agent = None
        self.config = config
        self.is_deterministic = is_deterministic
        self.sum_rewards = 0
        super().__init__(config_name, num_actors, **kwargs)
    
    def reset(self):
        if self.agent == None:
            self.create_agent()
        obs = self.env.reset()
        #self.opponent_obs = self.get_opponent_obs(obs)
        self.opponent_obs = find_wrapper(self.env, OpponentObservationWrapper).opponent_obs
        self.sum_rewards = 0
        return obs

    def create_agent(self):
        runner = Runner()
        from rl_games.common.env_configurations import get_env_info
        self.config['params']['config']['env_info'] = get_env_info(self.env)
        runner.load(self.config)
        #os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        self.agent = runner.create_player()
        self.agent.has_batch_dimension = True

    def step(self, action, *args, **kwargs):
        opponent_obs = self.agent.obs_to_torch(self.opponent_obs)
        opponent_action = self.agent.get_action(opponent_obs, self.is_deterministic)
        action[:, 2:] = -opponent_action[:, :2]
        obs, reward, dones, info = self.env.step(action, *args, **kwargs)
        #self.opponent_obs = self.get_opponent_obs(obs)
        self.opponent_obs = find_wrapper(self.env, OpponentObservationWrapper).opponent_obs
        return obs, reward, dones, info
    
    def set_weights(self, indices, weigths):
        print("SETTING WEIGHTS")
        self.agent.set_weights(weigths)


class KlaskAgentOpponentWrapper(Wrapper):
    
    def __init__(self, env, is_deterministic=False):
        super().__init__(env)
        self.opponent = None
        self.is_deterministic = is_deterministic

    def add_opponent(self, opponent):
        self.opponent = opponent
        self.opponent.has_batch_dimension = True
    
    def get_opponent_obs(self, obs):
        opponent_obs = obs.detach().clone()
        opponent_obs[:, :12] = -obs[:, :12]
        return opponent_obs

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        self.opponent_obs = self.get_opponent_obs(obs["opponent"])
        return obs, info
    
    def step(self, action, *args, **kwargs):
        opponent_obs = self.opponent.obs_to_torch(self.opponent_obs)
        opponent_action = self.opponent.get_action(opponent_obs, self.is_deterministic)
        action[:, 2:] = -opponent_action[:, :2]
        obs, reward, terminated, truncated, info = self.env.step(action, *args, **kwargs)
        self.opponent_obs = self.get_opponent_obs(obs["opponent"])
        return obs, reward, terminated, truncated, info
        

class KlaskCollisionAvoidanceWrapper(Wrapper):
    
    real_to_sim_factor = 0.0008285
    board_dimensions = (0.32, 0.44)
    speed_limit_weight = 70.0

    def __init__(self, env, action_factor=1.0):
        super().__init__(env)
        self.action_factor = action_factor
        self.x_min, self.x_max = 15.0, 360.0
        self.y_min_1, self.y_max_1 = 15.0, 235.0
        self.y_min_2, self.y_max_2 = 340.0, 530.0

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        self.state_1 = obs["policy"].clone()[:, :2]
        self.state_2 = obs["opponent"].clone()[:, 4:6]
        self.state_1[:, 0] += self.board_dimensions[0] / 2 
        self.state_1[:, 1] += self.board_dimensions[1] / 2
        self.state_2[:, 0] += self.board_dimensions[0] / 2 
        self.state_2[:, 1] += self.board_dimensions[1] / 2
        self.state_1 /= self.real_to_sim_factor
        self.state_2 /= self.real_to_sim_factor

        return obs, info
    
    def step(self, actions, *args, **kwargs):
        actions *= self.action_factor
        
        x_vel_max = torch.tanh((self.x_max - self.state_1[:, 0]) / self.speed_limit_weight).clamp(min=0.0)
        x_vel_min = torch.tanh((self.x_min - self.state_1[:, 0]) / self.speed_limit_weight).clamp(max=0.0)
        y_vel_max = torch.tanh((self.y_max_1 - self.state_1[:, 1]) / self.speed_limit_weight).clamp(min=0.0)
        y_vel_min = torch.tanh((self.y_min_1 - self.state_1[:, 1]) / self.speed_limit_weight).clamp(max=0.0)

        actions[:, 0] = actions[:, 0].clamp(min=x_vel_min, max=x_vel_max)
        actions[:, 1] = actions[:, 1].clamp(min=y_vel_min, max=y_vel_max)

        x_vel_max = torch.tanh((self.x_max - self.state_2[:, 0]) / self.speed_limit_weight).clamp(min=0.0)
        x_vel_min = torch.tanh((self.x_min - self.state_2[:, 0]) / self.speed_limit_weight).clamp(max=0.0)
        y_vel_max = torch.tanh((self.y_max_2 - self.state_2[:, 1]) / self.speed_limit_weight).clamp(min=0.0)
        y_vel_min = torch.tanh((self.y_min_2 - self.state_2[:, 1]) / self.speed_limit_weight).clamp(max=0.0)

        actions[:, 2] = actions[:, 2].clamp(min=x_vel_min, max=x_vel_max)
        actions[:, 3] = actions[:, 3].clamp(min=y_vel_min, max=y_vel_max)

        obs, rew, terminated, truncated, info = self.env.step(actions, *args, **kwargs)
        self.state_1 = obs["policy"].clone()[:, :2]
        self.state_2 = obs["opponent"].clone()[:, 4:6]
        self.state_1[:, 0] += self.board_dimensions[0] / 2 
        self.state_1[:, 1] += self.board_dimensions[1] / 2
        self.state_2[:, 0] += self.board_dimensions[0] / 2 
        self.state_2[:, 1] += self.board_dimensions[1] / 2
        self.state_1 /= self.real_to_sim_factor
        self.state_2 /= self.real_to_sim_factor

        return obs, rew, terminated, truncated, info
    

class ActionHistoryWrapper(Wrapper):

    def __init__(self, env, history_length):
        super().__init__(env)
        self.history_length = history_length
        self.history_x_player = torch.zeros(env.unwrapped.num_envs, history_length).to(env.unwrapped.device)
        self.history_y_player = torch.zeros(env.unwrapped.num_envs, history_length).to(env.unwrapped.device)
        self.history_x_opponent = torch.zeros(env.unwrapped.num_envs, history_length).to(env.unwrapped.device)
        self.history_y_opponent = torch.zeros(env.unwrapped.num_envs, history_length).to(env.unwrapped.device)

    def reset(self, *args, **kwargs):
        return self.env.reset(*args, **kwargs)

    def step(self, actions, *args, **kwargs):
        self.history_x_player[:, :-1] = self.history_x_player.clone()[:, 1:]
        self.history_x_player[:, -1] = actions[:, 0]
        self.history_y_player[:, :-1] = self.history_y_player.clone()[:, 1:]
        self.history_y_player[:, -1] = actions[:, 1]
        self.history_x_opponent[:, :-1] = self.history_x_opponent.clone()[:, 1:]
        self.history_x_opponent[:, -1] = actions[:, 2]
        self.history_y_opponent[:, :-1] = self.history_y_opponent.clone()[:, 1:]
        self.history_y_opponent[:, -1] = actions[:, 3]

        obs, rew, terminated, truncated, info = self.env.step(actions, *args, **kwargs)
        obs["policy"][:, 12:] = torch.cat([self.history_x_player, self.history_y_player], dim=1)
        obs["opponent"][:, 12:] = torch.cat([self.history_x_opponent, self.history_y_opponent], dim=1)
        return obs, rew, terminated, truncated, info