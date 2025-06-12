import numpy as np
import torch
import gymnasium as gym
from gymnasium import Wrapper, ObservationWrapper
from collections import OrderedDict
from rl_games.torch_runner import Runner
from isaaclab_rl.rl_games import RlGamesGpuEnv
from pathlib import Path
import re
import yaml
from isaaclab_assets.robots.klask import KLASK_PARAMS


def find_wrapper(env, wrapper_type):
    """Recursively searches for a wrapper of a given type."""
    while not isinstance(env, wrapper_type):
        env = env.env  # Move to the next layer
    if isinstance(env, wrapper_type):
            return env  # Found the wrapper
    return None  # Wrapper not found
    

class KlaskRandomOpponentWrapper(Wrapper):
    
    def step(self, actions, *args, **kwargs):
        actions[:, 2:] = 2 * torch.rand_like(actions)[:, :2] - 1
        return self.env.step(actions, *args, **kwargs)
    
    def reset(self, *args, **kwargs):
        return self.env.reset(*args, **kwargs)


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

    def __init__(self, env, cfg, num_steps=None, mode="train", dynamic = False):
        super().__init__(env)
        self.dynamic = dynamic
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
            
                if self.dynamic and not (term =='time_punishment' or term =='goal_scored' or term =='goal_conceded' or term == 'opponent_in_goal' or term =='player_in_goal'):
                    term_idx = self.env.unwrapped.reward_manager.active_terms.index(term)
                    self.env.unwrapped.reward_manager._term_cfgs[term_idx].weight = weight*(torch.exp(-torch.tensor(self._step/10000000,device=self.env.unwrapped.device,dtype=torch.float32))) #coeff chosen sucht that half the max reward at 20 mio steps
        
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

    def __init__(self, config_name, num_actors, config, training_curriculum = False,is_deterministic=True, **kwargs):
        self.agent = None
        self.config = config
        self.instance_device = config["params"]["config"]["device"]
        self.is_deterministic = is_deterministic
        self.sum_rewards = 0
        self.training_curriculum = training_curriculum
        
        self.folder_path_checkpoint = Path("/home/student/klask_rl/IsaacLab/logs/rl_games/klask/training_curriculum/best_agent")
        self.folder_path_config = Path("/home/student/klask_rl/IsaacLab/planned_runs/training_curriculum")
        self.current_checkpoint = str(list(self.folder_path_checkpoint.glob("*"))[-1])
        self.current_config = self.config
        super().__init__(config_name, num_actors, **kwargs)
    
    def reset(self):
        if self.training_curriculum:
            self.should_update_agents()
        if self.agent == None:
            self.create_agent()
        #if self.training_curriculum and new_file:

        obs = self.env.reset()
        #self.opponent_obs = self.get_opponent_obs(obs)
        self.opponent_obs = find_wrapper(self.env, OpponentObservationWrapper).opponent_obs
        self.sum_rewards = 0
        return obs

    def create_agent(self):
        
        runner = Runner() 
        from rl_games.common.env_configurations import get_env_info
        self.config['params']['config']['env_info'] = get_env_info(self.env)
        runner.load(self.current_config)
        
        #os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        restore_checkpoint = self.training_curriculum and self.agent!=None
        self.agent = runner.create_player()
        if restore_checkpoint:
            self.agent.restore(self.current_checkpoint)
        
        self.agent.has_batch_dimension = True
    
    def should_update_agents(self):
        matching_file = str(list(self.folder_path_checkpoint.glob("*"))[-1])
        if matching_file == self.current_checkpoint:
            return 
        self.current_checkpoint = matching_file
        
        match = re.search(r'best_agent\((\d+)\)', self.current_checkpoint)
        if match:
            agent_number = match.group(1)
        config_path = self.folder_path_config / f"klask_config_{agent_number}.yaml"
        with open(config_path, 'r') as f:
            self.current_config = yaml.safe_load(f)
            self.current_config["params"]["config"]["device"] = self.instance_device
            self.current_config["params"]["config"]["device_name"] = self.instance_device
            
        self.create_agent()

    def step(self, action, *args, **kwargs):
        opponent_obs = self.agent.obs_to_torch(self.opponent_obs)
        opponent_action = self.agent.get_action(opponent_obs, self.is_deterministic)
        full_action = torch.cat([action, -opponent_action], dim=1)
        obs, reward, dones, info = self.env.step(full_action, *args, **kwargs)
        #self.opponent_obs = self.get_opponent_obs(obs)
        self.opponent_obs = find_wrapper(self.env, OpponentObservationWrapper).opponent_obs
        return obs, reward, dones, info
    
    def set_weights(self, indices, weigths):
        print("SETTING WEIGHTS")

        self.agent.set_weights(weigths)
        self.is_deterministic = True
        



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
        full_action = torch.cat([action, -opponent_action], dim=1)
        obs, reward, terminated,truncated, info = self.env.step(full_action, *args, **kwargs)
        
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