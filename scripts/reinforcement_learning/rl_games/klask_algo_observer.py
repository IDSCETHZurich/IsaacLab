from rl_games.algos_torch import torch_ext
from rl_games.common.algo_observer import AlgoObserver
import torch
import numpy as np


class KlaskAlgoObserver(AlgoObserver):

    def __init__(self):
        pass

    def after_init(self, algo):
        self.algo = algo
        self.mean_scores = torch_ext.AverageMeter(1, self.algo.games_to_track).to(self.algo.ppo_device)
        self.ep_infos = []
        self.direct_info = {}
        self.writer = self.algo.writer

    def process_infos(self, infos, done_indices):
        if not isinstance(infos, dict):
            classname = self.__class__.__name__
            raise ValueError(f"{classname} expected 'infos' as dict. Received: {type(infos)}")
        # store episode information
        if "episode" in infos:
            self.ep_infos.append(infos["episode"])
        # log other variables directly
        if len(infos) > 0 and isinstance(infos, dict):  # allow direct logging from env
            self.direct_info = {}
            for k, v in infos.items():
                # only log scalars
                if isinstance(v, float) or isinstance(v, int) or (isinstance(v, torch.Tensor) and len(v.shape) == 0):
                    self.direct_info[k] = v
                if k == "Episode_Termination/goal_scored":
                    self.mean_scores.update(torch.ones(v, dtype=float).to(self.algo.ppo_device))
                elif k == "Episode_Termination/goal_conceded":
                    self.mean_scores.update(-torch.ones(v, dtype=float).to(self.algo.ppo_device))
                elif k == "Episode_Termination/player_in_goal":
                    self.mean_scores.update(-torch.ones(v, dtype=float).to(self.algo.ppo_device))

    def after_clear_stats(self):
        # clear stored buffers
        self.mean_scores.clear()

    def after_print_stats(self, frame, epoch_num, total_time):
        # log scalars from the episode
        if self.ep_infos:
            for key in self.ep_infos[0]:
                info_tensor = torch.tensor([], device=self.algo.device)
                for ep_info in self.ep_infos:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    info_tensor = torch.cat((info_tensor, ep_info[key].to(self.algo.device)))
                value = torch.mean(info_tensor)
                self.writer.add_scalar("Episode/" + key, value, epoch_num)
            self.ep_infos.clear()
        # log scalars from env information
        for k, v in self.direct_info.items():
            self.writer.add_scalar(f"{k}/frame", v, frame)
            self.writer.add_scalar(f"{k}/iter", v, epoch_num)
            self.writer.add_scalar(f"{k}/time", v, total_time)
        # log mean reward/score from the env
        if self.mean_scores.current_size > 0:
            mean_scores = self.mean_scores.get_mean()
            self.writer.add_scalar("scores/mean", mean_scores, frame)
            self.writer.add_scalar("scores/iter", mean_scores, epoch_num)
            self.writer.add_scalar("scores/time", mean_scores, total_time)