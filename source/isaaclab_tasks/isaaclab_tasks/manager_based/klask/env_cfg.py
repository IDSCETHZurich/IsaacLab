from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass
from isaaclab_assets.robots.klask import KLASK_PARAMS

from .config import (
    ActionsCfg,
    CurriculumCfg,
    EventCfg,
    ObservationsCfg,
    RewardsCfg,
    SceneCfg,
    TerminationsCfg,
)


@configclass
class EnvCfg(ManagerBasedRLEnvCfg):
    sim = SimulationCfg(
        physx=PhysxCfg(bounce_threshold_velocity=0.0),
        render_interval=KLASK_PARAMS["decimation"],
    )
    scene = SceneCfg(num_envs=1, env_spacing=1.0)
    observations = ObservationsCfg()
    actions = ActionsCfg()
    events = EventCfg()
    rewards = RewardsCfg()
    curriculum = CurriculumCfg()
    terminations = TerminationsCfg()
    episode_length_s = KLASK_PARAMS["timeout"]

    def __post_init__(self):
        self.viewer.eye = (0.0, 0.0, 1.0)
        self.viewer.lookat = (0.0, 0.0, 0.0)
        self.decimation = KLASK_PARAMS[
            "decimation"
        ]  # env step every 4 sim steps: 200Hz / 4 = 50Hz
        self.sim.dt = KLASK_PARAMS["physics_dt"]  # sim step every 5ms: 200Hz
