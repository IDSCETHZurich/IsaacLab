import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
import isaaclab.envs.mdp as mdp
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import CurriculumTermCfg as CurriculumTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim import SimulationCfg, PhysxCfg

from isaaclab_assets.robots.klask import KLASK_CFG, KLASK_PARAMS
from .utils_manager_based import *


@configclass
class KlaskSceneCfg(InteractiveSceneCfg):
    """Configuration for Klask scene."""

    # ground plane
    # ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    ball = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Ball",
        spawn=sim_utils.SphereCfg(
            radius=0.007,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=KLASK_PARAMS["ball_mass_initial"]),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                restitution=KLASK_PARAMS["ball_restitution"],
                static_friction=KLASK_PARAMS["ball_static_friction"],
                dynamic_friction=KLASK_PARAMS["ball_dynamic_friction"]
            ),
            activate_contact_sensors=True
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    #contact_sensor = ContactSensorCfg(
    #    prim_path="{ENV_REGEX_NS}/Ball",
    #    filter_prim_paths_expr=["{ENV_REGEX_NS}/Klask/Peg_1"],
    #    history_length=KLASK_PARAMS["decimation"]
    #)

    klask = KLASK_CFG.replace(prim_path="{ENV_REGEX_NS}/Klask")


@configclass
class ActionsCfg:
    """Action specifications for the environment."""
    #player_actions = mdp.JointEffortActionCfg(asset_name="klask", 
    #                                           joint_names=["slider_to_peg_1", "ground_to_slider_1"],)

    #opponent_actions = mdp.JointEffortActionCfg(asset_name="klask", 
    #                                           joint_names=["slider_to_peg_2", "ground_to_slider_2"],)

    player_x = mdp.JointVelocityActionCfg(
        asset_name="klask", 
        joint_names=["slider_to_peg_1"]
    )

    player_y = mdp.JointVelocityActionCfg(
        asset_name="klask", 
        joint_names=["ground_to_slider_1"]
    )

    opponent_x = mdp.JointVelocityActionCfg(
        asset_name="klask", 
        joint_names=["slider_to_peg_2"]
    )

    opponent_y = mdp.JointVelocityActionCfg(
        asset_name="klask", 
        joint_names=["ground_to_slider_2"]
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the environment."""

     # TODO: noise corruption
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        
        # observation terms (order preserved)
        peg_1_pos = ObsTerm(func=body_xy_pos_w, params={"asset_cfg": SceneEntityCfg(
            name="klask", 
            body_names=["Peg_1"]
        )}, )#scale=2/BOARD_WIDTH)

        peg_2_pos = ObsTerm(func=body_xy_pos_w, params={"asset_cfg": SceneEntityCfg(
            name="klask", 
            body_names=["Peg_2"],
        )}, )#scale=2/BOARD_LENGTH)

        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg(
            name="klask", 
            joint_names=["ground_to_slider_1", "slider_to_peg_1", "ground_to_slider_2", "slider_to_peg_2"]
        )}, )

        ball_pos_rel = ObsTerm(func=root_xy_pos_w, params={"asset_cfg": SceneEntityCfg(name="ball")})
        
        ball_vel_rel = ObsTerm(func=root_lin_xy_vel_w, params={"asset_cfg": SceneEntityCfg(name="ball")})

        if KLASK_PARAMS.get("action_history", 0):
            action_history = ObsTerm(func=mdp.last_action, history_length=KLASK_PARAMS["action_history"])


        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class HistoryObservationsCfg:
    """Observation specifications for the environment."""

     # TODO: noise corruption
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        
        # observation terms (order preserved)
        peg_1_pos = ObsTerm(func=body_xy_pos_w, params={"asset_cfg": SceneEntityCfg(
            name="klask", 
            body_names=["Peg_1"]
        )}, )#scale=2/BOARD_WIDTH)

        peg_2_pos = ObsTerm(func=body_xy_pos_w, params={"asset_cfg": SceneEntityCfg(
            name="klask", 
            body_names=["Peg_2"],
        )}, )#scale=2/BOARD_LENGTH)

        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg(
            name="klask", 
            joint_names=["ground_to_slider_1", "slider_to_peg_1", "ground_to_slider_2", "slider_to_peg_2"]
        )}, )

        ball_pos_rel = ObsTerm(func=root_xy_pos_w, params={"asset_cfg": SceneEntityCfg(name="ball")})
        
        ball_vel_rel = ObsTerm(func=root_lin_xy_vel_w, params={"asset_cfg": SceneEntityCfg(name="ball")})

        action_history = ObsTerm(func=mdp.last_action, history_length=KLASK_PARAMS["action_history"])

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class GoalObservationsCfg:
    """Observation specifications for the environment."""

     # TODO: noise corruption
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        
        # observation terms (order preserved)
        peg_1_pos = ObsTerm(func=body_xy_pos_w, params={"asset_cfg": SceneEntityCfg(
            name="klask", 
            body_names=["Peg_1"]
        )}, )#scale=2/BOARD_WIDTH)

        peg_2_pos = ObsTerm(func=body_xy_pos_w, params={"asset_cfg": SceneEntityCfg(
            name="klask", 
            body_names=["Peg_2"],
        )}, )#scale=2/BOARD_LENGTH)

        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg(
            name="klask", 
            joint_names=["ground_to_slider_1", "slider_to_peg_1", "ground_to_slider_2", "slider_to_peg_2"]
        )}, )

        ball_pos_rel = ObsTerm(func=root_xy_pos_w, params={"asset_cfg": SceneEntityCfg(name="ball")})
        
        ball_vel_rel = ObsTerm(func=root_lin_xy_vel_w, params={"asset_cfg": SceneEntityCfg(name="ball")})

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class AchievedGoalCfg(ObsGroup):
        ball_pos_rel = ObsTerm(func=root_xy_pos_w, params={"asset_cfg": SceneEntityCfg(name="ball")})
        ball_vel_rel = ObsTerm(func=root_lin_xy_vel_w, params={"asset_cfg": SceneEntityCfg(name="ball")})

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class DesiredGoalCfg(ObsGroup):
        ball_in_goal = ObsTerm(func=opponent_goal_obs, params={"goal": (0.0, 0.176215)})

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    observation: PolicyCfg = PolicyCfg()
    desired_goal: DesiredGoalCfg = DesiredGoalCfg()
    achieved_goal: AchievedGoalCfg = AchievedGoalCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # on reset
    add_ball_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("ball"),
            "mass_distribution_params": KLASK_PARAMS["ball_mass_dist"],
            "operation": "add",
        },
    )

    reset_x_position_peg_1 = EventTerm(
        func=reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("klask", joint_names=["slider_to_peg_1"]),
            "position_range": (-0.14, 0.14),
            "velocity_range": (0.0, 0.0),
        },
    )

    reset_x_position_peg_2 = EventTerm(
        func=reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("klask", joint_names=["slider_to_peg_2"]),
            "position_range": (-0.14, 0.14),
            "velocity_range": (0.0, 0.0),
        },
    )

    reset_y_position_peg_1 = EventTerm(
        func=reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("klask", joint_names=["ground_to_slider_1"]),
            "position_range": (-0.2, -0.01),
            "velocity_range": (0.0, 0.0),
        },
    )

    reset_y_position_peg_2 = EventTerm(
        func=reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("klask", joint_names=["ground_to_slider_2"]),
            "position_range": (0.01, 0.2),
            "velocity_range": (0.0, 0.0),
        },
    )

    reset_ball_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("ball"),
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.16, 0.0)},
            "velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 0.0)}
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    
    player_in_goal = RewTerm(
        func=in_goal, 
        params={
            "asset_cfg": SceneEntityCfg("klask", body_names=["Peg_1"]),
            "goal": KLASK_PARAMS["player_goal"]
        },
        weight=KLASK_PARAMS["rewards"]["player_in_goal"] / (KLASK_PARAMS["decimation"] * KLASK_PARAMS["physics_dt"])
    )

    goal_scored = RewTerm(
        func=ball_in_goal, 
        params={
            "asset_cfg": SceneEntityCfg("ball"),
            "goal": KLASK_PARAMS["opponent_goal"],
            "max_ball_vel": KLASK_PARAMS["max_ball_vel"]
        },
        weight=KLASK_PARAMS["rewards"]["goal_scored"] / (KLASK_PARAMS["decimation"] * KLASK_PARAMS["physics_dt"])
    )

    goal_conceded = RewTerm(
        func=ball_in_goal, 
        params={
            "asset_cfg": SceneEntityCfg("ball"),
            "goal": KLASK_PARAMS["player_goal"],
            "max_ball_vel": KLASK_PARAMS["max_ball_vel"]
        },
        weight=KLASK_PARAMS["rewards"]["goal_conceded"] / (KLASK_PARAMS["decimation"] * KLASK_PARAMS["physics_dt"])
    )

    distance_player_ball = RewTerm(
        func=distance_player_ball, 
        params={
            "player_cfg": SceneEntityCfg("klask", body_names=["Peg_1"]),
            "ball_cfg": SceneEntityCfg("ball"),
        },
        weight=KLASK_PARAMS["rewards"]["distance_player_goal"] / (KLASK_PARAMS["decimation"] * KLASK_PARAMS["physics_dt"])
    )

    distance_player_ball_own_half = RewTerm(
        func=distance_player_ball_own_half, 
        params={
            "player_cfg": SceneEntityCfg("klask", body_names=["Peg_1"]),
            "ball_cfg": SceneEntityCfg("ball"),
        },
        weight=KLASK_PARAMS["rewards"]["distance_player_ball_own_half"] / (KLASK_PARAMS["decimation"] * KLASK_PARAMS["physics_dt"])
    )

    distance_ball_opponent_goal = RewTerm(
        func=distance_ball_goal, 
        params={
            "ball_cfg": SceneEntityCfg("ball"),
            "goal": KLASK_PARAMS["opponent_goal"]
        },
        weight=KLASK_PARAMS["rewards"]["distance_ball_opponent_goal"] / (KLASK_PARAMS["decimation"] * KLASK_PARAMS["physics_dt"])
    )

    ball_speed = RewTerm(
        func=ball_speed, 
        params={
            "ball_cfg": SceneEntityCfg("ball"),
        },
        weight=KLASK_PARAMS["rewards"]["ball_speed"] / (KLASK_PARAMS["decimation"] * KLASK_PARAMS["physics_dt"])
    )

    ball_stationary = RewTerm(
        func=ball_stationary, 
        params={
            "ball_cfg": SceneEntityCfg("ball"),
        },
        weight=KLASK_PARAMS["rewards"]["ball_stationary"] / (KLASK_PARAMS["decimation"] * KLASK_PARAMS["physics_dt"])
    )

    collision_player_ball = RewTerm(
        func=collision_player_ball, 
        params={
            "player_cfg": SceneEntityCfg("klask", body_names=["Peg_1"]),
            "ball_cfg": SceneEntityCfg("ball"),
        },
        weight=KLASK_PARAMS["rewards"]["collision_player_ball"] / (KLASK_PARAMS["decimation"] * KLASK_PARAMS["physics_dt"])
    )

    ball_in_own_half = RewTerm(
        func=ball_in_own_half, 
        params={
            "ball_cfg": SceneEntityCfg("ball")
        },
        weight=KLASK_PARAMS["rewards"]["ball_in_own_half"]
    )

    
@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    if KLASK_PARAMS["terminations"]["goal_scored"]:
        goal_scored = DoneTerm(
            func=ball_in_goal, 
            params={
                "asset_cfg": SceneEntityCfg("ball"),
                "goal": KLASK_PARAMS["opponent_goal"]
            }
        )

    if KLASK_PARAMS["terminations"]["goal_conceded"]:
        goal_conceded = DoneTerm(
            func=ball_in_goal, 
            params={
                "asset_cfg": SceneEntityCfg("ball"),
                "goal": KLASK_PARAMS["player_goal"]
            }
        )

    if KLASK_PARAMS["terminations"]["player_in_goal"]:
        player_in_goal = DoneTerm(
            func=in_goal, 
            params={
                "asset_cfg": SceneEntityCfg("klask", body_names=["Peg_1"]),
                "goal": KLASK_PARAMS["player_goal"]
            }
        )


@configclass
class CurriculumCfg:
    pass
    

@configclass
class KlaskEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the cartpole environment."""

    sim = SimulationCfg(physx=PhysxCfg(bounce_threshold_velocity=0.0))
    # Scene settings
    scene = KlaskSceneCfg(num_envs=1, env_spacing=1.0)
    # Basic settings
    observations = ObservationsCfg()
    actions = ActionsCfg()
    events = EventCfg()
    rewards = RewardsCfg()
    terminations = TerminationsCfg()
    episode_length_s = 5.0

    def __post_init__(self):
        """Post initialization."""
        # viewer settings
        self.viewer.eye = (0.0, 0.0, 2.0)
        self.viewer.lookat = (0.0, 0.0, 0.0)
        # step settings
        self.decimation = KLASK_PARAMS['decimation']  # env step every 4 sim steps: 200Hz / 4 = 50Hz
        # simulation settings
        self.sim.dt = KLASK_PARAMS['physics_dt']  # sim step every 5ms: 200Hz


@configclass
class KlaskHistoryEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the cartpole environment."""

    sim = SimulationCfg(physx=PhysxCfg(bounce_threshold_velocity=0.0))
    # Scene settings
    scene = KlaskSceneCfg(num_envs=1, env_spacing=1.0)
    # Basic settings
    observations = ObservationsCfg()
    actions = ActionsCfg()
    events = EventCfg()
    rewards = RewardsCfg()
    terminations = TerminationsCfg()
    episode_length_s = 5.0

    def __post_init__(self):
        """Post initialization."""
        # viewer settings
        self.viewer.eye = (0.0, 0.0, 2.0)
        self.viewer.lookat = (0.0, 0.0, 0.0)
        # step settings
        self.decimation = KLASK_PARAMS['decimation']  # env step every 4 sim steps: 200Hz / 4 = 50Hz
        # simulation settings
        self.sim.dt = KLASK_PARAMS['physics_dt']  # sim step every 5ms: 200Hz


@configclass
class KlaskGoalEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the cartpole environment."""

    sim = SimulationCfg(physx=PhysxCfg(bounce_threshold_velocity=0.0))
    # Scene settings
    scene = KlaskSceneCfg(num_envs=1, env_spacing=1.0)
    # Basic settings
    observations = GoalObservationsCfg()
    actions = ActionsCfg()
    events = EventCfg()
    rewards = RewardsCfg()
    terminations = TerminationsCfg()
    episode_length_s = 5.0

    def __post_init__(self):
        """Post initialization."""
        # viewer settings
        self.viewer.eye = (0.0, 0.0, 6.0)
        self.viewer.lookat = (0.0, 0.0, 0.0)
        # step settings
        self.decimation = KLASK_PARAMS['decimation']  # env step every 4 sim steps: 200Hz / 4 = 50Hz
        # simulation settings
        self.sim.dt = KLASK_PARAMS['physics_dt']  # sim step every 5ms: 200Hz
        