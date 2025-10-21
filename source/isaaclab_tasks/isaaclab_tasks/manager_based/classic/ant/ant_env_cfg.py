# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTermCfg
from isaaclab.managers import ObservationGroupCfg as ObservationGroupCfg
from isaaclab.managers import ObservationTermCfg as ObservationTermCfg
from isaaclab.managers import RewardTermCfg as RewardTermCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as TerminationTermCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.classic.humanoid.mdp as mdp

##
# Pre-defined configs
##
from isaaclab_assets.robots.ant import ANT_CFG  # isort: skip


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with an ant robot."""

    # terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # robot
    robot = ANT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_effort = mdp.JointEffortActionCfg(
        asset_name="robot", joint_names=[".*"], scale=7.5
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObservationGroupCfg):
        """Observations for the policy."""

        base_height = ObservationTermCfg(func=mdp.base_pos_z)
        base_lin_vel = ObservationTermCfg(func=mdp.base_lin_vel)
        base_ang_vel = ObservationTermCfg(func=mdp.base_ang_vel)
        base_yaw_roll = ObservationTermCfg(func=mdp.base_yaw_roll)
        base_angle_to_target = ObservationTermCfg(
            func=mdp.base_angle_to_target, params={"target_pos": (1000.0, 0.0, 0.0)}
        )
        base_up_proj = ObservationTermCfg(func=mdp.base_up_proj)
        base_heading_proj = ObservationTermCfg(
            func=mdp.base_heading_proj, params={"target_pos": (1000.0, 0.0, 0.0)}
        )
        joint_pos_norm = ObservationTermCfg(func=mdp.joint_pos_limit_normalized)
        joint_vel_rel = ObservationTermCfg(func=mdp.joint_vel_rel, scale=0.2)
        feet_body_forces = ObservationTermCfg(
            func=mdp.body_incoming_wrench,
            scale=0.1,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=[
                        "front_left_foot",
                        "front_right_foot",
                        "left_back_foot",
                        "right_back_foot",
                    ],
                )
            },
        )
        actions = ObservationTermCfg(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_base = EventTermCfg(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={"pose_range": {}, "velocity_range": {}},
    )

    reset_robot_joints = EventTermCfg(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.2, 0.2),
            "velocity_range": (-0.1, 0.1),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # (1) Reward for moving forward
    progress = RewardTermCfg(
        func=mdp.progress_reward, weight=1.0, params={"target_pos": (1000.0, 0.0, 0.0)}
    )
    # (2) Stay alive bonus
    alive = RewardTermCfg(func=mdp.is_alive, weight=0.5)
    # (3) Reward for non-upright posture
    upright = RewardTermCfg(
        func=mdp.upright_posture_bonus, weight=0.1, params={"threshold": 0.93}
    )
    # (4) Reward for moving in the right direction
    move_to_target = RewardTermCfg(
        func=mdp.move_to_target_bonus,
        weight=0.5,
        params={"threshold": 0.8, "target_pos": (1000.0, 0.0, 0.0)},
    )
    # (5) Penalty for large action commands
    action_l2 = RewardTermCfg(func=mdp.action_l2, weight=-0.005)
    # (6) Penalty for energy consumption
    energy = RewardTermCfg(
        func=mdp.power_consumption, weight=-0.05, params={"gear_ratio": {".*": 15.0}}
    )
    # (7) Penalty for reaching close to joint limits
    joint_limits = RewardTermCfg(
        func=mdp.joint_limits_penalty_ratio,
        weight=-0.1,
        params={"threshold": 0.99, "gear_ratio": {".*": 15.0}},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Terminate if the episode length is exceeded
    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)
    # (2) Terminate if the robot falls
    torso_height = TerminationTermCfg(
        func=mdp.root_height_below_minimum, params={"minimum_height": 0.31}
    )


@configclass
class AntEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the MuJoCo-style Ant walking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=5.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 16.0
        # simulation settings
        self.sim.dt = 1 / 120.0
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.2
        # default friction material
        self.sim.physics_material.static_friction = 1.0
        self.sim.physics_material.dynamic_friction = 1.0
        self.sim.physics_material.restitution = 0.0
