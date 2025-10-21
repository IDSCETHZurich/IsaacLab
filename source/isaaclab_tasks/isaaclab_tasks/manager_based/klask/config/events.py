import isaaclab.envs.mdp as mdp
from isaaclab.managers import EventTermCfg, SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab_assets.robots.klask import KLASK_PARAMS

from ..utils_manager_based import reset_joints_by_offset


@configclass
class EventCfg:
    if KLASK_PARAMS["domain_randomization"]["use_domain_randomization"]:
        # on reset
        add_ball_mass = EventTermCfg(
            func=mdp.randomize_rigid_body_mass,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("ball"),
                "mass_distribution_params": KLASK_PARAMS["domain_randomization"][
                    "ball_mass_range"
                ],
                "operation": "abs",
            },
        )

        randomize_material_ball = EventTermCfg(
            func=mdp.randomize_rigid_body_material,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("ball"),
                "static_friction_range": KLASK_PARAMS["domain_randomization"][
                    "static_friction_range"
                ],
                "dynamic_friction_range": KLASK_PARAMS["domain_randomization"][
                    "dynamic_friction_range"
                ],
                "restitution_range": KLASK_PARAMS["domain_randomization"][
                    "restitution_range"
                ],
                "num_buckets": 100,
                "make_consistent": True,
            },
        )

        randomize_material_klask = EventTermCfg(
            func=mdp.randomize_rigid_body_material,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("klask"),
                "static_friction_range": KLASK_PARAMS["domain_randomization"][
                    "static_friction_range"
                ],
                "dynamic_friction_range": KLASK_PARAMS["domain_randomization"][
                    "dynamic_friction_range"
                ],
                "restitution_range": KLASK_PARAMS["domain_randomization"][
                    "restitution_range"
                ],
                "num_buckets": 100,
                "make_consistent": True,
            },
        )

        randomize_actuator = EventTermCfg(
            func=mdp.randomize_actuator_gains,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("klask"),
                "stiffness_distribution_params": KLASK_PARAMS["domain_randomization"][
                    "stiffness_range"
                ],
                "damping_distribution_params": KLASK_PARAMS["domain_randomization"][
                    "damping_range"
                ],
                "operation": "abs",
            },
        )

    reset_x_position_peg_1 = EventTermCfg(
        func=reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("klask", joint_names=["slider_to_peg_1"]),
            # "position_range": (0.0202, 0.0202),
            # "velocity_range": (0.086, 0.086)
            "position_range": (-0.1, 0.1),
            "velocity_range": (0.0, 0.0),
        },
    )

    reset_x_position_peg_2 = EventTermCfg(
        func=reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("klask", joint_names=["slider_to_peg_2"]),
            "position_range": (-0.1, 0.1),
            "velocity_range": (0.0, 0.0),
        },
    )

    reset_y_position_peg_1 = EventTermCfg(
        func=reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("klask", joint_names=["ground_to_slider_1"]),
            "position_range": (-0.14, -0.03),
            # "position_range": (-0.1103, -0.1103),
            # "velocity_range": (-0.0043, -0.0043)
            "velocity_range": (0.0, 0.0),
        },
    )

    reset_y_position_peg_2 = EventTermCfg(
        func=reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("klask", joint_names=["ground_to_slider_2"]),
            "position_range": (0.03, 0.14),
            "velocity_range": (0.0, 0.0),
        },
    )

    reset_ball_position = EventTermCfg(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("ball"),
            "pose_range": {
                "x": KLASK_PARAMS["ball_reset_position_x"],
                "y": KLASK_PARAMS["ball_reset_position_y"],
                "z": (0.032, 0.032),
            },
            "velocity_range": {"x": (-0.0, 0.0), "y": (-0.00, 0.00)},
        },
    )
