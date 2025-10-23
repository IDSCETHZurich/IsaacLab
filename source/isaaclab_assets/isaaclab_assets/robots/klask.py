import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

KLASK_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=os.path.join(
            os.getcwd(), "source/isaaclab_assets/isaaclab_assets/robots/klask.usd"
        ),
    ),
    actuators={
        "peg_1x_actuator": IdealPDActuatorCfg(
            joint_names_expr=["slider_to_peg_1"],
            stiffness=0.0,
            damping=10.0,
            velocity_limit=3.0,
            effort_limit=30.0,
        ),
        "peg_1y_actuator": IdealPDActuatorCfg(
            joint_names_expr=["ground_to_slider_1"],
            stiffness=0.0,
            damping=100.0,
            velocity_limit=3.0,
            effort_limit=300.0,
        ),
        "peg_2x_actuator": IdealPDActuatorCfg(
            joint_names_expr=["slider_to_peg_2"],
            stiffness=0.0,
            damping=10.0,
            velocity_limit=3.0,
            effort_limit=30.0,
        ),
        "peg_2y_actuator": IdealPDActuatorCfg(
            joint_names_expr=["ground_to_slider_2"],
            stiffness=0.0,
            damping=100.0,
            velocity_limit=3.0,
            effort_limit=300.0,
        ),
    },
)
