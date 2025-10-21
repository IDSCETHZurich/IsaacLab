import isaaclab.envs.mdp as mdp
from isaaclab.utils import configclass


@configclass
class ActionsCfg:
    player_x = mdp.JointVelocityActionCfg(
        asset_name="klask", joint_names=["slider_to_peg_1"]
    )

    player_y = mdp.JointVelocityActionCfg(
        asset_name="klask", joint_names=["ground_to_slider_1"]
    )

    opponent_x = mdp.JointVelocityActionCfg(
        asset_name="klask", joint_names=["slider_to_peg_2"]
    )

    opponent_y = mdp.JointVelocityActionCfg(
        asset_name="klask", joint_names=["ground_to_slider_2"]
    )
