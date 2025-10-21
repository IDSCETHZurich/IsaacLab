import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab_assets.robots.klask import KLASK_CFG, KLASK_PARAMS


@configclass
class SceneCfg(InteractiveSceneCfg):
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )

    ball = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Ball",
        spawn=sim_utils.SphereCfg(
            radius=0.007,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(
                mass=KLASK_PARAMS["ball_mass_initial"]
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.0, 1.0, 0.0), metallic=0.2
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                restitution=KLASK_PARAMS["ball_restitution"],
                static_friction=KLASK_PARAMS["ball_static_friction"],
                dynamic_friction=KLASK_PARAMS["ball_dynamic_friction"],
            ),
            activate_contact_sensors=True,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.032)),
    )

    # contact_sensor = ContactSensorCfg(
    #    prim_path="{ENV_REGEX_NS}/Ball",
    #    filter_prim_paths_expr=["{ENV_REGEX_NS}/Klask/Peg_1"],
    #    history_length=KLASK_PARAMS["decimation"]
    # )

    klask = KLASK_CFG.replace(prim_path="{ENV_REGEX_NS}/Klask")
