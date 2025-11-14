# hexapod.py
from pathlib import Path
import os
import math
import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg, ImplicitActuatorCfg 
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR  # or your own base dir

#
# 1. Actuator config for all 18 leg joints
#

HEXAPOD_DC_MOTOR_CFG = DCMotorCfg(
    # all revolute joints are the legs in your URDF:
    #  base[1-6]_rotate, leg[1-6]_thigh, leg[1-6]_foot
    joint_names_expr=[
        "base[1-6]_rotate",
        "leg[1-6]_thigh",
        "leg[1-6]_foot",
    ],

    # --- torque / speed limits ---
    # URDF says effort="50" and velocity="10" for all joints.
    # That's pretty big, so start smaller for stable training and increase later.
    saturation_effort=5.0,   # hard saturation (N·m)
    effort_limit=3.0,        # PD torque limit used in torque-speed model
    velocity_limit=10.0,     # rad/s  (matches your URDF limit)

    # --- PD gains (position-based control under the hood) ---
    stiffness={".*": 30.0},  # Kp
    damping={".*": 1.0},     # Kd
)

HEXAPOD_IMPLICIT_ACTUATOR_CFG = ImplicitActuatorCfg(
    joint_names_expr=[
        "base[1-6]_rotate",
        "leg[1-6]_thigh",
        "leg[1-6]_foot",
    ],
    effort_limit_sim=3.0,
    velocity_limit_sim=10.0,
    stiffness=30.0,
    damping=1.0,
    friction=0.01,
)

#
# 2. Articulation config for the hexapod body
#

HEXAPOD_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{Path(__file__).parent}/kai_robot/six_leg_urdffile/six_leg_urdffile.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
        # You can add collision_props here if needed
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.01, rest_offset=0.0),
    ),

    # Start slightly above ground in a neutral-ish pose
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.04),  # adjust if it spawns intersecting the ground
        # regexes work like in ANYmal: all rotate/thigh/foot joints to 0 rad
        joint_pos={
            ".*_rotate": 0.0,
            ".*_thigh": 0.0,
            ".*_foot": 0.0,
        },
    ),

    actuators={
        # "legs": HEXAPOD_DC_MOTOR_CFG,
        "legs": HEXAPOD_IMPLICIT_ACTUATOR_CFG,
    },

    # keep some margin away from the hard URDF limits
    soft_joint_pos_limit_factor=0.95,
)
