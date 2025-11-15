# hexapod_env_cfg.py
# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

##
# Pre-defined configs
##
from ....assets.hexapod import HEXAPOD_CFG       # <--- use your hexapod.py
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip


@configclass
class EventCfg:
    """Configuration for randomization."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            # base link name from your URDF
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "mass_distribution_params": (-0.3, 0.3),
            "operation": "add",
        },
    )


@configclass
class HexapodFlatEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 20.0
    decimation = 4
    action_scale = 5.0

    # 18 joints → action dim 18 (one command per joint)
    action_space = 18

    # Observation vector (flat env) consists of:
    #   root_lin_vel_b (3)
    #   root_ang_vel_b (3)
    #   projected_gravity_b (3)
    #   commands (3)
    #   joint_pos - default (18)
    #   joint_vel (18)
    #   actions (18)
    # total = 3 + 3 + 3 + 3 + 18 + 18 + 18 = 66
    observation_space = 66

    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    # events
    events: EventCfg = EventCfg()

    # robot
    robot: ArticulationCfg = HEXAPOD_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*", history_length=3, update_period=0.005, track_air_time=True
    )

    # reward scales (start by copying ANYmal’s)
    lin_vel_reward_scale = 1.0
    # yaw_rate_reward_scale = 0.5
    yaw_rate_reward_scale = 1.0
    z_vel_reward_scale = -2.0
    ang_vel_reward_scale = -0.05
    # joint_torque_reward_scale = -2.5e-5
    # joint_accel_reward_scale = -2.5e-7
    # action_rate_reward_scale = -0.01
    feet_air_time_reward_scale = 0.5
    undesired_contact_reward_scale = -1.0
    # flat_orientation_reward_scale = -5.0
    base_contact_reward_scale = -0.1

    joint_torque_reward_scale = -1.0e-5
    joint_accel_reward_scale = -1.0e-7
    action_rate_reward_scale = -5.0e-3
    flat_orientation_reward_scale = -2.0
    heading_reward_scale = 1.0 



@configclass
class HexapodRoughEnvCfg(HexapodFlatEnvCfg):
    # env
    # Rough env adds height scanner info: ANYmal goes 48 -> 235 (+187 dims),
    # here: 66 + 187 = 253
    observation_space = 253

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=9,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
        debug_vis=False,
    )

    # # we add a height scanner for perceptive locomotion
    # height_scanner = RayCasterCfg(
    #     prim_path="/World/envs/env_.*/Robot/base_link",  # base link prim
    #     offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
    #     ray_alignment="yaw",
    #     pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
    #     debug_vis=False,
    #     mesh_prim_paths=["/World/ground"],
    # )

    # reward scales (override from flat config)
    flat_orientation_reward_scale = 0.0
