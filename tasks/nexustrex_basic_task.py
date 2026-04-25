# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause
#
# nexustrex_basic_task.py - Wheeled-legged switching: wheels on flat ground, legs over obstacles

import math
import os, sys
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.managers import CurriculumTermCfg as CurrTerm

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

# Wheeled-leg rewards: obstacle leg lift + global home (no duplicate of joint_default_pos)
from .mdp.custom_rewards import (
    obstacle_leg_lift,
    joint_default_position_reward,
)

from .mdp.terminations import joint_pos_out_of_manual_limit

from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(project_root)
from assets.NexusTrex_CFG import NexusTrex_CONFIG
from .mdp.terrain_cfg import NEXUS_ROUGH_TERRAINS_CFG


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=NEXUS_ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    robot: ArticulationCfg = NexusTrex_CONFIG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=NexusTrex_CONFIG.spawn.replace(activate_contact_sensors=True),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.45),  # Higher spawn to match legs at default
            joint_pos={
                ".*Passive": 0.0,  # All leg joints at default
                ".*Motor": 0.0
            },
            joint_vel={".*": 0.0}
        ),
    )

    contact_sensor = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/BipedalWheeledRobot/.*",
        history_length=3,
        track_air_time=True
    )

    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/BipedalWheeledRobot/chassis_base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.2, 0.0, 0.5)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.2, 0.6]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.05,  # Fewer standing episodes
        rel_heading_envs=0.0,
        heading_command=False,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.3, 0.8),  # Higher forward speed range
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(-0.3, 0.3),
            heading=(-math.pi, math.pi),
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    left_leg_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["DriveLeftMotor", "DriveLeftPassive"],
        scale=1.0,
        use_default_offset=True,
    )
    right_leg_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["DriveRightMotor", "DriveRighttPassive"],
        scale=1.0,
        use_default_offset=True,
    )

    right_wheel_vel = mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=["MotorRight"],
        scale=10.0,
        use_default_offset=True,
    )
    left_wheel_vel = mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=["MotorLeft"],
        scale=-10.0,
        use_default_offset=True,
    )


@configclass
class ObservationsCfg:
    """Observation specs with clipping on all terms to avoid numerical blow-up."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, clip=(-5.0, 5.0))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, clip=(-5.0, 5.0))
        base_orientation = ObsTerm(func=mdp.projected_gravity, clip=(-1.0, 1.0))
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "base_velocity"},
            clip=(-2.0, 2.0)
        )
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, clip=(-3.14, 3.14))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, clip=(-20.0, 20.0))
        actions = ObsTerm(func=mdp.last_action, clip=(-5.0, 5.0))

        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.01, n_max=0.01),
            clip=(-0.5, 0.5),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""
    # --- Light domain randomization: sim2real benefit without ruining stairs ---
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 1.0),
            "dynamic_friction_range": (0.7, 0.9),
            "restitution_range": (0.0, 0.02),
            "num_buckets": 24,
        },
    )
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="chassis_base"),
            "mass_distribution_params": (-0.5, 0.5),
            "operation": "add",
        },
    )
    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="chassis_base"),
            "com_range": {"x": (-0.01, 0.01), "y": (-0.01, 0.01), "z": (-0.008, 0.008)},
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.06, 0.06),
                "y": (-0.06, 0.06),
                "z": (0.0, 0.0),
                "roll": (-0.04, 0.04),
                "pitch": (-0.04, 0.04),
                "yaw": (-0.05, 0.05),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.995, 1.005),
            "velocity_range": (0.0, 0.0),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms (simplified; conflicting terms removed)."""

    # ==================== Survival ====================
    alive = RewTerm(func=mdp.is_alive, weight=2.0)  # Stronger alive bonus

    # Termination penalty disabled for more exploration
    # terminated = RewTerm(
    #     func=mdp.is_terminated_term,
    #     weight=-20.0,
    #     params={"term_keys": "bad_orientation"}
    # )

    # ==================== Primary: velocity tracking ====================
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )

    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )

    # ==================== Wheeled-leg: lift legs on obstacles ====================
    leg_lift = RewTerm(
        func=obstacle_leg_lift,
        weight=1.5,
        params={
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "min_lift": 0.0,
            "command_name": "base_velocity",
        },
    )

    # ==================== Gait: joint home (pairs with leg_lift: lift on obstacles, home on flat) ===
    joint_default_pos = RewTerm(
        func=joint_default_position_reward,
        weight=0.8,
        params={"sigma": 0.25},
    )

    # ==================== Stability ====================
    # Stronger orientation penalty (reduce backward lean)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-3.0)

    # No fixed base height — chassis can move with legs
    # base_height_l2 = RewTerm(
    #     func=mdp.base_height_l2,
    #     weight=-0.2,
    #     params={"target_height": 0.30}
    # )

    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.5,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names="chassis_base"), "threshold": 1.0},
    )

    # ==================== Smoothness ====================
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.3)

    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.1)

    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.02)

    joint_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-1e-6)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    command_update = DoneTerm(
        func=mdp.command_resample, params={"command_name": "base_velocity"}
    )

    # Chassis contact termination disabled for learning
    # base_contact = DoneTerm(
    #     func=mdp.illegal_contact,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_sensor", body_names="chassis_base"),
    #         "threshold": 10.0,
    #     },
    # )

    # Looser orientation limit while learning to stand
    bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": 1.57},  # 90 deg
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""
    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)


@configclass
class NexusTrexBasicEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 10.0
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2 ** 15
        if self.scene.contact_sensor is not None:
            self.scene.contact_sensor.update_period = self.sim.dt

        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False
