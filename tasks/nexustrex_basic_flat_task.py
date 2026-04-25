# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

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
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from .mdp.terminations import joint_pos_out_of_manual_limit

##
# Pre-defined configs
##
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(project_root)
from assets.NexusTrex_CFG import NexusTrex_CONFIG  # Scene definition


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the flat scene."""

    # 1. Flat ground plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, 0]),
        spawn=GroundPlaneCfg(),
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    # 2. Robot configuration
    # Override initial state so the robot spawns in a standing pose
    robot: ArticulationCfg = NexusTrex_CONFIG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=NexusTrex_CONFIG.spawn.replace(activate_contact_sensors=True),
        init_state=ArticulationCfg.InitialStateCfg(
            # Raise spawn height to leave room for legs to extend
            pos=(0.0, 0.0, 0.60),
            # Initial leg joint angle (0.5 rad) to avoid a folded pose.
            # ".*Passive" matches DriveLeftPassive and DriveRighttPassive
            # ".*Motor" matches DriveLeftMotor and DriveRightMotor
            # If the robot faces the wrong way, try -0.5 instead of 0.5
            joint_pos={
                ".*Passive": 0.5,
                ".*Motor": 0.0
            },
            joint_vel={".*": 0.0}
        ),
    )

    # 3. Contact sensor
    contact_sensor = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/BipedalWheeledRobot/.*",
        history_length=3,
        track_air_time=True
    )


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(0.0, 0.0),  # Two-wheeled base: no lateral command
            ang_vel_z=(-1.0, 1.0),
            heading=(-math.pi, math.pi),
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # Larger scale allows bigger leg motion
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

    # Right wheel velocity
    right_wheel_vel = mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=["MotorRight"],
        scale=10.0,
        use_default_offset=True,
    )
    # Left wheel velocity (negated for mirrored mounting)
    left_wheel_vel = mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=["MotorLeft"],
        scale=-10.0,  # Negative scale fixes mirrored wheel direction
        use_default_offset=True,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        base_orientation = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "base_velocity"}
        )
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""
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
    """Reward terms for the MDP."""

    # 1. Base survival reward
    alive = RewTerm(func=mdp.is_alive, weight=1.0)

    # Removed terminated reward (caused "Not all regular expressions are matched").
    # TerminationCfg no longer uses out_of_joint_limit; drop matching penalty here.

    # 2. Task rewards
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=0.5,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )

    # 3. Shaping and penalties

    # Encourage maintaining this base height (avoid crouching)
    base_height_l2 = RewTerm(
        func=mdp.base_height_l2,
        weight=-1.0,
        params={"target_height": 0.45}
    )

    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.1)

    # Penalize chassis ground contact
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-10.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names="chassis_base"), "threshold": 1.0},
    )

    # Orientation: stay level, avoid tipping backward
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-0.5)

    # Damp aggressive pitch/roll rates
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    command_update = DoneTerm(
        func=mdp.command_resample, params={"command_name": "base_velocity"}
    )

    # Removed out_of_joint_limit (episodes died in ~5 steps). Free leg extension.


@configclass
class CurriculumCfg:
    pass


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

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 10.0
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2 ** 15
        if self.scene.contact_sensor is not None:
            self.scene.contact_sensor.update_period = self.sim.dt