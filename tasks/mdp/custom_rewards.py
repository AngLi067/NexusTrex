# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

"""Custom reward functions for the NexusTrex wheeled-legged robot.
- On flat ground: use wheels; keep legs stable
- On obstacles/steps: use legs to traverse
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _get_terrain_info(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg):
    """Terrain features from the height scanner (numerically safe).

    Returns:
        terrain_roughness: Height variation (std dev)
        front_height_diff: Max height ahead relative to current pose
        is_flat: Whether terrain is flat
        obstacle_ahead: Whether an obstacle is ahead
    """
    height_scanner: RayCaster = env.scene.sensors[sensor_cfg.name]
    heights = height_scanner.data.ray_hits_w[..., 2]  # (num_envs, num_rays)
    
    # Numerical safety: replace NaN/Inf
    heights = torch.nan_to_num(heights, nan=0.0, posinf=10.0, neginf=-10.0)
    heights = torch.clamp(heights, min=-10.0, max=10.0)
    
    # Robot height
    robot = env.scene["robot"]
    current_height = robot.data.root_pos_w[:, 2]  # (num_envs,)
    current_height = torch.clamp(current_height, min=-10.0, max=10.0)
    
    # Relative heights
    relative_heights = heights - current_height.unsqueeze(-1)
    relative_heights = torch.clamp(relative_heights, min=-5.0, max=5.0)
    
    # Terrain roughness (std)
    terrain_roughness = relative_heights.std(dim=-1)
    terrain_roughness = torch.clamp(terrain_roughness, min=0.0, max=1.0)
    
    # Forward height diff (max over front half of rays)
    num_rays = heights.shape[-1]
    front_heights = relative_heights[:, :num_rays // 2]  # front half
    front_height_diff = front_heights.max(dim=-1)[0]
    front_height_diff = torch.clamp(front_height_diff, min=-1.0, max=1.0)
    
    # Flat if roughness below threshold
    flat_threshold = 0.03  # ~3 cm
    is_flat = terrain_roughness < flat_threshold
    
    # Obstacle if forward height diff above threshold
    obstacle_threshold = 0.02  # ~2 cm
    obstacle_ahead = front_height_diff > obstacle_threshold
    
    return terrain_roughness, front_height_diff, is_flat, obstacle_ahead


def _get_leg_positions(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg):
    """Leg joint positions (passive joints: vertical lift)."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos
    
    leg_joint_names = ['DriveLeftPassive', 'DriveRighttPassive']
    leg_indices = []
    for name in leg_joint_names:
        if name in asset.joint_names:
            leg_indices.append(asset.joint_names.index(name))
    
    if len(leg_indices) > 0:
        leg_positions = joint_pos[:, leg_indices]
        return leg_positions, leg_indices
    else:
        return None, []


def _get_all_leg_positions(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg):
    """All leg joint positions (motor + passive: fore/aft and lift)."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos
    
    # Motor (fore/aft) and passive (lift)
    leg_joint_names = ['DriveLeftMotor', 'DriveLeftPassive', 'DriveRightMotor', 'DriveRighttPassive']
    leg_indices = []
    for name in leg_joint_names:
        if name in asset.joint_names:
            leg_indices.append(asset.joint_names.index(name))
    
    if len(leg_indices) > 0:
        leg_positions = joint_pos[:, leg_indices]
        return leg_positions, leg_indices
    else:
        return None, []


# ============== Core rewards ==============

def flat_ground_wheel_mode(
        env: ManagerBasedRLEnv,
        sensor_cfg: SceneEntityCfg,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        leg_stability_threshold: float = 0.1,
        target_position: float = 0.0,  # Target: all joints at zero
) -> torch.Tensor:
    """Reward stable legs on flat ground (wheel mode).

    On flat terrain, reward keeping all leg joints (motor+passive) near target.
    """
    _, _, is_flat, _ = _get_terrain_info(env, sensor_cfg)
    # All leg joints (fore/aft + lift)
    leg_positions, leg_indices = _get_all_leg_positions(env, asset_cfg)
    
    if leg_positions is None:
        return torch.zeros(env.num_envs, device=env.device)
    
    # Stability: closer to target is better
    leg_deviation = torch.abs(leg_positions - target_position).mean(dim=-1)
    stability_reward = torch.exp(-leg_deviation / leg_stability_threshold)
    
    # Only on flat ground
    reward = stability_reward * is_flat.float()
    
    return reward


def obstacle_leg_lift(
        env: ManagerBasedRLEnv,
        sensor_cfg: SceneEntityCfg,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        min_lift: float = 0.0,
        command_name: str = "base_velocity",
) -> torch.Tensor:

    terrain_roughness, front_height_diff, is_flat, obstacle_ahead = _get_terrain_info(env, sensor_cfg)
    leg_positions, leg_indices = _get_leg_positions(env, asset_cfg)
    
    if leg_positions is None:
        return torch.zeros(env.num_envs, device=env.device)
    
    # Numerical safety
    leg_positions = torch.clamp(leg_positions, min=-3.14, max=3.14)
    
    # Obstacle strength
    obstacle_strength = torch.clamp(front_height_diff / 0.05, min=0.0, max=1.0)
    
    # Lift magnitude
    lift_amount = torch.abs(leg_positions).max(dim=-1)[0]
    lift_amount = torch.clamp(lift_amount, min=0.0, max=2.0)
    
    # Lift reward
    lift_reward = 1.0 - torch.exp(-lift_amount * 3.0)
    lift_reward = torch.clamp(lift_reward, min=0.0, max=1.0)
    
    # Weight by obstacle strength
    reward = lift_reward * obstacle_strength
    reward = torch.clamp(reward, min=0.0, max=1.0)
    
    # Only when a motion command is active
    command = env.command_manager.get_command(command_name)
    moving = torch.norm(command[:, :2], dim=1) > 0.1
    reward *= moving.float()
    
    return reward


def obstacle_chassis_lift_penalty(
        env: ManagerBasedRLEnv,
        sensor_cfg: SceneEntityCfg,
        target_height: float = 0.25,
        lift_margin: float = 0.05,
) -> torch.Tensor:
    """Penalize chassis lift on obstacles; encourage leg lift instead of pushing through."""
    # Obstacle strength (more sensitive)
    _, front_height_diff, _, _ = _get_terrain_info(env, sensor_cfg)
    obstacle_strength = torch.clamp(front_height_diff / 0.04, min=0.0, max=1.0)

    # Chassis height
    robot = env.scene["robot"]
    current_height = robot.data.root_pos_w[:, 2]

    # Penalize height above target + margin
    excess_height = torch.clamp(current_height - (target_height + lift_margin), min=0.0)
    penalty = excess_height * obstacle_strength

    return penalty


def unnecessary_leg_lift_penalty(
        env: ManagerBasedRLEnv,
        sensor_cfg: SceneEntityCfg,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        penalty_threshold: float = 0.15,
        target_position: float = 0.0,  # Target: all joints at zero
) -> torch.Tensor:
    """Penalize leg deviation from default on flat ground.

    When flat, penalize motor+passive joints away from the initial pose.
    """
    _, _, is_flat, _ = _get_terrain_info(env, sensor_cfg)
    # All leg joints (fore/aft + lift)
    leg_positions, leg_indices = _get_all_leg_positions(env, asset_cfg)
    
    if leg_positions is None:
        return torch.zeros(env.num_envs, device=env.device)
    
    # Max deviation from default across joints
    deviation = torch.abs(leg_positions - target_position).max(dim=-1)[0]
    unnecessary_lift = torch.clamp(deviation - penalty_threshold, min=0.0)
    
    # Only on flat ground
    penalty = unnecessary_lift * is_flat.float()
    
    return penalty


def joint_default_position_reward(
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        sigma: float = 0.2,
) -> torch.Tensor:
    """Global reward: pull all leg joints toward default pose.

    Terrain-agnostic. Smaller sigma = sharper penalty away from default.
    """
    all_leg_pos, _ = _get_all_leg_positions(env, asset_cfg)
    
    if all_leg_pos is None:
        return torch.zeros(env.num_envs, device=env.device)
    
    deviation_sq = (all_leg_pos ** 2).sum(dim=-1)
    reward = torch.exp(-deviation_sq / (sigma ** 2))
    
    return reward


def terrain_adaptive_behavior(
        env: ManagerBasedRLEnv,
        sensor_cfg: SceneEntityCfg,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        command_name: str = "base_velocity",
) -> torch.Tensor:
    """Combined terrain-adaptive wheeled-leg behavior reward.

    - Flat + stable legs: high
    - Obstacle + lifted legs: high
    - Flat + restless legs: low
    - Obstacle + static legs: low
    """
    terrain_roughness, front_height_diff, is_flat, obstacle_ahead = _get_terrain_info(env, sensor_cfg)
    leg_positions, leg_indices = _get_leg_positions(env, asset_cfg)
    
    if leg_positions is None:
        return torch.zeros(env.num_envs, device=env.device)
    
    # Leg state
    leg_deviation = torch.abs(leg_positions).mean(dim=-1)
    legs_stable = leg_deviation < 0.1
    legs_lifted = leg_deviation > 0.15

    # Correct-mode bonuses
    correct_flat = is_flat & legs_stable
    correct_obstacle = obstacle_ahead & legs_lifted
    
    reward = correct_flat.float() + correct_obstacle.float()
    
    # Only while moving
    command = env.command_manager.get_command(command_name)
    moving = torch.norm(command[:, :2], dim=1) > 0.1
    reward *= moving.float()
    
    return reward


def leg_lift_exploration(
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Exploration bonus for leg lift (terrain-agnostic).

    Rewards any passive-joint lift to encourage early exploration.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos
    
    # Passive joints control lift height
    leg_joint_names = ['DriveLeftPassive', 'DriveRighttPassive']
    leg_indices = []
    
    for name in leg_joint_names:
        if name in asset.joint_names:
            leg_indices.append(asset.joint_names.index(name))
    
    if len(leg_indices) == 0:
        # Warn once if joints are missing
        if not hasattr(leg_lift_exploration, '_warned'):
            print(f"WARNING: Leg joints not found! Available joints: {asset.joint_names}")
            leg_lift_exploration._warned = True
        return torch.zeros(env.num_envs, device=env.device)
    
    leg_positions = joint_pos[:, leg_indices]
    
    lift_amount = torch.abs(leg_positions).max(dim=-1)[0]
    
    reward = lift_amount * 2.0  # scale signal
    
    return reward


def leg_action_magnitude(
        env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """Reward magnitude of leg actions (exploration).

    Assumes first four action dims are legs: left motor/passive, right motor/passive.
    """
    actions = env.action_manager.action
    
    # [left_leg(2), right_leg(2), right_wheel, left_wheel] — first four are legs
    leg_actions = actions[:, :4]
    
    action_magnitude = torch.abs(leg_actions).mean(dim=-1)
    
    return action_magnitude


def wheel_velocity_tracking(
        env: ManagerBasedRLEnv,
        sensor_cfg: SceneEntityCfg,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        command_name: str = "base_velocity",
) -> torch.Tensor:
    """On flat ground, reward matching forward wheel/body speed to command."""
    _, _, is_flat, _ = _get_terrain_info(env, sensor_cfg)
    
    command = env.command_manager.get_command(command_name)
    cmd_vel_x = command[:, 0]
    
    robot = env.scene["robot"]
    actual_vel_x = robot.data.root_lin_vel_b[:, 0]
    
    vel_error = torch.abs(cmd_vel_x - actual_vel_x)
    tracking_reward = torch.exp(-vel_error / 0.5)
    
    # Extra bonus only on flat terrain
    reward = tracking_reward * is_flat.float()
    
    return reward


# ============== Penalties to favor legs on obstacles ==============

def obstacle_wheel_penalty(
        env: ManagerBasedRLEnv,
        sensor_cfg: SceneEntityCfg,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize high wheel speed on obstacles (discourage pushing with wheels)."""
    _, front_height_diff, _, obstacle_ahead = _get_terrain_info(env, sensor_cfg)
    
    obstacle_strength = torch.clamp(front_height_diff / 0.04, min=0.0, max=1.0)
    
    asset: Articulation = env.scene[asset_cfg.name]
    joint_vel = asset.data.joint_vel
    
    wheel_joint_names = ['MotorRight', 'MotorLeft']
    wheel_indices = []
    for name in wheel_joint_names:
        if name in asset.joint_names:
            wheel_indices.append(asset.joint_names.index(name))
    
    if len(wheel_indices) > 0:
        wheel_vel = joint_vel[:, wheel_indices]
        wheel_speed = torch.abs(wheel_vel).mean(dim=-1)
    else:
        wheel_speed = torch.zeros(env.num_envs, device=env.device)
    
    penalty = wheel_speed * obstacle_strength
    
    return penalty


def leg_stepping_reward(
        env: ManagerBasedRLEnv,
        sensor_cfg: SceneEntityCfg,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        command_name: str = "base_velocity",
) -> torch.Tensor:
    """Reward alternating leg motion on obstacles only."""
    _, front_height_diff, is_flat, obstacle_ahead = _get_terrain_info(env, sensor_cfg)
    
    obstacle_strength = torch.clamp(front_height_diff / 0.04, min=0.0, max=1.0)
    
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos
    joint_vel = asset.data.joint_vel
    
    # Left/right passive joints (lift)
    left_leg_idx = None
    right_leg_idx = None
    
    if 'DriveLeftPassive' in asset.joint_names:
        left_leg_idx = asset.joint_names.index('DriveLeftPassive')
    if 'DriveRighttPassive' in asset.joint_names:
        right_leg_idx = asset.joint_names.index('DriveRighttPassive')
    
    if left_leg_idx is None or right_leg_idx is None:
        return torch.zeros(env.num_envs, device=env.device)
    
    left_pos = joint_pos[:, left_leg_idx]
    right_pos = joint_pos[:, right_leg_idx]
    left_vel = joint_vel[:, left_leg_idx]
    right_vel = joint_vel[:, right_leg_idx]
    
    alternating = -left_vel * right_vel  # positive when velocities oppose
    alternating_reward = torch.clamp(alternating, min=0.0)
    
    leg_speed = (torch.abs(left_vel) + torch.abs(right_vel)) / 2.0
    motion_reward = torch.clamp(leg_speed, min=0.0, max=2.0)
    
    reward = (alternating_reward * 0.3 + motion_reward * 0.7) * obstacle_strength
    
    # Only under motion command
    command = env.command_manager.get_command(command_name)
    moving = torch.norm(command[:, :2], dim=1) > 0.1
    reward *= moving.float()
    
    return reward


def foot_contact_reward(
        env: ManagerBasedRLEnv,
        sensor_cfg: SceneEntityCfg,
        foot_names: list = None,
) -> torch.Tensor:
    """Simple foot/wheel contact magnitude (tune foot_names for your asset)."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    
    if foot_names is None:
        foot_names = ['wheel_01', 'wheel']
    
    net_forces = contact_sensor.data.net_forces_w_history
    
    contact_force = torch.norm(net_forces[:, 0, :], dim=-1)
    has_contact = (contact_force > 0.1).float()
    
    return has_contact


def direct_leg_velocity_reward(
        env: ManagerBasedRLEnv,
        sensor_cfg: SceneEntityCfg,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward leg joint speed on obstacles only."""
    _, front_height_diff, is_flat, obstacle_ahead = _get_terrain_info(env, sensor_cfg)
    
    obstacle_strength = torch.clamp(front_height_diff / 0.04, min=0.0, max=1.0)
    
    asset: Articulation = env.scene[asset_cfg.name]
    joint_vel = asset.data.joint_vel
    
    # All leg joints
    leg_joint_names = ['DriveLeftMotor', 'DriveLeftPassive', 'DriveRightMotor', 'DriveRighttPassive']
    leg_indices = []
    for name in leg_joint_names:
        if name in asset.joint_names:
            leg_indices.append(asset.joint_names.index(name))
    
    if len(leg_indices) == 0:
        return torch.zeros(env.num_envs, device=env.device)
    
    leg_vel = joint_vel[:, leg_indices]
    leg_speed = torch.abs(leg_vel).mean(dim=-1)
    
    reward = leg_speed * obstacle_strength
    
    return reward


# ============== Feet air time: wheel–leg coordination ==============

def feet_air_time_wheel_penalty(
        env: ManagerBasedRLEnv,
        contact_sensor_cfg: SceneEntityCfg,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        air_time_threshold: float = 0.05,  # ~50 ms airborne counts as stepping
) -> torch.Tensor:
    """Penalize wheel speed while feet are airborne (discourage wheel shove during steps)."""
    contact_sensor: ContactSensor = env.scene.sensors[contact_sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]
    
    air_time = contact_sensor.data.current_air_time  # (num_envs, num_bodies)
    
    mean_air_time = air_time.mean(dim=-1)
    
    stepping_intensity = torch.clamp(mean_air_time / 0.3, min=0.0, max=1.0)
    
    joint_vel = asset.data.joint_vel
    wheel_joint_names = ['MotorRight', 'MotorLeft']
    wheel_indices = []
    for name in wheel_joint_names:
        if name in asset.joint_names:
            wheel_indices.append(asset.joint_names.index(name))
    
    if len(wheel_indices) > 0:
        wheel_vel = joint_vel[:, wheel_indices]
        wheel_speed = torch.abs(wheel_vel).mean(dim=-1)
    else:
        wheel_speed = torch.zeros(env.num_envs, device=env.device)
    
    penalty = stepping_intensity * wheel_speed
    
    return penalty


def feet_air_time_leg_reward(
        env: ManagerBasedRLEnv,
        contact_sensor_cfg: SceneEntityCfg,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        target_air_time: float = 0.2,
) -> torch.Tensor:
    """Reward foot air time near a target (real stepping vs. dragging)."""
    contact_sensor: ContactSensor = env.scene.sensors[contact_sensor_cfg.name]
    
    air_time = contact_sensor.data.current_air_time  # (num_envs, num_bodies)
    
    max_air_time = air_time.max(dim=-1)[0]
    
    reward = torch.exp(-torch.abs(max_air_time - target_air_time) / 0.1)
    
    has_air_time = max_air_time > 0.02
    reward *= has_air_time.float()
    
    return reward


def alternating_gait_reward(
        env: ManagerBasedRLEnv,
        contact_sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward alternating gait: one foot airborne while the other supports."""
    contact_sensor: ContactSensor = env.scene.sensors[contact_sensor_cfg.name]
    
    air_time = contact_sensor.data.current_air_time  # (num_envs, num_bodies)
    
    if air_time.shape[-1] >= 2:
        left_air = air_time[:, 0]
        right_air = air_time[:, 1]
        
        left_in_air = left_air > 0.02
        right_in_air = right_air > 0.02
        
        alternating = (left_in_air ^ right_in_air).float()
        
        reward = alternating
    else:
        reward = torch.zeros(env.num_envs, device=env.device)
    
    return reward


# ============== Legacy alias ==============

def leg_lift_on_obstacle(
        env: ManagerBasedRLEnv,
        sensor_cfg: SceneEntityCfg,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        min_lift: float = 0.05,
        obstacle_threshold: float = 0.03,
        command_name: str = "base_velocity",
) -> torch.Tensor:
    """Legacy wrapper: leg lift on ramps/steps (delegates to obstacle_leg_lift)."""
    return obstacle_leg_lift(
        env, sensor_cfg, asset_cfg, min_lift, command_name
    )
