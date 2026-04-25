# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause
#
# Controlled benchmark: fixed terrain level + fixed velocity command; no curriculum / cmd-resample terminations.
# Tune BENCH_* below; task ids: nexustrex-cv-b-v0 / cv-d / cv-s

from __future__ import annotations

import math

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

from .mdp.bench_mdp import set_fixed_terrain_level
from .nexustrex_basic_task import EventCfg as BasicEventCfg, NexusTrexBasicEnvCfg
from .nexustrex_distillation_task import EventCfg as DistillEventCfg, NexusTrexDistillationEnvCfg
from .nexustrex_distillation_with_scanner_task import (
    EventCfg as ScanEventCfg,
    NexusTrexDistillationWithScannerEnvCfg,
)

# ========= Default bench parameters (edit as needed) =========
BENCH_TERRAIN_LEVEL = 0          # 0 = flat (trajectory benchmark); 2 = hard terrain
BENCH_VX = 0.5
BENCH_WZ = 0.0
BENCH_EPISODE_S = 120.0          # long horizon for slow reference tracking
# Velocity command nearly constant over the episode (huge resampling interval)
BENCH_RESAMPLE_TIME = (1.0e9, 1.0e9)


def _bench_velocity_command():
    return mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=BENCH_RESAMPLE_TIME,
        rel_standing_envs=0.0,
        rel_heading_envs=0.0,
        heading_command=False,
        heading_control_stiffness=0.5,
        debug_vis=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(BENCH_VX, BENCH_VX),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(BENCH_WZ, BENCH_WZ),
            heading=(-math.pi, math.pi),
        ),
    )


@configclass
class BenchTerminationsCfg:
    """No command_resample termination (fixed commands should not early-stop)."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": 1.57},
    )


@configclass
class BenchCommandsBasicCfg:
    base_velocity = _bench_velocity_command()


@configclass
class BenchEventBasicCfg(BasicEventCfg):
    fix_terrain_startup = EventTerm(
        func=set_fixed_terrain_level,
        mode="startup",
        params={"level": BENCH_TERRAIN_LEVEL},
    )
    fix_terrain_reset = EventTerm(
        func=set_fixed_terrain_level,
        mode="reset",
        params={"level": BENCH_TERRAIN_LEVEL},
    )


@configclass
class BenchEventDistillCfg(DistillEventCfg):
    fix_terrain_startup = EventTerm(
        func=set_fixed_terrain_level,
        mode="startup",
        params={"level": BENCH_TERRAIN_LEVEL},
    )
    fix_terrain_reset = EventTerm(
        func=set_fixed_terrain_level,
        mode="reset",
        params={"level": BENCH_TERRAIN_LEVEL},
    )


@configclass
class BenchEventScanCfg(ScanEventCfg):
    fix_terrain_startup = EventTerm(
        func=set_fixed_terrain_level,
        mode="startup",
        params={"level": BENCH_TERRAIN_LEVEL},
    )
    fix_terrain_reset = EventTerm(
        func=set_fixed_terrain_level,
        mode="reset",
        params={"level": BENCH_TERRAIN_LEVEL},
    )


@configclass
class NexusTrexCvBasicEnvCfg(NexusTrexBasicEnvCfg):
    """Teacher/PPO observations; fixed terrain + fixed command."""

    commands: BenchCommandsBasicCfg = BenchCommandsBasicCfg()
    terminations: BenchTerminationsCfg = BenchTerminationsCfg()
    events: BenchEventBasicCfg = BenchEventBasicCfg()
    curriculum: object | None = None

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = BENCH_EPISODE_S


@configclass
class BenchCommandsDistillCfg:
    base_velocity = _bench_velocity_command()


@configclass
class NexusTrexCvDistillEnvCfg(NexusTrexDistillationEnvCfg):
    commands: BenchCommandsDistillCfg = BenchCommandsDistillCfg()
    terminations: BenchTerminationsCfg = BenchTerminationsCfg()
    events: BenchEventDistillCfg = BenchEventDistillCfg()
    curriculum: object | None = None

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = BENCH_EPISODE_S


@configclass
class BenchCommandsScanCfg:
    base_velocity = _bench_velocity_command()


@configclass
class NexusTrexCvDistillScanEnvCfg(NexusTrexDistillationWithScannerEnvCfg):
    commands: BenchCommandsScanCfg = BenchCommandsScanCfg()
    terminations: BenchTerminationsCfg = BenchTerminationsCfg()
    events: BenchEventScanCfg = BenchEventScanCfg()
    curriculum: object | None = None

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = BENCH_EPISODE_S
