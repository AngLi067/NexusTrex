# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
NexusTrex wheeled-leg task registrations.
- basic: terrain curriculum task
- flat: flat-ground task
- haptic: terrain-aware / mode-switching task
"""

import gymnasium as gym
from . import agents
from .agents import NexusTrexBasicPPORunnerCfg

##
# Register Gym environments.
##

# Basic terrain task
gym.register(
    id="nexustrex-basic-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.nexustrex_basic_task:NexusTrexBasicEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:NexusTrexBasicPPORunnerCfg",
    },
)

# Flat terrain task
gym.register(
    id="nexustrex-basic-flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.nexustrex_basic_flat_task:NexusTrexBasicEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:NexusTrexBasicPPORunnerCfg",
    },
)

# Haptic / terrain-perception task
gym.register(
    id="nexustrex-haptic-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.nexustrex_haptic_task:NexusTrexHapticEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:NexusTrexBasicPPORunnerCfg",
    },
)

# Distillation: MDP teacher -> POMDP student (no height scanner)
gym.register(
    id="nexustrex-distillation-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.nexustrex_distillation_task:NexusTrexDistillationEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:NexusTrexBasicPPORunnerCfg",
        "rsl_rl_distillation_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_distillation_cfg:NexusTrexDistillationRunnerCfg"
        ),
    },
)

# Distillation: MDP teacher -> student with height scanner
gym.register(
    id="nexustrex-distillation-with-scanner-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.nexustrex_distillation_with_scanner_task:NexusTrexDistillationWithScannerEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:NexusTrexBasicPPORunnerCfg",
        "rsl_rl_distillation_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_distillation_with_scanner_cfg:NexusTrexDistillationWithScannerRunnerCfg"
        ),
    },
)

# Controlled benchmark tasks (fixed terrain level + command); ids cv-b / cv-d / cv-s
gym.register(
    id="nexustrex-cv-b-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.nexustrex_bench_task:NexusTrexCvBasicEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:NexusTrexBasicPPORunnerCfg",
    },
)
gym.register(
    id="nexustrex-cv-d-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.nexustrex_bench_task:NexusTrexCvDistillEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:NexusTrexBasicPPORunnerCfg",
        "rsl_rl_distillation_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_distillation_cfg:NexusTrexDistillationRunnerCfg"
        ),
    },
)
gym.register(
    id="nexustrex-cv-s-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.nexustrex_bench_task:NexusTrexCvDistillScanEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:NexusTrexBasicPPORunnerCfg",
        "rsl_rl_distillation_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_distillation_with_scanner_cfg:NexusTrexDistillationWithScannerRunnerCfg"
        ),
    },
)

