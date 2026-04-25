# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause
"""Bench helper: pin all envs to one terrain curriculum level."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def set_fixed_terrain_level(env: ManagerBasedRLEnv, env_ids: Sequence[int] | None, level: int) -> None:
    """Set terrain level to ``level`` for listed envs (ignores curriculum)."""
    terrain = env.scene.terrain
    if terrain.terrain_origins is None:
        return
    device = terrain.env_origins.device
    n_rows = terrain.terrain_origins.shape[0]
    lv = int(max(0, min(int(level), n_rows - 1)))
    if env_ids is None:
        ids = torch.arange(env.num_envs, device=device, dtype=torch.long)
    else:
        ids = torch.as_tensor(env_ids, device=device, dtype=torch.long)
    terrain.terrain_levels[ids] = lv
    terrain.env_origins[ids] = terrain.terrain_origins[terrain.terrain_levels[ids], terrain.terrain_types[ids]]
