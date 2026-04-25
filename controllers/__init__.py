# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

"""NexusTrex controller package."""

from .lqr_controller import (
    LQRController,
    BalanceLQRController,
    lqr_gain_discrete,
    lqr_gain_continuous,
    design_balance_lqr,
    balance_lip_matrices,
    solve_dare,
    lqr_residual_passive_balance,
    policy_obs_to_tensor,
)

__all__ = [
    "LQRController",
    "BalanceLQRController",
    "lqr_gain_discrete",
    "lqr_gain_continuous",
    "design_balance_lqr",
    "balance_lip_matrices",
    "solve_dare",
    "lqr_residual_passive_balance",
    "policy_obs_to_tensor",
]
