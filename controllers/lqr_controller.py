# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

"""
LQR (Linear Quadratic Regulator) utilities.

Includes:
1. Discrete-time LQR -- general state feedback
2. Balance LQR -- linearized inverted-pendulum model (pitch/roll)
3. Continuous-time LQR
"""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from scipy import linalg
from typing import Optional


def policy_obs_to_tensor(obs_policy) -> torch.Tensor:
    """
    Unpack RSL-RL / TensorDict policy observations to a (batch, dim) float tensor.

    ``obs["policy"]`` may be a nested TensorDict; it cannot be sliced with ``[:, 6:9]`` directly.
    """
    x = obs_policy
    while not isinstance(x, torch.Tensor):
        if hasattr(x, "keys"):
            ks = list(x.keys())
            if len(ks) == 0:
                raise TypeError(f"Empty observation container: {type(obs_policy)}")
            if len(ks) == 1:
                x = x[ks[0]]
            else:
                x = torch.cat([x[k] for k in sorted(ks)], dim=-1)
        else:
            raise TypeError(f"Cannot unwrap observation to tensor: {type(obs_policy)}")
    return x


def solve_dare(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray) -> np.ndarray:
    """
    Discrete-time algebraic Riccati equation (DARE):
    P = A'PA - A'PB(R + B'PB)^{-1}B'PA + Q

    Returns:
        P: optimal cost matrix
    """
    P = linalg.solve_discrete_are(A, B, Q, R)
    return P


def lqr_gain_discrete(
    A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Discrete-time LQR: compute optimal gain K, u = -K @ x.

    Args:
        A: state transition (n x n)
        B: control input (n x m)
        Q: state cost (n x n), PSD
        R: control cost (m x m), PD

    Returns:
        K: feedback gain (m x n)
        P: Riccati solution (n x n)
    """
    P = solve_dare(A, B, Q, R)
    R_inv = np.linalg.inv(R + B.T @ P @ B)
    K = R_inv @ B.T @ P @ A
    return K, P


def lqr_gain_continuous(
    A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Continuous-time LQR via continuous algebraic Riccati equation (CARE):
    A'P + PA - PBR^{-1}B'P + Q = 0

    Returns:
        K: optimal feedback gain (m x n)
        P: Riccati solution (n x n)
    """
    P = linalg.solve_continuous_are(A, B, Q, R)
    R_inv = np.linalg.inv(R)
    K = R_inv @ B.T @ P
    return K, P


# ============== Linear inverted pendulum (LIP) ==============


def balance_lip_matrices(
    mass: float = 1.0,
    height: float = 0.45,
    g: float = 9.81,
    dt: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Linearized inverted-pendulum dynamics (pitch/roll).
    State: x = [theta, theta_dot]
    Control: u = [tau]

    Simplified: I * theta_ddot = tau - m*g*h*sin(theta) ~ tau - m*g*h*theta

    Args:
        mass: mass (kg)
        height: COM height (m)
        g: gravity
        dt: discretization step

    Returns:
        A_d, B_d: discrete-time (A, B)
    """
    # Point-mass moment of inertia approximation
    I = mass * height**2
    k = mass * g * height / I  # open-loop pole scale sqrt(k)

    Ac = np.array([[0.0, 1.0], [-k, 0.0]])
    Bc = np.array([[0.0], [1.0 / I]])

    n = Ac.shape[0]
    m = Bc.shape[1]
    M = np.zeros((n + m, n + m))
    M[:n, :n] = Ac
    M[:n, n:] = Bc
    exp_M = linalg.expm(M * dt)
    A_d = exp_M[:n, :n]
    B_d = exp_M[:n, n:]

    return A_d, B_d


def design_balance_lqr(
    mass: float = 1.0,
    height: float = 0.45,
    dt: float = 0.01,
    Q_diag: Optional[tuple[float, float]] = None,
    R_val: float = 1.0,
) -> np.ndarray:
    """
    Design balance LQR gain for pitch/roll.

    Args:
        mass, height, dt: physical / timing parameters
        Q_diag: state weights [theta, theta_dot], default [100, 10]
        R_val: control weight

    Returns:
        K: gain (1x2), u = -K @ [theta, theta_dot]
    """
    A_d, B_d = balance_lip_matrices(mass, height, 9.81, dt)
    Q = np.diag(Q_diag or [100.0, 10.0])
    R = np.array([[R_val]])
    K, _ = lqr_gain_discrete(A_d, B_d, Q, R)
    return K


# ============== PyTorch LQR modules ==============


class LQRController(nn.Module):
    """
    LQR state feedback (PyTorch, batched): u = -K @ (x - x_ref).
    """

    def __init__(self, K: np.ndarray, device: str = "cuda"):
        super().__init__()
        self.register_buffer("K", torch.from_numpy(K).float())
        self.device = device

    def forward(
        self,
        x: torch.Tensor,
        x_ref: Optional[torch.Tensor] = None,
        u_ff: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: state (batch, n) or (n,)
            x_ref: reference state (batch, n), default zero
            u_ff: feedforward (batch, m), optional

        Returns:
            u: control (batch, m)
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x_ref is None:
            x_ref = torch.zeros_like(x, device=x.device)
        elif x_ref.dim() == 1:
            x_ref = x_ref.unsqueeze(0)

        x_err = x - x_ref
        K = self.K.to(x.device)
        u = -x_err @ K.T

        if u_ff is not None:
            u = u + u_ff
        return u.squeeze(0) if x.dim() == 1 else u


class BalanceLQRController(nn.Module):
    """
    Balance LQR for pitch/roll.
    Input: [pitch, pitch_dot] or [roll, roll_dot]
    Output: torque / position correction scalar per channel
    """

    def __init__(
        self,
        mass: float = 1.0,
        height: float = 0.45,
        dt: float = 0.01,
        Q_angle: float = 100.0,
        Q_velocity: float = 10.0,
        R: float = 1.0,
        device: str = "cuda",
    ):
        super().__init__()
        K = design_balance_lqr(
            mass=mass,
            height=height,
            dt=dt,
            Q_diag=(Q_angle, Q_velocity),
            R_val=R,
        )
        self.lqr = LQRController(K, device)

    def forward(
        self,
        angle: torch.Tensor,
        angle_dot: torch.Tensor,
        angle_ref: Optional[torch.Tensor] = None,
        angle_dot_ref: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            angle, angle_dot: angle and rate (batch,) or (batch, 1)
            angle_ref, angle_dot_ref: references, default 0

        Returns:
            u: control (batch,)
        """
        x = torch.stack([angle.flatten(), angle_dot.flatten()], dim=-1)
        if angle_ref is None:
            x_ref = torch.zeros_like(x, device=x.device)
        else:
            x_ref = torch.stack(
                [angle_ref.flatten(), (angle_dot_ref or torch.zeros_like(angle_ref)).flatten()],
                dim=-1,
            )
        return self.lqr(x, x_ref).squeeze(-1)


def lqr_residual_passive_balance(
    obs_policy: torch.Tensor,
    lqr: BalanceLQRController,
    *,
    gain: float = 0.05,
    sign: float = 1.0,
    smooth_alpha: float = 0.88,
    action_clip: float = 0.12,
    prev_left: torch.Tensor | None = None,
    prev_right: torch.Tensor | None = None,
    # Adaptive LQR (runtime): leaky integrator + gain scheduling
    int_pitch: torch.Tensor | None = None,
    int_roll: torch.Tensor | None = None,
    adaptive_gain_schedule: float = 0.0,
    adaptive_ki: float = 0.0,
    integrator_leak: float = 0.999,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """
    Compute left/right passive-joint LQR residuals from policy observations
    (prefix aligned with NexusTrex basic): base_lin_vel(3), base_ang_vel(3), projected_gravity(3), ...

    Optional adaptive terms:
    - Gain scheduling if ``adaptive_gain_schedule > 0``: scale gain with |pitch|+|roll|.
    - Leaky integrator if ``adaptive_ki > 0``: bias / slow-error rejection.

    Returns:
        comp_left, comp_right: add to action[:,1] and action[:,3]
        int_pitch, int_roll: updated integrator states (or None if disabled)
    """
    obs_t = policy_obs_to_tensor(obs_policy)
    base_orient = obs_t[:, 6:9]
    base_ang_vel = obs_t[:, 3:6]
    gx, gy, gz = base_orient[:, 0], base_orient[:, 1], base_orient[:, 2]
    pitch = torch.atan2(-gx, torch.sqrt(gy**2 + gz**2 + 1e-8))
    roll = torch.atan2(gy, torch.sqrt(gx**2 + gz**2 + 1e-8))
    pitch_dot = base_ang_vel[:, 1]
    roll_dot = base_ang_vel[:, 0]

    if adaptive_ki > 0.0:
        if int_pitch is None:
            int_pitch = torch.zeros_like(pitch)
            int_roll = torch.zeros_like(roll)
        u_pitch = lqr(pitch, pitch_dot) * sign + adaptive_ki * int_pitch
        u_roll = lqr(roll, roll_dot) * sign + adaptive_ki * int_roll
        int_pitch = integrator_leak * int_pitch + (1.0 - integrator_leak) * pitch
        int_roll = integrator_leak * int_roll + (1.0 - integrator_leak) * roll
    else:
        u_pitch = lqr(pitch, pitch_dot) * sign
        u_roll = lqr(roll, roll_dot) * sign
        int_pitch, int_roll = None, None

    eff_gain: float | torch.Tensor = gain
    if adaptive_gain_schedule > 0.0:
        att = torch.abs(pitch) + torch.abs(roll)
        eff_gain = gain * (1.0 + adaptive_gain_schedule * torch.tanh(att * 2.5))

    comp_left_raw = (u_pitch - u_roll) * eff_gain
    comp_right_raw = (u_pitch + u_roll) * eff_gain

    comp_left_raw = comp_left_raw.clamp(-action_clip, action_clip)
    comp_right_raw = comp_right_raw.clamp(-action_clip, action_clip)

    if prev_left is None:
        comp_left = comp_left_raw.clone()
        comp_right = comp_right_raw.clone()
    else:
        comp_left = smooth_alpha * prev_left + (1 - smooth_alpha) * comp_left_raw
        comp_right = smooth_alpha * prev_right + (1 - smooth_alpha) * comp_right_raw

    return comp_left, comp_right, int_pitch, int_roll


if __name__ == "__main__":
    A = np.array([[1.0, 0.01], [0.0, 1.0]])
    B = np.array([[0.0], [0.01]])
    Q = np.diag([10.0, 1.0])
    R = np.array([[1.0]])
    K, P = lqr_gain_discrete(A, B, Q, R)
    print("LQR Gain K:", K)

    K_bal = design_balance_lqr(mass=10.0, height=0.45, dt=0.01)
    print("Balance LQR Gain:", K_bal)
