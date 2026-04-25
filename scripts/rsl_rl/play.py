# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent",
    type=str,
    default="rsl_rl_cfg_entry_point",
    help="Agent config entry point (use rsl_rl_distillation_cfg_entry_point for distilled student).",
)
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument(
    "--use_lqr",
    action="store_true",
    help="Add LQR residual on passive joints; policy leads, LQR fine-tunes.",
)
parser.add_argument("--lqr_gain", type=float, default=0.05, help="LQR residual gain (keep small vs policy)")
parser.add_argument("--lqr_smooth", type=float, default=0.88, help="Exponential smoothing on LQR output (0-1)")
parser.add_argument("--lqr_flip_sign", action="store_true", help="Flip residual sign if direction is wrong")
parser.add_argument(
    "--lqr_adaptive",
    action="store_true",
    help="Adaptive LQR: leaky integrator + gain schedule on |pitch|+|roll| (sim2real helper)",
)
parser.add_argument(
    "--lqr_adaptive_ki",
    type=float,
    default=0.08,
    help="Leaky integrator gain (only with --lqr_adaptive); 0 disables bias integral",
)
parser.add_argument(
    "--lqr_adaptive_schedule",
    type=float,
    default=0.25,
    help="Gain schedule strength vs |angle|; 0 disables (only with --lqr_adaptive)",
)
parser.add_argument("--lqr_int_leak", type=float, default=0.998, help="Integrator leak factor (anti-windup)")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--log_tracking", action="store_true",
    help="Log env#0 tracking to TXT (vel/attitude/action/LQR); visualize with plot_tracking.py",
)
parser.add_argument("--log_steps", type=int, default=1000, help="Stop logging after this many steps (--log_tracking)")
parser.add_argument(
    "--log_out",
    type=str,
    default=None,
    help="Full path for tracking TXT; default: tracking_logs/ next to checkpoint",
)
parser.add_argument("--circle", action="store_true", help="Closed-loop reference tracking (instead of random cmd)")
parser.add_argument("--traj", choices=["circle", "eight", "lissa"], default="eight",
                    help="Reference shape: circle; eight (1:2 Lissajous); lissa (2:3)")
parser.add_argument("--circle_r", type=float, default=1.5, help="Trajectory scale (m): radius / amplitude")
parser.add_argument("--circle_omega", type=float, default=0.10, help="Base frequency rad/s (~period scale)")
parser.add_argument("--circle_kp", type=float, default=0.5, help="Forward position P gain (feedback)")
parser.add_argument("--circle_kp_yaw", type=float, default=1.2, help="Heading P gain (feedback)")
parser.add_argument(
    "--traj_kp_lat",
    type=float,
    default=0.2,
    help="Small gain: lateral error -> yaw (body left positive), improves path following",
)
parser.add_argument(
    "--traj_smooth",
    type=float,
    default=0.12,
    help="Outer-loop velocity command low-pass alpha: cmd <- alpha*raw + (1-alpha)*cmd",
)
parser.add_argument(
    "--traj_no_ff",
    action="store_true",
    help="Disable tangential/yaw feedforward (P feedback only, for ablations)",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import datetime
import numpy as np
import torch

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

# PLACEHOLDER: Extension template (do not remove this comment)
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import tasks
from controllers import BalanceLQRController, lqr_residual_passive_balance, policy_obs_to_tensor


def _policy_obs_tensor(obs):
    """Unpack nested TensorDict policy observations to a flat tensor."""
    try:
        if "policy" in obs:
            return policy_obs_to_tensor(obs["policy"])
    except (TypeError, KeyError):
        pass
    return policy_obs_to_tensor(obs)


def main():
    """Play with RSL-RL agent."""
    task_name = args_cli.task.split(":")[-1]
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    agent_cfg: RslRlBaseRunnerCfg = load_cfg_from_registry(task_name, args_cli.agent)
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)

    # specify directory for logging experiments
    # Use the same path as train.py: logs/rsl_rl/experiment_name
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    print(f"[INFO] Loading checkpoint: {resume_path}")

    # obtain the trained policy for inference (DistillationRunner -> student)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    # export policy to onnx/jit (distillation exports student)
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    normalizer = (
        getattr(runner, "obs_normalizer", None)
        or getattr(runner, "normalizer", None)
        or (getattr(policy_nn, "student_obs_normalizer", None) if hasattr(policy_nn, "student_obs_normalizer") else None)
    )

    export_policy_as_jit(policy_nn, normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt
    decimation = getattr(env.unwrapped.cfg, "decimation", 1)
    lqr_dt = dt * decimation
    lqr_module = None
    lqr_prev_left = None
    lqr_prev_right = None
    lqr_int_pitch = None
    lqr_int_roll = None
    adaptive_ki = args_cli.lqr_adaptive_ki if args_cli.lqr_adaptive else 0.0
    adaptive_sched = args_cli.lqr_adaptive_schedule if args_cli.lqr_adaptive else 0.0
    if args_cli.use_lqr:
        lqr_module = BalanceLQRController(
            mass=10.0,
            height=0.45,
            dt=lqr_dt,
            Q_angle=80.0,
            Q_velocity=8.0,
            R=1.2,
            device=env.unwrapped.device,
        )
        msg = (
            "[INFO] LQR residual: action = policy(obs) + passive joints (1,3); "
            "tune --lqr_gain / --lqr_smooth / --lqr_flip_sign"
        )
        if args_cli.lqr_adaptive:
            msg += (
                f"; Adaptive LQR (ki={adaptive_ki}, schedule={adaptive_sched}, leak={args_cli.lqr_int_leak})"
            )
        print(msg)

    # ---- short auto filename map ----
    _SHORT = {
        "nexustrex-basic-v0": "b", "nexustrex-basic-flat-v0": "bf",
        "nexustrex-distillation-v0": "d", "nexustrex-distillation-with-scanner-v0": "s",
        "nexustrex-cv-b-v0": "cb", "nexustrex-cv-d-v0": "cd", "nexustrex-cv-s-v0": "cs",
    }

    # ---- circle tracking state ----
    circle_phase = None
    traj_lp_vx = None
    traj_lp_wz = None
    if args_cli.circle:
        _dev = env.unwrapped.device
        _n = env.unwrapped.num_envs
        circle_phase = torch.zeros(_n, device=_dev)
        traj_lp_vx = torch.zeros(_n, device=_dev)
        traj_lp_wz = torch.zeros(_n, device=_dev)
        _ff = "off" if args_cli.traj_no_ff else "on"
        print(
            f"[INFO] Closed-loop tracking: shape={args_cli.traj}  R={args_cli.circle_r} "
            f"omega={args_cli.circle_omega}  Kp_fwd={args_cli.circle_kp} Kp_yaw={args_cli.circle_kp_yaw} "
            f"Kp_lat={args_cli.traj_kp_lat}  smooth={args_cli.traj_smooth}  feedforward={_ff}"
        )

    # ---- tracking logger setup ----
    tracking_log = []
    log_file_path = None
    if args_cli.log_tracking:
        if args_cli.log_out:
            log_file_path = os.path.abspath(args_cli.log_out)
            _d = os.path.dirname(log_file_path)
            if _d:
                os.makedirs(_d, exist_ok=True)
        else:
            short = _SHORT.get(task_name, task_name.replace("nexustrex-", "").replace("-v0", ""))
            ts = datetime.datetime.now().strftime("%H%M")
            log_dir_track = os.path.join(os.path.dirname(resume_path), "tracking_logs")
            os.makedirs(log_dir_track, exist_ok=True)
            log_file_path = os.path.join(log_dir_track, f"{short}_{ts}.txt")
        print(f"[INFO] Tracking log -> {log_file_path}  ({args_cli.log_steps} steps)")

    # reset environment
    obs = env.get_observations()
    if isinstance(obs, tuple):
        obs = obs[0]
    timestep = 0
    _ref0_x = _ref0_y = float("nan")

    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            # ---- closed-loop trajectory tracking: override velocity command ----
            if circle_phase is not None:
                raw_e = env.unwrapped
                robot_c = raw_e.scene["robot"]
                origins = raw_e.scene.env_origins
                R = args_cli.circle_r
                omega = args_cli.circle_omega
                kp = args_cli.circle_kp
                kp_yaw = args_cli.circle_kp_yaw
                ph = circle_phase

                # --- reference position & tangent by trajectory shape ---
                shape = args_cli.traj
                o2 = omega * omega
                if shape == "circle":
                    ref_x = origins[:, 0] + R * torch.cos(ph)
                    ref_y = origins[:, 1] + R * torch.sin(ph)
                    dx = -R * omega * torch.sin(ph)
                    dy = R * omega * torch.cos(ph)
                    ddx = -R * o2 * torch.cos(ph)
                    ddy = -R * o2 * torch.sin(ph)
                elif shape == "eight":
                    ref_x = origins[:, 0] + R * torch.sin(ph)
                    ref_y = origins[:, 1] + R * torch.sin(2.0 * ph)
                    dx = R * omega * torch.cos(ph)
                    dy = R * 2.0 * omega * torch.cos(2.0 * ph)
                    ddx = -R * o2 * torch.sin(ph)
                    ddy = -R * 4.0 * o2 * torch.sin(2.0 * ph)
                else:  # "lissa"
                    ref_x = origins[:, 0] + R * torch.sin(2.0 * ph)
                    ref_y = origins[:, 1] + R * torch.sin(3.0 * ph)
                    dx = R * 2.0 * omega * torch.cos(2.0 * ph)
                    dy = R * 3.0 * omega * torch.cos(3.0 * ph)
                    ddx = -R * 4.0 * o2 * torch.sin(2.0 * ph)
                    ddy = -R * 9.0 * o2 * torch.sin(3.0 * ph)

                pos_w = robot_c.data.root_pos_w
                ex_w = ref_x - pos_w[:, 0]
                ey_w = ref_y - pos_w[:, 1]

                q = robot_c.data.root_quat_w
                yaw = torch.atan2(
                    2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
                    1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2),
                )
                cos_y, sin_y = torch.cos(yaw), torch.sin(yaw)
                e_fwd = ex_w * cos_y + ey_w * sin_y
                e_lat = -ex_w * sin_y + ey_w * cos_y

                ref_hdg = torch.atan2(dy, dx)
                hdg_err = torch.atan2(
                    torch.sin(ref_hdg - yaw), torch.cos(ref_hdg - yaw)
                )

                eps = 1.0e-5
                v2 = dx * dx + dy * dy + eps
                wz_ff = (dx * ddy - dy * ddx) / v2
                vx_ff = dx * cos_y + dy * sin_y
                if args_cli.traj_no_ff:
                    vx_ff = vx_ff * 0.0
                    wz_ff = wz_ff * 0.0
                # Near figure-eight crossing, ref speed ~0 and yaw-rate ill-conditioned: soften feedforward
                slow = v2 < 0.01
                wz_ff = torch.where(slow, torch.zeros_like(wz_ff), wz_ff)
                wz_ff = wz_ff.clamp(-0.5, 0.5)
                vx_ff = vx_ff.clamp(-0.4, 0.4)

                fb_fwd = kp * e_fwd
                fb_wz = kp_yaw * hdg_err + args_cli.traj_kp_lat * e_lat
                vx_raw = (vx_ff + fb_fwd).clamp(-0.45, 0.45)
                wz_raw = (wz_ff + fb_wz).clamp(-0.5, 0.5)

                a = float(args_cli.traj_smooth)
                a = max(0.01, min(1.0, a))
                traj_lp_vx[:] = a * vx_raw + (1.0 - a) * traj_lp_vx
                traj_lp_wz[:] = a * wz_raw + (1.0 - a) * traj_lp_wz

                cmd_buf = raw_e.command_manager.get_command("base_velocity")
                cmd_buf[:, 0] = traj_lp_vx
                cmd_buf[:, 1] = 0.0
                cmd_buf[:, 2] = traj_lp_wz

                _ref0_x = (ref_x[0] - origins[0, 0]).item()
                _ref0_y = (ref_y[0] - origins[0, 1]).item()

            # agent stepping
            actions = policy(obs)
            if lqr_module is not None:
                po = _policy_obs_tensor(obs)
                sign = -1.0 if args_cli.lqr_flip_sign else 1.0
                comp_l, comp_r, lqr_int_pitch, lqr_int_roll = lqr_residual_passive_balance(
                    po,
                    lqr_module,
                    gain=args_cli.lqr_gain,
                    sign=sign,
                    smooth_alpha=args_cli.lqr_smooth,
                    action_clip=0.12,
                    prev_left=lqr_prev_left,
                    prev_right=lqr_prev_right,
                    int_pitch=lqr_int_pitch,
                    int_roll=lqr_int_roll,
                    adaptive_gain_schedule=adaptive_sched,
                    adaptive_ki=adaptive_ki,
                    integrator_leak=args_cli.lqr_int_leak,
                )
                lqr_prev_left, lqr_prev_right = comp_l.clone(), comp_r.clone()
                actions = actions.clone()
                actions[:, 1] = actions[:, 1] + comp_l
                actions[:, 3] = actions[:, 3] + comp_r
            # env stepping
            obs, _, dones, _ = env.step(actions)

            # ---- circle: advance phase & handle resets ----
            if circle_phase is not None:
                circle_phase += omega * dt
                if dones is not None and dones.any():
                    dm = dones.view(-1).bool()
                    circle_phase[dm] = 0.0
                    traj_lp_vx[dm] = 0.0
                    traj_lp_wz[dm] = 0.0

            if lqr_module is not None and dones is not None and dones.any():
                dm = dones.view(-1).bool()
                lqr_prev_left[dm] = 0.0
                lqr_prev_right[dm] = 0.0
                if lqr_int_pitch is not None:
                    lqr_int_pitch[dm] = 0.0
                if lqr_int_roll is not None:
                    lqr_int_roll[dm] = 0.0

        # ---- collect tracking data for env #0 ----
        if args_cli.log_tracking and timestep < args_cli.log_steps:
            try:
                raw_env = env.unwrapped
                robot = raw_env.scene["robot"]
                origin0 = raw_env.scene.env_origins[0]
                cmd = raw_env.command_manager.get_command("base_velocity")
                cmd_vx = cmd[0, 0].item()
                cmd_vy = cmd[0, 1].item()
                cmd_wz = cmd[0, 2].item()
                act_vx = robot.data.root_lin_vel_b[0, 0].item()
                act_vy = robot.data.root_lin_vel_b[0, 1].item()
                act_vz = robot.data.root_lin_vel_b[0, 2].item()
                act_wx = robot.data.root_ang_vel_b[0, 0].item()
                act_wy = robot.data.root_ang_vel_b[0, 1].item()
                act_wz = robot.data.root_ang_vel_b[0, 2].item()
                pos_x = (robot.data.root_pos_w[0, 0] - origin0[0]).item()
                pos_y = (robot.data.root_pos_w[0, 1] - origin0[1]).item()
                pos_z = (robot.data.root_pos_w[0, 2] - origin0[2]).item()
                quat = robot.data.root_quat_w[0]
                qw, qx, qy, qz = quat[0].item(), quat[1].item(), quat[2].item(), quat[3].item()
                pitch = np.arctan2(2 * (qw * qy - qz * qx), 1 - 2 * (qx * qx + qy * qy))
                roll = np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qz * qz))
                a0 = actions[0, 0].item()
                a1 = actions[0, 1].item()
                a2 = actions[0, 2].item()
                a3 = actions[0, 3].item()
                lqr_l = lqr_prev_left[0].item() if lqr_prev_left is not None else 0.0
                lqr_r = lqr_prev_right[0].item() if lqr_prev_right is not None else 0.0
                line = (
                    f"{timestep},{timestep * dt:.4f},"
                    f"{cmd_vx:.4f},{cmd_vy:.4f},{cmd_wz:.4f},"
                    f"{act_vx:.4f},{act_vy:.4f},{act_vz:.4f},"
                    f"{act_wx:.4f},{act_wy:.4f},{act_wz:.4f},"
                    f"{pos_x:.4f},{pos_y:.4f},{pos_z:.4f},"
                    f"{pitch:.5f},{roll:.5f},"
                    f"{a0:.4f},{a1:.4f},{a2:.4f},{a3:.4f},"
                    f"{lqr_l:.5f},{lqr_r:.5f}"
                )
                if circle_phase is not None:
                    line += f",{_ref0_x:.4f},{_ref0_y:.4f}"
                tracking_log.append(line)
            except Exception as e:
                if timestep == 0:
                    print(f"[WARN] tracking log error: {e}")

        if args_cli.video:
            timestep += 1
            if timestep == args_cli.video_length:
                break
        else:
            timestep += 1

        if args_cli.log_tracking and timestep >= args_cli.log_steps:
            break

        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # ---- save tracking log ----
    if args_cli.log_tracking and tracking_log and log_file_path:
        header = (
            "step,time_s,"
            "cmd_vx,cmd_vy,cmd_wz,"
            "act_vx,act_vy,act_vz,"
            "act_wx,act_wy,act_wz,"
            "pos_x,pos_y,pos_z,"
            "pitch_rad,roll_rad,"
            "action_0,action_1,action_2,action_3,"
            "lqr_left,lqr_right"
        )
        if circle_phase is not None:
            header += ",ref_x,ref_y"
        with open(log_file_path, "w") as f:
            f.write(header + "\n")
            f.write("\n".join(tracking_log) + "\n")
        print(f"[INFO] Tracking log saved -> {log_file_path}  ({len(tracking_log)} steps)")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
