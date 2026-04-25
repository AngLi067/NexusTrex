# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

"""Teacher–student distillation agent configuration.
- policy (Student/POMDP): uses student_policy observations
- teacher (MDP): uses teacher_policy observations
"""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlDistillationAlgorithmCfg,
    RslRlDistillationRunnerCfg,
    RslRlDistillationStudentTeacherCfg,
)


@configclass
class NexusTrexDistillationRunnerCfg(RslRlDistillationRunnerCfg):
    """NexusTrex distillation runner configuration."""

    load_experiment: str = "nexus-trex-basic-v0"
    

    num_steps_per_env = 24
    max_iterations = 8000
    save_interval = 50
    experiment_name = "nexus-trex-distillation"

    # Student uses student_policy; teacher uses teacher_policy
    obs_groups = {"policy": ["student_policy"], "teacher": ["teacher_policy"]}

    policy = RslRlDistillationStudentTeacherCfg(
        init_noise_std=0.3,
        noise_std_type="scalar",
        student_obs_normalization=False,
        teacher_obs_normalization=False,
        student_hidden_dims=[512, 256, 128],
        teacher_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=6,
        learning_rate=1.0e-3,
        gradient_length=24,
        loss_type="mse",
    )
