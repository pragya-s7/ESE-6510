# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Modular strategy classes for quadcopter environment rewards, observations, and resets."""

from __future__ import annotations

import torch
import numpy as np
from typing import TYPE_CHECKING, Dict, Optional, Tuple

from isaaclab.utils.math import subtract_frame_transforms, quat_from_euler_xyz, euler_xyz_from_quat, wrap_to_pi, matrix_from_quat

if TYPE_CHECKING:
    from .quadcopter_env import QuadcopterEnv

D2R = np.pi / 180.0
R2D = 180.0 / np.pi


class DefaultQuadcopterStrategy:
    """Default strategy implementation for quadcopter environment."""

    def __init__(self, env: QuadcopterEnv):
        """Initialize the default strategy.

        Args:
            env: The quadcopter environment instance.
        """
        self.env = env
        self.device = env.device
        self.num_envs = env.num_envs
        self.cfg = env.cfg

        # Initialize episode sums for logging if in training mode
        if self.cfg.is_train and hasattr(env, 'rew'):
            keys = [key.split("_reward_scale")[0] for key in env.rew.keys() if key != "death_cost"]
            self._episode_sums = {
                key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
                for key in keys
            }

        # Initialize fixed parameters once (no domain randomization)
        # These parameters remain constant throughout the simulation
        # Aerodynamic drag coefficients
        self.env._K_aero[:, :2] = self.env._k_aero_xy_value
        self.env._K_aero[:, 2] = self.env._k_aero_z_value

        # PID controller gains for angular rate control
        # Roll and pitch use the same gains
        self.env._kp_omega[:, :2] = self.env._kp_omega_rp_value
        self.env._ki_omega[:, :2] = self.env._ki_omega_rp_value
        self.env._kd_omega[:, :2] = self.env._kd_omega_rp_value

        # Yaw has different gains
        self.env._kp_omega[:, 2] = self.env._kp_omega_y_value
        self.env._ki_omega[:, 2] = self.env._ki_omega_y_value
        self.env._kd_omega[:, 2] = self.env._kd_omega_y_value

        # Motor time constants (same for all 4 motors)
        self.env._tau_m[:] = self.env._tau_m_value

        # Thrust to weight ratio
        self.env._thrust_to_weight[:] = self.env._twr_value

    def get_rewards(self) -> torch.Tensor:
        """get_rewards() is called per timestep. This is where you define your reward structure and compute them
        according to the reward scales you tune in train_race.py. The following is an example reward structure that
        causes the drone to hover near the zeroth gate. It will not produce a racing policy, but simply serves as proof
        if your PPO implementation works. You should delete it or heavily modify it once you begin the racing task."""

        # TODO ----- START ----- Define the tensors required for your custom reward structure
        drone_pos_w = self.env._robot.data.root_link_pos_w
        drone_pos_gate = self.env._pose_drone_wrt_gate

        if not hasattr(self.env, "_prev_pose_drone_wrt_gate"):
            self.env._prev_pose_drone_wrt_gate = drone_pos_gate.clone()
        if not hasattr(self.env, "_prev_drone_pos_w"):
            self.env._prev_drone_pos_w = drone_pos_w.clone()
        if not hasattr(self.env, "_last_passed_gate_idx"):
            self.env._last_passed_gate_idx = torch.full(
                (self.num_envs,), -1, dtype=self.env._idx_wp.dtype, device=self.device
            )
        if not hasattr(self.env, "_prev_pose_drone_wrt_last_gate"):
            self.env._prev_pose_drone_wrt_last_gate = torch.zeros(
                (self.num_envs, 3), dtype=torch.float, device=self.device
            )

        just_reset = self.env.episode_length_buf <= 1
        prev_pos_gate = torch.where(
            just_reset.unsqueeze(1), drone_pos_gate, self.env._prev_pose_drone_wrt_gate
        )
        prev_drone_pos_w = torch.where(
            just_reset.unsqueeze(1), drone_pos_w, self.env._prev_drone_pos_w
        )

        x_gate = drone_pos_gate[:, 0]
        y_gate = drone_pos_gate[:, 1]
        z_gate = drone_pos_gate[:, 2]
        prev_x_gate = prev_pos_gate[:, 0]
        prev_y_gate = prev_pos_gate[:, 1]
        prev_z_gate = prev_pos_gate[:, 2]

        crossed_plane = (prev_x_gate > 0.0) & (x_gate <= 0.0)

        denom = x_gate - prev_x_gate
        denom = torch.where(denom.abs() < 1e-6, denom + 1e-6, denom)
        alpha = (-prev_x_gate / denom).clamp(0.0, 1.0)

        y_cross = prev_y_gate + alpha * (y_gate - prev_y_gate)
        z_cross = prev_z_gate + alpha * (z_gate - prev_z_gate)

        gate_half_width = 0.5
        gate_half_height = 0.5
        inside_gate = (y_cross.abs() <= gate_half_width) & (z_cross.abs() <= gate_half_height)

        current_gate_idx = self.env._idx_wp
        next_gate_idx = (current_gate_idx + 1) % self.env._waypoints.shape[0]
        segment_to_next = self.env._waypoints[next_gate_idx, :3] - self.env._waypoints[current_gate_idx, :3]
        segment_to_next = torch.nn.functional.normalize(segment_to_next, dim=1, eps=1e-6)
        drone_delta_w = drone_pos_w - prev_drone_pos_w
        forward_course_motion = torch.sum(drone_delta_w * segment_to_next, dim=1) > 0.0

        gate_passed = crossed_plane & inside_gate & forward_course_motion
        ids_gate_passed = torch.where(gate_passed)[0]

        cheated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        valid_last_gate = self.env._last_passed_gate_idx >= 0
        valid_last_ids = torch.where(valid_last_gate)[0]
        if len(valid_last_ids) > 0:
            last_gate_idx = self.env._last_passed_gate_idx[valid_last_ids]
            pose_wrt_last_gate, _ = subtract_frame_transforms(
                self.env._waypoints[last_gate_idx, :3],
                self.env._waypoints_quat[last_gate_idx, :],
                drone_pos_w[valid_last_ids],
            )
            prev_pose_wrt_last_gate = torch.where(
                just_reset[valid_last_ids].unsqueeze(1),
                pose_wrt_last_gate,
                self.env._prev_pose_drone_wrt_last_gate[valid_last_ids],
            )

            prev_x_last = prev_pose_wrt_last_gate[:, 0]
            x_last = pose_wrt_last_gate[:, 0]
            prev_y_last = prev_pose_wrt_last_gate[:, 1]
            y_last = pose_wrt_last_gate[:, 1]
            prev_z_last = prev_pose_wrt_last_gate[:, 2]
            z_last = pose_wrt_last_gate[:, 2]

            crossed_back_plane = (prev_x_last <= 0.0) & (x_last > 0.0)
            denom_last = x_last - prev_x_last
            denom_last = torch.where(denom_last.abs() < 1e-6, denom_last + 1e-6, denom_last)
            alpha_last = (-prev_x_last / denom_last).clamp(0.0, 1.0)
            y_cross_last = prev_y_last + alpha_last * (y_last - prev_y_last)
            z_cross_last = prev_z_last + alpha_last * (z_last - prev_z_last)
            inside_last_gate = (y_cross_last.abs() <= gate_half_width) & (z_cross_last.abs() <= gate_half_height)

            cheated_last = crossed_back_plane & inside_last_gate
            cheated[valid_last_ids] = cheated_last
            self.env._prev_pose_drone_wrt_last_gate[valid_last_ids] = pose_wrt_last_gate.detach()

        desired_pos_w = self.env._desired_pos_w
        distance_to_goal = torch.linalg.norm(desired_pos_w - drone_pos_w, dim=1)
        prev_distance = self.env._last_distance_to_goal
        forward_progress = torch.clamp(prev_distance - distance_to_goal, min=0.0, max=0.2)
        progress = forward_progress + 0.5 * gate_passed.float()
        self.env._last_distance_to_goal = distance_to_goal

        if len(ids_gate_passed) > 0:
            passed_gate_idx = current_gate_idx[ids_gate_passed].clone()
            self.env._idx_wp[ids_gate_passed] = (self.env._idx_wp[ids_gate_passed] + 1) % self.env._waypoints.shape[0]
            self.env._n_gates_passed[ids_gate_passed] += 1
            self.env._desired_pos_w[ids_gate_passed, :2] = self.env._waypoints[self.env._idx_wp[ids_gate_passed], :2]
            self.env._desired_pos_w[ids_gate_passed, 2] = self.env._waypoints[self.env._idx_wp[ids_gate_passed], 2]
            self.env._last_distance_to_goal[ids_gate_passed] = torch.linalg.norm(
                self.env._desired_pos_w[ids_gate_passed] - drone_pos_w[ids_gate_passed], dim=1
            )
            self.env._last_passed_gate_idx[ids_gate_passed] = passed_gate_idx
            self.env._prev_pose_drone_wrt_last_gate[ids_gate_passed] = drone_pos_gate[ids_gate_passed].detach()

        self.env._prev_x_drone_wrt_gate = x_gate.detach()
        self.env._prev_pose_drone_wrt_gate = drone_pos_gate.detach()
        self.env._prev_drone_pos_w = drone_pos_w.detach()

        contact_forces = self.env._contact_sensor.data.net_forces_w
        crashed = (torch.norm(contact_forces, dim=-1) > 1e-8).squeeze(1).int()
        crashed = torch.maximum(crashed, cheated.int())
        mask = (self.env.episode_length_buf > 100).int()
        self.env._crashed = self.env._crashed + crashed * mask
        # TODO ----- END -----

        if self.cfg.is_train:
            # TODO ----- START ----- Compute per-timestep rewards by multiplying with your reward scales (in train_race.py)
            rewards = {
                "progress_goal": progress * self.env.rew['progress_goal_reward_scale'],
                "crash": crashed.float() * self.env.rew['crash_reward_scale'],
            }
            reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
            reward = torch.where(self.env.reset_terminated,
                                torch.ones_like(reward) * self.env.rew['death_cost'], reward)

            # Logging
            for key, value in rewards.items():
                self._episode_sums[key] += value
        else:   # This else condition implies eval is called with play_race.py. Can be useful to debug at test-time
            reward = torch.zeros(self.num_envs, device=self.device)
            # TODO ----- END -----

        return reward

    def get_observations(self) -> Dict[str, torch.Tensor]:
        """Get observations. Read reset_idx() and quadcopter_env.py to see which drone info is extracted from the sim.
        The following code is an example. You should delete it or heavily modify it once you begin the racing task."""

        # TODO ----- START ----- Define tensors for your observation space. Be careful with frame transformations
        curr_idx = self.env._idx_wp % self.env._waypoints.shape[0]
        next_idx = (self.env._idx_wp + 1) % self.env._waypoints.shape[0]

        wp_curr_pos = self.env._waypoints[curr_idx, :3]
        wp_next_pos = self.env._waypoints[next_idx, :3]
        quat_curr = self.env._waypoints_quat[curr_idx]
        quat_next = self.env._waypoints_quat[next_idx]

        rot_curr = matrix_from_quat(quat_curr)
        rot_next = matrix_from_quat(quat_next)

        verts_curr = (
            torch.bmm(self.env._local_square, rot_curr.transpose(1, 2))
            + wp_curr_pos.unsqueeze(1)
            + self.env._terrain.env_origins.unsqueeze(1)
        )
        verts_next = (
            torch.bmm(self.env._local_square, rot_next.transpose(1, 2))
            + wp_next_pos.unsqueeze(1)
            + self.env._terrain.env_origins.unsqueeze(1)
        )

        waypoint_pos_b_curr, _ = subtract_frame_transforms(
            self.env._robot.data.root_link_state_w[:, :3].repeat_interleave(4, dim=0),
            self.env._robot.data.root_link_state_w[:, 3:7].repeat_interleave(4, dim=0),
            verts_curr.view(-1, 3),
        )
        waypoint_pos_b_next, _ = subtract_frame_transforms(
            self.env._robot.data.root_link_state_w[:, :3].repeat_interleave(4, dim=0),
            self.env._robot.data.root_link_state_w[:, 3:7].repeat_interleave(4, dim=0),
            verts_next.view(-1, 3),
        )

        waypoint_pos_b_curr = waypoint_pos_b_curr.view(self.num_envs, 4, 3)
        waypoint_pos_b_next = waypoint_pos_b_next.view(self.num_envs, 4, 3)

        quat_w = self.env._robot.data.root_quat_w
        attitude_mat = matrix_from_quat(quat_w)

        rpy = euler_xyz_from_quat(quat_w)
        yaw_w = wrap_to_pi(rpy[2])
        delta_yaw = yaw_w - self.env._previous_yaw
        self.env._previous_yaw = yaw_w
        self.env._yaw_n_laps += torch.where(delta_yaw < -np.pi, 1, 0)
        self.env._yaw_n_laps -= torch.where(delta_yaw > np.pi, 1, 0)
        self.env.unwrapped_yaw = yaw_w + 2 * np.pi * self.env._yaw_n_laps
        self.env._previous_actions = self.env._actions.clone()

        # TODO ----- END -----

        obs = torch.cat(
            # TODO ----- START ----- List your observation tensors here to be concatenated together
            [
                self.env._robot.data.root_com_lin_vel_b,
                attitude_mat.view(attitude_mat.shape[0], -1),
                waypoint_pos_b_curr.view(waypoint_pos_b_curr.shape[0], -1),
                waypoint_pos_b_next.view(waypoint_pos_b_next.shape[0], -1),
            ],
            # TODO ----- END -----
            dim=-1,
        )
        observations = {"policy": obs}

        return observations

    def reset_idx(self, env_ids: Optional[torch.Tensor]):
        """Reset specific environments to initial states."""
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.env._robot._ALL_INDICES

        # Logging for training mode
        if self.cfg.is_train and hasattr(self, '_episode_sums'):
            extras = dict()
            for key in self._episode_sums.keys():
                episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
                extras["Episode_Reward/" + key] = episodic_sum_avg / self.env.max_episode_length_s
                self._episode_sums[key][env_ids] = 0.0
            self.env.extras["log"] = dict()
            self.env.extras["log"].update(extras)
            extras = dict()
            extras["Episode_Termination/died"] = torch.count_nonzero(self.env.reset_terminated[env_ids]).item()
            extras["Episode_Termination/time_out"] = torch.count_nonzero(self.env.reset_time_outs[env_ids]).item()
            self.env.extras["log"].update(extras)

        # Call robot reset first
        self.env._robot.reset(env_ids)

        # Initialize model paths if needed
        if not self.env._models_paths_initialized:
            num_models_per_env = self.env._waypoints.size(0)
            model_prim_names_in_env = [f"{self.env.target_models_prim_base_name}_{i}" for i in range(num_models_per_env)]

            self.env._all_target_models_paths = []
            for env_path in self.env.scene.env_prim_paths:
                paths_for_this_env = [f"{env_path}/{name}" for name in model_prim_names_in_env]
                self.env._all_target_models_paths.append(paths_for_this_env)

            self.env._models_paths_initialized = True

        n_reset = len(env_ids)
        if n_reset == self.num_envs and self.num_envs > 1:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf,
                                                             high=int(self.env.max_episode_length))

        # Reset action buffers
        self.env._actions[env_ids] = 0.0
        self.env._previous_actions[env_ids] = 0.0
        self.env._previous_yaw[env_ids] = 0.0
        self.env._motor_speeds[env_ids] = 0.0
        self.env._previous_omega_meas[env_ids] = 0.0
        self.env._previous_omega_err[env_ids] = 0.0
        self.env._omega_err_integral[env_ids] = 0.0

        # Reset joints state
        joint_pos = self.env._robot.data.default_joint_pos[env_ids]
        joint_vel = self.env._robot.data.default_joint_vel[env_ids]
        self.env._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        default_root_state = self.env._robot.data.default_root_state[env_ids]

        # TODO ----- START ----- Define the initial state during training after resetting an environment.
        if self.cfg.is_train:
            num_gates = self.env._waypoints.shape[0]
            gate0_mask = torch.rand(n_reset, device=self.device) < 0.10

            waypoint_indices = torch.randint(
                low=0,
                high=num_gates,
                size=(n_reset,),
                device=self.device,
                dtype=self.env._idx_wp.dtype,
            )
            waypoint_indices[gate0_mask] = 0

            x_local = torch.empty(n_reset, device=self.device)
            y_local = torch.empty(n_reset, device=self.device)
            z_local = torch.empty(n_reset, device=self.device)

            num_gate0 = int(gate0_mask.sum().item())
            if num_gate0 > 0:
                x_local[gate0_mask] = torch.empty(num_gate0, device=self.device).uniform_(-3.0, -0.5)
                y_local[gate0_mask] = torch.empty(num_gate0, device=self.device).uniform_(-1.0, 1.0)
                z_local[gate0_mask] = 0.05

            non_mask = ~gate0_mask
            num_non = int(non_mask.sum().item())
            if num_non > 0:
                x_local[non_mask] = torch.empty(num_non, device=self.device).uniform_(-3.5, -0.3)
                y_local[non_mask] = torch.empty(num_non, device=self.device).uniform_(-1.5, 1.5)
                z_local[non_mask] = torch.empty(num_non, device=self.device).uniform_(0.05, 0.85)
        else:
            waypoint_indices = torch.zeros(n_reset, device=self.device, dtype=self.env._idx_wp.dtype)
            x_local = torch.zeros(n_reset, device=self.device)
            y_local = torch.zeros(n_reset, device=self.device)
            z_local = torch.full((n_reset,), 0.05, device=self.device)

        x0_wp = self.env._waypoints[waypoint_indices][:, 0]
        y0_wp = self.env._waypoints[waypoint_indices][:, 1]
        z_wp = self.env._waypoints[waypoint_indices][:, 2]
        theta = self.env._waypoints[waypoint_indices][:, -1]

        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        x_rot = cos_theta * x_local - sin_theta * y_local
        y_rot = sin_theta * x_local + cos_theta * y_local

        initial_x = x0_wp - x_rot
        initial_y = y0_wp - y_rot
        initial_z = z_local

        default_root_state[:, 0] = initial_x
        default_root_state[:, 1] = initial_y
        default_root_state[:, 2] = initial_z

        initial_yaw = torch.atan2(y0_wp - initial_y, x0_wp - initial_x)
        yaw_jitter = torch.empty(n_reset, device=self.device).uniform_(-0.12, 0.12)
        quat = quat_from_euler_xyz(
            torch.zeros(n_reset, device=self.device),
            torch.zeros(n_reset, device=self.device),
            initial_yaw + yaw_jitter,
        )
        default_root_state[:, 3:7] = quat
        # TODO ----- END -----

        # Handle play mode initial position
        if not self.cfg.is_train:
            # x_local and y_local are randomly sampled
            x_local = torch.empty(1, device=self.device).uniform_(-3.0, -0.5)
            y_local = torch.empty(1, device=self.device).uniform_(-1.0, 1.0)

            x0_wp = self.env._waypoints[self.env._initial_wp, 0]
            y0_wp = self.env._waypoints[self.env._initial_wp, 1]
            theta = self.env._waypoints[self.env._initial_wp, -1]

            # rotate local pos to global frame
            cos_theta, sin_theta = torch.cos(theta), torch.sin(theta)
            x_rot = cos_theta * x_local - sin_theta * y_local
            y_rot = sin_theta * x_local + cos_theta * y_local
            x0 = x0_wp - x_rot
            y0 = y0_wp - y_rot
            z0 = 0.05

            # point drone towards the zeroth gate
            yaw0 = torch.atan2(y0_wp - y0, x0_wp - x0)

            default_root_state = self.env._robot.data.default_root_state[0].unsqueeze(0)
            default_root_state[:, 0] = x0
            default_root_state[:, 1] = y0
            default_root_state[:, 2] = z0

            quat = quat_from_euler_xyz(
                torch.zeros(1, device=self.device),
                torch.zeros(1, device=self.device),
                yaw0
            )
            default_root_state[:, 3:7] = quat
            waypoint_indices = self.env._initial_wp

        # Set waypoint indices and desired positions
        self.env._idx_wp[env_ids] = waypoint_indices

        self.env._desired_pos_w[env_ids, :2] = self.env._waypoints[waypoint_indices, :2].clone()
        self.env._desired_pos_w[env_ids, 2] = self.env._waypoints[waypoint_indices, 2].clone()

        self.env._last_distance_to_goal[env_ids] = torch.linalg.norm(
            self.env._desired_pos_w[env_ids, :2] - self.env._robot.data.root_link_pos_w[env_ids, :2], dim=1
        )
        self.env._n_gates_passed[env_ids] = 0

        # Write state to simulation
        self.env._robot.write_root_link_pose_to_sim(default_root_state[:, :7], env_ids)
        self.env._robot.write_root_com_velocity_to_sim(default_root_state[:, 7:], env_ids)

        # Reset variables
        self.env._yaw_n_laps[env_ids] = 0

        self.env._pose_drone_wrt_gate[env_ids], _ = subtract_frame_transforms(
            self.env._waypoints[self.env._idx_wp[env_ids], :3],
            self.env._waypoints_quat[self.env._idx_wp[env_ids], :],
            self.env._robot.data.root_link_state_w[env_ids, :3]
        )

        self.env._prev_x_drone_wrt_gate[env_ids] = 1.0

        self.env._crashed[env_ids] = 0
