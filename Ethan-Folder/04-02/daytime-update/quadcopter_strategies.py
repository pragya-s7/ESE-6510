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

    def _dis_to_active_gate(self, pose_drone_wrt_gate: torch.Tensor) -> torch.Tensor:
        # Previous "window approach" kept here for reference while we move back to a
        # simpler geometric target based on the gate plane itself.
        
        h = 0.45  # target window half extent 
        x_gate = pose_drone_wrt_gate[:, 0]
        y_gate = pose_drone_wrt_gate[:, 1]
        z_gate = pose_drone_wrt_gate[:, 2]
        y_nearest = torch.clamp(y_gate, -h, h)
        z_nearest = torch.clamp(z_gate, -h, h)
        return torch.sqrt(
            x_gate.square()
            + (y_gate - y_nearest).square()
            + (z_gate - z_nearest).square()
        )

        # In gate coordinates, the gate plane is x = 0. The nearest point on that
        # plane keeps the same y and z coordinates, so the distance is just |x|.
        #x_gate = pose_drone_wrt_gate[:, 0]
        #return x_gate.abs()
        

    def get_rewards(self) -> torch.Tensor:
        """get_rewards() is called per timestep. This is where you define your reward structure and compute them
        according to the reward scales you tune in train_race.py. The following is an example reward structure that
        causes the drone to hover near the zeroth gate. It will not produce a racing policy, but simply serves as proof
        if your PPO implementation works. You should delete it or heavily modify it once you begin the racing task."""

        # TODO ----- START ----- Define the tensors required for your custom reward structure
        world_pos = self.env._robot.data.root_link_pos_w
        gate_frame_pos = self.env._pose_drone_wrt_gate

        # Lazily allocate reward history that persists across environment steps.
        if not hasattr(self.env, "_prev_pose_drone_wrt_gate"):
            self.env._prev_pose_drone_wrt_gate = gate_frame_pos.clone()
        if not hasattr(self.env, "_prev_drone_pos_w"):
            self.env._prev_drone_pos_w = world_pos.clone()
        if not hasattr(self.env, "_last_passed_gate_idx"):
            self.env._last_passed_gate_idx = torch.full(
                (self.num_envs,), -1, dtype=self.env._idx_wp.dtype, device=self.device
            )
        if not hasattr(self.env, "_prev_pose_drone_wrt_last_gate"):
            self.env._prev_pose_drone_wrt_last_gate = torch.zeros(
                (self.num_envs, 3), dtype=torch.float, device=self.device
            )
        if not hasattr(self.env, "_special_pair_over_top_satisfied"):
            self.env._special_pair_over_top_satisfied = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            )
        if not hasattr(self.env, "_lap_time_sum"):
            self.env._lap_time_sum = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        if not hasattr(self.env, "_lap_count"):
            self.env._lap_count = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        if not hasattr(self.env, "_last_lap_step"):
            self.env._last_lap_step = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        just_reset_mask = self.env.episode_length_buf <= 1
        prev_gate_frame_pos = torch.where(
            just_reset_mask.unsqueeze(1), gate_frame_pos, self.env._prev_pose_drone_wrt_gate
        )
        prev_world_pos = torch.where(
            just_reset_mask.unsqueeze(1), world_pos, self.env._prev_drone_pos_w
        )

        # Compute gate-plane crossings in the active gate frame.
        gate_x = gate_frame_pos[:, 0]
        gate_y = gate_frame_pos[:, 1]
        gate_z = gate_frame_pos[:, 2]
        prev_gate_x = prev_gate_frame_pos[:, 0]
        prev_gate_y = prev_gate_frame_pos[:, 1]
        prev_gate_z = prev_gate_frame_pos[:, 2]

        crossed_forward_plane = (prev_gate_x > 0.0) & (gate_x <= 0.0)
        crossed_backward_plane = (prev_gate_x <= 0.0) & (gate_x > 0.0)

        plane_cross_denom = gate_x - prev_gate_x
        plane_cross_denom = torch.where(plane_cross_denom.abs() < 1e-6, plane_cross_denom + 1e-6, plane_cross_denom)
        plane_cross_alpha = (-prev_gate_x / plane_cross_denom).clamp(0.0, 1.0)
        plane_cross_y = prev_gate_y + plane_cross_alpha * (gate_y - prev_gate_y)
        plane_cross_z = prev_gate_z + plane_cross_alpha * (gate_z - prev_gate_z)

        gate_half_extent = 0.5  # Matches the physical gate opening.
        crossed_inside_gate = (plane_cross_y.abs() <= gate_half_extent) & (plane_cross_z.abs() <= gate_half_extent)

        # Use the active-to-next segment for both legality and optional speed shaping.
        active_gate_idx = self.env._idx_wp
        next_gate_idx = (active_gate_idx + 1) % self.env._waypoints.shape[0]
        active_to_next_world = self.env._waypoints[next_gate_idx, :3] - self.env._waypoints[active_gate_idx, :3]
        active_to_next_unit = torch.nn.functional.normalize(active_to_next_world, dim=1, eps=1e-6)
        step_world_delta = world_pos - prev_world_pos
        moving_along_course = torch.sum(step_world_delta * active_to_next_unit, dim=1) > 0.0

        gate_passed_mask = crossed_forward_plane & crossed_inside_gate & moving_along_course
        wrong_way_current_gate_mask = crossed_inside_gate & crossed_backward_plane

        cheated_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        cheated_mask |= wrong_way_current_gate_mask
        cheated_special_pair_pass = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Check whether the drone illegally re-crossed the previously passed gate.
        has_last_gate_mask = self.env._last_passed_gate_idx >= 0
        envs_with_last_gate = torch.where(has_last_gate_mask)[0]
        if len(envs_with_last_gate) > 0:
            last_gate_idx = self.env._last_passed_gate_idx[envs_with_last_gate]
            current_pose_wrt_last_gate, _ = subtract_frame_transforms(
                self.env._waypoints[last_gate_idx, :3],
                self.env._waypoints_quat[last_gate_idx, :],
                world_pos[envs_with_last_gate],
            )
            prev_pose_wrt_last_gate = torch.where(
                just_reset_mask[envs_with_last_gate].unsqueeze(1),
                current_pose_wrt_last_gate,
                self.env._prev_pose_drone_wrt_last_gate[envs_with_last_gate],
            )

            prev_last_gate_x = prev_pose_wrt_last_gate[:, 0]
            current_last_gate_x = current_pose_wrt_last_gate[:, 0]
            prev_last_gate_y = prev_pose_wrt_last_gate[:, 1]
            current_last_gate_y = current_pose_wrt_last_gate[:, 1]
            prev_last_gate_z = prev_pose_wrt_last_gate[:, 2]
            current_last_gate_z = current_pose_wrt_last_gate[:, 2]

            last_gate_crossed_backward = (prev_last_gate_x <= 0.0) & (current_last_gate_x > 0.0)
            last_gate_denom = current_last_gate_x - prev_last_gate_x
            last_gate_denom = torch.where(last_gate_denom.abs() < 1e-6, last_gate_denom + 1e-6, last_gate_denom)
            last_gate_alpha = (-prev_last_gate_x / last_gate_denom).clamp(0.0, 1.0)
            last_gate_cross_y = prev_last_gate_y + last_gate_alpha * (current_last_gate_y - prev_last_gate_y)
            last_gate_cross_z = prev_last_gate_z + last_gate_alpha * (current_last_gate_z - prev_last_gate_z)
            crossed_inside_last_gate = (
                (last_gate_cross_y.abs() <= gate_half_extent) & (last_gate_cross_z.abs() <= gate_half_extent)
            )

            cheated_last_gate_mask = last_gate_crossed_backward & crossed_inside_last_gate
            cheated_mask[envs_with_last_gate] |= cheated_last_gate_mask
            self.env._prev_pose_drone_wrt_last_gate[envs_with_last_gate] = current_pose_wrt_last_gate.detach()

        cheated_mask |= cheated_special_pair_pass
        gate_passed_mask = gate_passed_mask & (~cheated_special_pair_pass)
        passed_env_ids = torch.where(gate_passed_mask)[0]
        """desired_pos_w = self.env._desired_pos_w
        distance_to_goal = torch.linalg.norm(desired_pos_w - world_pos, dim=1)
        
        Testing new gate window reward
        """
        distance_to_gate = self._dis_to_active_gate(gate_frame_pos)
        prev_distance_to_gate = self.env._last_distance_to_goal
        forward_progress = torch.clamp(prev_distance_to_gate - distance_to_gate, min=0.0, max=0.2)
        gate_pass_reward = gate_passed_mask.float()

        # Reversible speed shaping: set segment_velocity_reward_scale=0.0 in train_race.py
        # to recover the old reward structure.
        step_dt = max(float(self.cfg.sim.dt * self.cfg.decimation), 1e-6)
        segment_velocity = torch.clamp(
            torch.sum(step_world_delta * active_to_next_unit, dim=1) / step_dt, min=0.0, max=4.0
        )
        near_gate_mask = distance_to_gate < 1.5
        segment_velocity_reward = segment_velocity * near_gate_mask.float()
        lap_completed_reward = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.env._last_distance_to_goal = distance_to_gate

        # Advance the active gate after legal passes.
        if len(passed_env_ids) > 0:
            passed_gate_idx = active_gate_idx[passed_env_ids].clone()
            self.env._idx_wp[passed_env_ids] = (self.env._idx_wp[passed_env_ids] + 1) % self.env._waypoints.shape[0]
            self.env._n_gates_passed[passed_env_ids] += 1
            lap_completed_reward[passed_env_ids] = (
                self.env._n_gates_passed[passed_env_ids] % self.env._waypoints.shape[0] == 0
            ).float()
            lap_completed_mask = lap_completed_reward[passed_env_ids] > 0.0
            if torch.any(lap_completed_mask):
                lap_completed_env_ids = passed_env_ids[lap_completed_mask]
                current_lap_step = self.env.episode_length_buf[lap_completed_env_ids].float()
                lap_step_delta = current_lap_step - self.env._last_lap_step[lap_completed_env_ids]
                lap_time_s = lap_step_delta * float(self.cfg.sim.dt * self.cfg.decimation)
                self.env._lap_time_sum[lap_completed_env_ids] += lap_time_s
                self.env._lap_count[lap_completed_env_ids] += 1.0
                self.env._last_lap_step[lap_completed_env_ids] = current_lap_step

            self.env._desired_pos_w[passed_env_ids, :2] = self.env._waypoints[self.env._idx_wp[passed_env_ids], :2]
            self.env._desired_pos_w[passed_env_ids, 2] = self.env._waypoints[self.env._idx_wp[passed_env_ids], 2]
            self.env._last_distance_to_goal[passed_env_ids] = torch.linalg.norm(
                self.env._desired_pos_w[passed_env_ids] - world_pos[passed_env_ids], dim=1
            )

            new_gate_idx = self.env._idx_wp[passed_env_ids]
            pose_wrt_new_gate, _ = subtract_frame_transforms(
                self.env._waypoints[new_gate_idx, :3],
                self.env._waypoints_quat[new_gate_idx, :],
                world_pos[passed_env_ids],
            )

            self.env._last_distance_to_goal[passed_env_ids] = self._dis_to_active_gate(pose_wrt_new_gate)
            self.env._last_passed_gate_idx[passed_env_ids] = passed_gate_idx
            self.env._prev_pose_drone_wrt_last_gate[passed_env_ids] = gate_frame_pos[passed_env_ids].detach()

        # Store the latest pose for use on the next reward step.
        self.env._prev_x_drone_wrt_gate = gate_x.detach()
        self.env._prev_pose_drone_wrt_gate = gate_frame_pos.detach()
        self.env._prev_drone_pos_w = world_pos.detach()

        contact_forces = self.env._contact_sensor.data.net_forces_w
        crashed = (torch.norm(contact_forces, dim=-1) > 1e-8).squeeze(1).int()
        crashed = torch.maximum(crashed, cheated_mask.int())
        mask = (self.env.episode_length_buf > 100).int()
        self.env._crashed = self.env._crashed + crashed * mask
        # TODO ----- END -----

        if self.cfg.is_train:
            # TODO ----- START ----- Compute per-timestep rewards by multiplying with your reward scales (in train_race.py)
            rewards = {
                "forward_progress": forward_progress * self.env.rew['forward_progress_reward_scale'],
                "gate_pass": gate_pass_reward * self.env.rew['gate_pass_reward_scale'],
                "lap_completed": lap_completed_reward * self.env.rew['lap_completed_reward_scale'],
                "segment_velocity": segment_velocity_reward * self.env.rew['segment_velocity_reward_scale'],
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
        current_gate_idx = self.env._idx_wp % self.env._waypoints.shape[0]
        next_gate_idx = (self.env._idx_wp + 1) % self.env._waypoints.shape[0]

        current_gate_pos = self.env._waypoints[current_gate_idx, :3]
        next_gate_pos = self.env._waypoints[next_gate_idx, :3]
        current_gate_quat = self.env._waypoints_quat[current_gate_idx]
        next_gate_quat = self.env._waypoints_quat[next_gate_idx]

        current_gate_rot = matrix_from_quat(current_gate_quat)
        next_gate_rot = matrix_from_quat(next_gate_quat)

        # Build the current and next gate corners in world coordinates.
        current_gate_corners_w = (
            torch.bmm(self.env._local_square, current_gate_rot.transpose(1, 2))
            + current_gate_pos.unsqueeze(1)
            + self.env._terrain.env_origins.unsqueeze(1)
        )
        next_gate_corners_w = (
            torch.bmm(self.env._local_square, next_gate_rot.transpose(1, 2))
            + next_gate_pos.unsqueeze(1)
            + self.env._terrain.env_origins.unsqueeze(1)
        )

        # Express those corners in the drone body frame for the policy.
        current_gate_corners_b, _ = subtract_frame_transforms(
            self.env._robot.data.root_link_state_w[:, :3].repeat_interleave(4, dim=0),
            self.env._robot.data.root_link_state_w[:, 3:7].repeat_interleave(4, dim=0),
            current_gate_corners_w.view(-1, 3),
        )
        next_gate_corners_b, _ = subtract_frame_transforms(
            self.env._robot.data.root_link_state_w[:, :3].repeat_interleave(4, dim=0),
            self.env._robot.data.root_link_state_w[:, 3:7].repeat_interleave(4, dim=0),
            next_gate_corners_w.view(-1, 3),
        )

        current_gate_corners_b = current_gate_corners_b.view(self.num_envs, 4, 3)
        next_gate_corners_b = next_gate_corners_b.view(self.num_envs, 4, 3)

        world_quat = self.env._robot.data.root_quat_w
        attitude_mat = matrix_from_quat(world_quat)

        # Keep the existing unwrapped yaw bookkeeping identical to the old version.
        roll_pitch_yaw = euler_xyz_from_quat(world_quat)
        wrapped_yaw = wrap_to_pi(roll_pitch_yaw[2])
        delta_yaw = wrapped_yaw - self.env._previous_yaw
        self.env._previous_yaw = wrapped_yaw
        self.env._yaw_n_laps += torch.where(delta_yaw < -np.pi, 1, 0)
        self.env._yaw_n_laps -= torch.where(delta_yaw > np.pi, 1, 0)
        self.env.unwrapped_yaw = wrapped_yaw + 2 * np.pi * self.env._yaw_n_laps
        self.env._previous_actions = self.env._actions.clone()

        # TODO ----- END -----

        obs = torch.cat(
            # TODO ----- START ----- List your observation tensors here to be concatenated together
            [
                self.env._robot.data.root_com_lin_vel_b,
                self.env._robot.data.root_ang_vel_b,
                attitude_mat.view(attitude_mat.shape[0], -1),
                current_gate_corners_b.view(current_gate_corners_b.shape[0], -1),
                next_gate_corners_b.view(next_gate_corners_b.shape[0], -1),
                self.env._previous_actions,
            ],
            # TODO ----- END -----
            dim=-1,
        )
        observations = {"policy": obs}

        return observations

    def reset_idx(self, env_ids: Optional[torch.Tensor]):
        """Reset specific environments to initial states."""
        reset_env_ids = self.env._robot._ALL_INDICES if env_ids is None or len(env_ids) == self.num_envs else env_ids
        num_resets = len(reset_env_ids)

        # Average episode statistics over the environments that are being reset.
        if self.cfg.is_train and hasattr(self, "_episode_sums"):
            reward_logs = {}
            for reward_name in self._episode_sums.keys():
                episode_average = torch.mean(self._episode_sums[reward_name][reset_env_ids])
                reward_logs[f"Episode_Reward/{reward_name}"] = episode_average / self.env.max_episode_length_s
                self._episode_sums[reward_name][reset_env_ids] = 0.0

            if hasattr(self.env, "_lap_count"):
                lap_count_sum = torch.sum(self.env._lap_count[reset_env_ids])
                if lap_count_sum > 0:
                    lap_time_sum = torch.sum(self.env._lap_time_sum[reset_env_ids])
                    reward_logs["Episode_Time/avg_completed_lap_time"] = (lap_time_sum / lap_count_sum).item()

            termination_logs = {
                "Episode_Termination/died": torch.count_nonzero(self.env.reset_terminated[reset_env_ids]).item(),
                "Episode_Termination/time_out": torch.count_nonzero(self.env.reset_time_outs[reset_env_ids]).item(),
            }
            self.env.extras["log"] = {}
            self.env.extras["log"].update(reward_logs)
            self.env.extras["log"].update(termination_logs)

        # Reset the robot before writing fresh state into the simulator.
        self.env._robot.reset(reset_env_ids)

        # Cache the per-environment gate model paths once for later use.
        if not self.env._models_paths_initialized:
            gate_count = self.env._waypoints.size(0)
            gate_prim_names = [f"{self.env.target_models_prim_base_name}_{i}" for i in range(gate_count)]
            self.env._all_target_models_paths = []
            for env_path in self.env.scene.env_prim_paths:
                self.env._all_target_models_paths.append([f"{env_path}/{name}" for name in gate_prim_names])
            self.env._models_paths_initialized = True

        # Randomize episode progress when all environments reset together.
        if num_resets == self.num_envs and self.num_envs > 1:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        # Clear controller and actuator history for the reset environments.
        self.env._actions[reset_env_ids] = 0.0
        self.env._previous_actions[reset_env_ids] = 0.0
        self.env._previous_yaw[reset_env_ids] = 0.0
        self.env._motor_speeds[reset_env_ids] = 0.0
        self.env._previous_omega_meas[reset_env_ids] = 0.0
        self.env._previous_omega_err[reset_env_ids] = 0.0
        self.env._omega_err_integral[reset_env_ids] = 0.0

        # Restore joint state to the robot defaults.
        default_joint_pos = self.env._robot.data.default_joint_pos[reset_env_ids]
        default_joint_vel = self.env._robot.data.default_joint_vel[reset_env_ids]
        self.env._robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel, None, reset_env_ids)

        spawn_root_state = self.env._robot.data.default_root_state[reset_env_ids]

        # TODO ----- START ----- Define the initial state during training after resetting an environment.
        if self.cfg.is_train:
            gate_count = self.env._waypoints.shape[0]

            # Optional curriculum over a small list of important gates.
            use_priority_gate_curriculum = True
            priority_gate_reset_prob = 0.35
            priority_gate_indices = torch.tensor([1], device=self.device, dtype=self.env._idx_wp.dtype)

            # Sample which gate should be active immediately after reset.
            next_gate_indices = torch.randint(
                low=0,
                high=gate_count,
                size=(num_resets,),
                device=self.device,
                dtype=self.env._idx_wp.dtype,
            )

            use_priority_mask = (
                torch.rand(num_resets, device=self.device) < priority_gate_reset_prob
            ) if use_priority_gate_curriculum else torch.zeros(num_resets, dtype=torch.bool, device=self.device)

            if torch.any(use_priority_mask):
                sampled_priority_slots = torch.randint(
                    low=0,
                    high=priority_gate_indices.numel(),
                    size=(int(use_priority_mask.sum().item()),),
                    device=self.device,
                )
                next_gate_indices[use_priority_mask] = priority_gate_indices[sampled_priority_slots]

            # Local spawn offsets are expressed in the chosen gate's frame.
            spawn_x_local = torch.empty(num_resets, device=self.device)
            spawn_y_local = torch.empty(num_resets, device=self.device)
            spawn_z_local = torch.empty(num_resets, device=self.device)

            priority_env_ids = torch.where(use_priority_mask)[0]
            broad_env_ids = torch.where(~use_priority_mask)[0]

            if len(priority_env_ids) > 0:
                # Focused starts spawn slightly behind the gate in the legal entry direction.
                spawn_x_local[priority_env_ids] = torch.empty(len(priority_env_ids), device=self.device).uniform_(-0.35, -0.15)
                spawn_y_local[priority_env_ids] = torch.empty(len(priority_env_ids), device=self.device).uniform_(-0.5, 0.5)
                spawn_z_local[priority_env_ids] = torch.empty(len(priority_env_ids), device=self.device).uniform_(1.0, 1.25)

            if len(broad_env_ids) > 0:
                # Broad random starts cover general approaches from behind the target gate.
                spawn_x_local[broad_env_ids] = torch.empty(len(broad_env_ids), device=self.device).uniform_(-3.5, -0.3)
                spawn_y_local[broad_env_ids] = torch.empty(len(broad_env_ids), device=self.device).uniform_(-1.5, 1.5)
                spawn_z_local[broad_env_ids] = torch.empty(len(broad_env_ids), device=self.device).uniform_(0.05, 0.85)
        else:
            next_gate_indices = torch.zeros(num_resets, device=self.device, dtype=self.env._idx_wp.dtype)
            spawn_x_local = torch.zeros(num_resets, device=self.device)
            spawn_y_local = torch.zeros(num_resets, device=self.device)
            spawn_z_local = torch.full((num_resets,), 0.05, device=self.device)

        target_gate_x = self.env._waypoints[next_gate_indices, 0]
        target_gate_y = self.env._waypoints[next_gate_indices, 1]
        target_gate_z = self.env._waypoints[next_gate_indices, 2]
        target_gate_yaw = self.env._waypoints[next_gate_indices, -1]

        cos_yaw = torch.cos(target_gate_yaw)
        sin_yaw = torch.sin(target_gate_yaw)
        spawn_x_world_offset = cos_yaw * spawn_x_local - sin_yaw * spawn_y_local
        spawn_y_world_offset = sin_yaw * spawn_x_local + cos_yaw * spawn_y_local

        spawn_x_world = target_gate_x - spawn_x_world_offset
        spawn_y_world = target_gate_y - spawn_y_world_offset
        spawn_z_world = target_gate_z + spawn_z_local

        spawn_root_state[:, 0] = spawn_x_world
        spawn_root_state[:, 1] = spawn_y_world
        spawn_root_state[:, 2] = spawn_z_world

        spawn_yaw = torch.atan2(target_gate_y - spawn_y_world, target_gate_x - spawn_x_world)
        spawn_yaw_jitter = torch.empty(num_resets, device=self.device).uniform_(-0.12, 0.12)
        spawn_root_state[:, 3:7] = quat_from_euler_xyz(
            torch.zeros(num_resets, device=self.device),
            torch.zeros(num_resets, device=self.device),
            spawn_yaw + spawn_yaw_jitter,
        )
        # TODO ----- END -----

        # Evaluation mode always starts from the configured playback gate.
        if not self.cfg.is_train:
            eval_spawn_x_local = torch.empty(1, device=self.device).uniform_(-3.0, -0.5)
            eval_spawn_y_local = torch.empty(1, device=self.device).uniform_(-1.0, 1.0)

            eval_gate_x = self.env._waypoints[self.env._initial_wp, 0]
            eval_gate_y = self.env._waypoints[self.env._initial_wp, 1]
            eval_gate_yaw = self.env._waypoints[self.env._initial_wp, -1]

            eval_cos_yaw = torch.cos(eval_gate_yaw)
            eval_sin_yaw = torch.sin(eval_gate_yaw)
            eval_x_world_offset = eval_cos_yaw * eval_spawn_x_local - eval_sin_yaw * eval_spawn_y_local
            eval_y_world_offset = eval_sin_yaw * eval_spawn_x_local + eval_cos_yaw * eval_spawn_y_local

            eval_spawn_x_world = eval_gate_x - eval_x_world_offset
            eval_spawn_y_world = eval_gate_y - eval_y_world_offset
            eval_spawn_z_world = 0.05
            eval_spawn_yaw = torch.atan2(eval_gate_y - eval_spawn_y_world, eval_gate_x - eval_spawn_x_world)

            spawn_root_state = self.env._robot.data.default_root_state[0].unsqueeze(0)
            spawn_root_state[:, 0] = eval_spawn_x_world
            spawn_root_state[:, 1] = eval_spawn_y_world
            spawn_root_state[:, 2] = eval_spawn_z_world
            spawn_root_state[:, 3:7] = quat_from_euler_xyz(
                torch.zeros(1, device=self.device),
                torch.zeros(1, device=self.device),
                eval_spawn_yaw,
            )
            next_gate_indices = self.env._initial_wp

        # Record which gate is active and where the corresponding target lives in world space.
        self.env._idx_wp[reset_env_ids] = next_gate_indices
        self.env._desired_pos_w[reset_env_ids, :2] = self.env._waypoints[next_gate_indices, :2].clone()
        self.env._desired_pos_w[reset_env_ids, 2] = self.env._waypoints[next_gate_indices, 2].clone()
        self.env._n_gates_passed[reset_env_ids] = 0

        # Push the new pose and velocity state into simulation.
        self.env._robot.write_root_link_pose_to_sim(spawn_root_state[:, :7], reset_env_ids)
        self.env._robot.write_root_com_velocity_to_sim(spawn_root_state[:, 7:], reset_env_ids)

        # Refresh episode-local bookkeeping after the simulator state update.
        self.env._yaw_n_laps[reset_env_ids] = 0
        self.env._pose_drone_wrt_gate[reset_env_ids], _ = subtract_frame_transforms(
            self.env._waypoints[self.env._idx_wp[reset_env_ids], :3],
            self.env._waypoints_quat[self.env._idx_wp[reset_env_ids], :],
            self.env._robot.data.root_link_state_w[reset_env_ids, :3],
        )
        self.env._last_distance_to_goal[reset_env_ids] = self._dis_to_active_gate(
            self.env._pose_drone_wrt_gate[reset_env_ids]
        )
        self.env._prev_x_drone_wrt_gate[reset_env_ids] = 1.0
        if hasattr(self.env, "_special_pair_over_top_satisfied"):
            self.env._special_pair_over_top_satisfied[reset_env_ids] = False
        if hasattr(self.env, "_lap_time_sum"):
            self.env._lap_time_sum[reset_env_ids] = 0.0
        if hasattr(self.env, "_lap_count"):
            self.env._lap_count[reset_env_ids] = 0.0
        if hasattr(self.env, "_last_lap_step"):
            self.env._last_lap_step[reset_env_ids] = self.env.episode_length_buf[reset_env_ids].float()
        self.env._crashed[reset_env_ids] = 0
