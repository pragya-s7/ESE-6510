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
        if not hasattr(self.env, "_special_pair_over_top_satisfied"):
            self.env._special_pair_over_top_satisfied = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
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
        crossed_plane_wrong_way = (prev_x_gate <= 0.0) & (x_gate > 0.0)

        denom = x_gate - prev_x_gate
        denom = torch.where(denom.abs() < 1e-6, denom + 1e-6, denom)
        alpha = (-prev_x_gate / denom).clamp(0.0, 1.0)

        y_cross = prev_y_gate + alpha * (y_gate - prev_y_gate)
        z_cross = prev_z_gate + alpha * (z_gate - prev_z_gate)

        gate_half_extent = 0.5 #changing to 0.25 to test what smaller window encourages. 
        inside_gate = (y_cross.abs() <= gate_half_extent) & (z_cross.abs() <= gate_half_extent)

        current_gate_idx = self.env._idx_wp
        next_gate_idx = (current_gate_idx + 1) % self.env._waypoints.shape[0]
        segment_to_next = self.env._waypoints[next_gate_idx, :3] - self.env._waypoints[current_gate_idx, :3]
        segment_to_next = torch.nn.functional.normalize(segment_to_next, dim=1, eps=1e-6)
        drone_delta_w = drone_pos_w - prev_drone_pos_w
        forward_course_motion = torch.sum(drone_delta_w * segment_to_next, dim=1) > 0.0

        gate_passed = crossed_plane & inside_gate & forward_course_motion
        cheated_active_gate = inside_gate & crossed_plane_wrong_way
        ids_gate_passed = torch.where(gate_passed)[0]

        cheated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        cheated |= cheated_active_gate
        cheated_special_pair_pass = torch.zeros(self.num_envs, dtype=torch.bool, device =self.device)
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
            inside_last_gate = (y_cross_last.abs() <= gate_half_extent) & (z_cross_last.abs() <= gate_half_extent)

            cheated_last = crossed_back_plane & inside_last_gate
            cheated[valid_last_ids] |= cheated_last
            self.env._prev_pose_drone_wrt_last_gate[valid_last_ids] = pose_wrt_last_gate.detach()
            
        cheated |= cheated_special_pair_pass
        gate_passed = gate_passed & (~cheated_special_pair_pass)
        ids_gate_passed = torch.where(gate_passed)[0]
        """desired_pos_w = self.env._desired_pos_w
        distance_to_goal = torch.linalg.norm(desired_pos_w - drone_pos_w, dim=1)
        
        Testing new gate window reward
        """
        d2goal = self._dis_to_active_gate(drone_pos_gate)
        prev_distance = self.env._last_distance_to_goal
        forward_progress = torch.clamp(prev_distance - d2goal, min=0.0, max=0.2)
        gate_pass_reward = gate_passed.float()
        # Reversible speed shaping: set segment_velocity_reward_scale=0.0 in train_race.py
        # to recover the old reward structure.
        step_dt = max(float(self.cfg.sim.dt * self.cfg.decimation), 1e-6)
        segment_velocity = torch.clamp(torch.sum(drone_delta_w * segment_to_next, dim=1) / step_dt, min=0.0, max=4.0)
        near_gate_mask = d2goal < 1.5
        segment_velocity_reward = segment_velocity * near_gate_mask.float()
        lap_completed_reward = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.env._last_distance_to_goal = d2goal
        
        if len(ids_gate_passed) > 0:
            passed_gate_idx = current_gate_idx[ids_gate_passed].clone()
            self.env._idx_wp[ids_gate_passed] = (self.env._idx_wp[ids_gate_passed] + 1) % self.env._waypoints.shape[0]
            self.env._n_gates_passed[ids_gate_passed] += 1
            lap_completed_reward[ids_gate_passed] = (
                self.env._n_gates_passed[ids_gate_passed] % self.env._waypoints.shape[0] == 0
            ).float()
            self.env._desired_pos_w[ids_gate_passed, :2] = self.env._waypoints[self.env._idx_wp[ids_gate_passed], :2]
            self.env._desired_pos_w[ids_gate_passed, 2] = self.env._waypoints[self.env._idx_wp[ids_gate_passed], 2]
            self.env._last_distance_to_goal[ids_gate_passed] = torch.linalg.norm(
                self.env._desired_pos_w[ids_gate_passed] - drone_pos_w[ids_gate_passed], dim=1
            )
            new_gate_idx = self.env._idx_wp[ids_gate_passed]
            pose_drone_wrt_new_gate, _ = subtract_frame_transforms(
                self.env._waypoints[new_gate_idx, :3],
                self.env._waypoints_quat[new_gate_idx, :],
                drone_pos_w[ids_gate_passed],
            )
            
            self.env._last_distance_to_goal[ids_gate_passed] = self._dis_to_active_gate(pose_drone_wrt_new_gate)

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
        reset_env_ids = self.env._robot._ALL_INDICES if env_ids is None or len(env_ids) == self.num_envs else env_ids
        num_resets = len(reset_env_ids)

        # Average episode statistics over the environments that are being reset.
        if self.cfg.is_train and hasattr(self, "_episode_sums"):
            reward_logs = {}
            for reward_name in self._episode_sums.keys():
                episode_average = torch.mean(self._episode_sums[reward_name][reset_env_ids])
                reward_logs[f"Episode_Reward/{reward_name}"] = episode_average / self.env.max_episode_length_s
                self._episode_sums[reward_name][reset_env_ids] = 0.0

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
            use_priority_gate_curriculum = False
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
                # Legacy focused curriculum: close, centered, and elevated near the chosen gate.
                spawn_x_local[priority_env_ids] = torch.empty(len(priority_env_ids), device=self.device).uniform_(-0.2, 0.2)
                spawn_y_local[priority_env_ids] = torch.empty(len(priority_env_ids), device=self.device).uniform_(-0.5, 0.5)
                spawn_z_local[priority_env_ids] = torch.empty(len(priority_env_ids), device=self.device).uniform_(0.55, 1.25)

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
        self.env._crashed[reset_env_ids] = 0
