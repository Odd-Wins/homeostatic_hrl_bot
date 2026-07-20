"""Sparse reward (no homeostatic drive-reduction) for ablation .

Retains: step cost, collision penalty, goal bonus, death penalty.
Isolates the contribution of the homeostatic signal.
"""

from dataclasses import dataclass

import numpy as np


# Observation indices - keep in sync with env_wrapper._compute_obs()
_IDX_SOC = 5
_IDX_LIDAR_FRONT = 7
_IDX_LIDAR_LEFT = 8
_IDX_LIDAR_RIGHT = 9


@dataclass
class NoHomeostaticReward:
    #Sparse goal/death + step cost + collision. No drive-reduction

    goal_bonus: float = 10.0
    death_penalty: float = 10.0
    step_cost: float = 0.1
    collision_penalty: float = 1.0
    collision_threshold: float = 0.25

    def __call__(self, prev_obs, action, next_obs, info):
        #Compute reward for one (s, a, s') transition.
        reward = 0.0
        # No homeostatic drive-reduction
        # Step cost (every step).
        reward -= self.step_cost

        # Collision penalty- per-step while in danger zone.
        lidar_min = min(
            next_obs[_IDX_LIDAR_FRONT],
            next_obs[_IDX_LIDAR_LEFT],
            next_obs[_IDX_LIDAR_RIGHT],
        )
        if lidar_min < self.collision_threshold:
            reward -= self.collision_penalty

        # Sparse terminal bonuses.
        if info.get("reached_goal", False):
            reward += self.goal_bonus
        if info.get("battery_dead", False):
            reward -= self.death_penalty

        return float(reward)
