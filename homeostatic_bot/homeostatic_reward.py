"""Drive-reduction reward for the homeostatic HRL thesis (Keramati & Gutkin 2014)."""

from dataclasses import dataclass

import numpy as np


# Observation indices — keep in sync with env_wrapper._compute_obs()
_IDX_SOC = 5
_IDX_LIDAR_FRONT = 7
_IDX_LIDAR_LEFT = 8
_IDX_LIDAR_RIGHT = 9


@dataclass
class HomeostaticReward:
    """Drive-reduction + sparse guardrails. Plug into HomeostaticBotEnv as reward_fn."""

    # Setpoint: 80% — Keramati & Gutkin (2014) literature default.
    # Also matches IEC 61960 EOL threshold (70-80%) and gives 30% buffer above
    # the ~50% danger zone where power fade seriously impacts mission range.
    setpoint: float = 80.0

    # Sparse bonuses — kept small so the homeostatic drive stays dominant.
    goal_bonus: float = 10.0           # Yoshida et al. (2024), Bouhamed et al. (2020)
    death_penalty: float = 10.0        # Ramezani & Atashgah (2024)

    # Per-step terms.
    step_cost: float = 0.1             # Henderson et al. (2018), Imanberdiyev et al. (2016)
    collision_penalty: float = 1.0     # Raj & Kos (2024), tuned from -5 → -1 to match
                                       # goal magnitude per second at 10 Hz control rate.
    collision_threshold: float = 0.25  # ~11 cm clearance from Waffle's ~14 cm body.

    def __call__(self, prev_obs, action, next_obs, info):
        """Compute reward for one (s, a, s') transition."""

        # 1. Homeostatic drive-reduction — dominant signal.
        # Positive when SOC moves toward setpoint, negative when away.
        drive_before = abs(prev_obs[_IDX_SOC] - self.setpoint)
        drive_after = abs(next_obs[_IDX_SOC] - self.setpoint)
        reward = drive_before - drive_after

        # 2. Step cost (every step).
        reward -= self.step_cost

        # 3. Collision penalty — per-step while in danger zone.
        # Per-step (not edge-triggered) matches Raj & Kos dense formulation.
        lidar_min = min(
            next_obs[_IDX_LIDAR_FRONT],
            next_obs[_IDX_LIDAR_LEFT],
            next_obs[_IDX_LIDAR_RIGHT],
        )
        if lidar_min < self.collision_threshold:
            reward -= self.collision_penalty

        # 4. Sparse terminal bonuses — guardrails, not the main signal.
        if info.get("reached_goal", False):
            reward += self.goal_bonus
        if info.get("battery_dead", False):
            reward -= self.death_penalty

        return float(reward)


def make_default_reward() -> HomeostaticReward:
    """Default reward for the Phase 4 flat TD3 baseline."""
    return HomeostaticReward()
