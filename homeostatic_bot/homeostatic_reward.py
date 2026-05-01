"""
homeostatic_reward.py — Drive-reduction reward for the homeostatic HRL thesis.

Install target: ~/ros2_ws/src/homeostatic_bot/homeostatic_bot/homeostatic_reward.py

Implements the Keramati & Gutkin (2014) homeostatic reinforcement learning
framework as the *dominant* learning signal, with sparse task bonuses and
guardrail penalties.

DESIGN PRINCIPLE
    The agent receives reward each step based on how its action moved the
    battery state of charge (SOC) toward or away from a homeostatic setpoint.
    Drive is defined as |SOC - setpoint|; reward is the negative change in
    drive across one step. When SOC moves toward setpoint, drive decreases
    and reward is positive. This produces energy-aware behavior without
    engineered "go charge when below threshold X" rules.

REWARD COMPONENTS
    Homeostatic drive:  drive_before - drive_after          Keramati & Gutkin (2014)
    Goal reached:       +10                                 Yoshida et al. (2024),
                                                            Bouhamed et al. (2020)
    Battery death:      -10                                 Ramezani & Atashgah (2024)
    Step cost:          -0.1                                Henderson et al. (2018),
                                                            Imanberdiyev et al. (2016)
    Collision penalty:  -5  (lidar sector min < 0.25 m)     Raj & Kos (2024)

SETPOINT CHOICE: 80% SOC
    Three justifications for the defense:
      1. Keramati & Gutkin (2014) literature default — direct framework match.
      2. IEC 61960 end-of-life threshold (70-80% capacity) — physical meaning.
      3. Hospital-robot framing — 30% buffer above the ~50% danger zone where
         power-fade multiplier seriously impacts mission range.

IMPORTANT — DOMINANCE OF DRIVE
    The homeostatic drive is intended to be the *dominant* learning signal.
    Sparse bonuses are guardrails. If during training the agent ignores the
    drive and chases the +10 goal bonus, REDUCE the sparse rewards rather
    than scaling up the drive — the thesis claim depends on drive-reduction
    generating energy-aware behavior without engineered reward shaping.

OBSERVATION INDICES (synced with env_wrapper.HomeostaticBotEnv._compute_obs)
    [0,1,2]   x, y, yaw
    [3,4]     linear_vel, angular_vel
    [5]       SOC
    [6]       SOH
    [7,8,9]   lidar_front, lidar_left, lidar_right
    [10,11]   dist_to_goal, dist_to_charger
"""

from dataclasses import dataclass

import numpy as np


# Observation indices — keep in sync with env_wrapper._compute_obs()
_IDX_SOC = 5
_IDX_LIDAR_FRONT = 7
_IDX_LIDAR_LEFT = 8
_IDX_LIDAR_RIGHT = 9


@dataclass
class HomeostaticReward:
    """Keramati & Gutkin (2014) drive-reduction reward with sparse guardrails.

    Implemented as a dataclass so weights can be reconfigured without
    rewriting the function — useful for Phase 6 ablation studies (e.g.,
    disable collision penalty by setting collision_penalty=0 to isolate
    its contribution to learned behavior).

    Attributes:
        setpoint: Homeostatic SOC target (%).
        goal_bonus: Reward added when info["reached_goal"] is True.
        death_penalty: Reward subtracted when info["battery_dead"] is True.
        step_cost: Reward subtracted every step (encourages efficiency).
        collision_penalty: Reward subtracted while any lidar sector min is
            below collision_threshold.
        collision_threshold: Distance (m) below which a sector counts as a
            near-collision. 0.25 m gives ~11 cm clearance from the Waffle's
            ~14 cm body radius.
    """

    setpoint: float = 80.0

    goal_bonus: float = 10.0
    death_penalty: float = 10.0

    step_cost: float = 0.1
    collision_penalty: float = 1.0
    collision_threshold: float = 0.25

    def __call__(
        self,
        prev_obs: np.ndarray,
        action: np.ndarray,
        next_obs: np.ndarray,
        info: dict,
    ) -> float:
        """Compute reward for one (s, a, s') transition.

        Conforms to env_wrapper.RewardFn signature so it can be passed
        directly: HomeostaticBotEnv(reward_fn=HomeostaticReward()).
        """

        # 1. Homeostatic drive-reduction — the dominant signal.
        #    Positive when SOC moves toward setpoint, negative when away.
        drive_before = abs(prev_obs[_IDX_SOC] - self.setpoint)
        drive_after = abs(next_obs[_IDX_SOC] - self.setpoint)
        reward = drive_before - drive_after

        # 2. Step cost — applied every step, including terminal steps.
        #    At max episode length (1200 steps): -120 cumulative.
        reward -= self.step_cost

        # 3. Collision penalty — per-step while in the danger zone.
        #    Per-step (not edge-triggered) matches Raj & Kos (2024) dense
        #    formulation. If agent becomes overcautious in training, switch
        #    to edge-triggered or lower magnitude.
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
    """Construct the default reward for Phase 4 flat TD3 baseline."""
    return HomeostaticReward()
