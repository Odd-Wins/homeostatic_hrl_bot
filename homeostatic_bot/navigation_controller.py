"""Low-level proportional navigation controller for HRL options."""

from dataclasses import dataclass

import numpy as np

# Observation indices - keep in sync with env_wrapper._compute_flat_obs()
_IDX_X = 0
_IDX_Y = 1
_IDX_YAW = 2
_IDX_LIDAR_FRONT = 7
_IDX_LIDAR_LEFT = 8
_IDX_LIDAR_RIGHT = 9


@dataclass
class NavigationController:
    #Proportional controller with reactive obstacle avoidance and stop-at-target

    linear_vel: float = 0.15
    angular_gain: float = 0.6
    heading_deadband: float = 0.1
    max_angular: float = 1.82
    avoid_threshold: float = 0.4
    avoid_angular_vel: float = 1.0
    stop_radius: float = 0.3

    def __call__(self, obs: np.ndarray, target: np.ndarray) -> np.ndarray:
        #Compute velocity command to navigate toward target.
        x, y = obs[_IDX_X], obs[_IDX_Y]

        # 1. Stop at target if within radius.
        dist = np.hypot(target[0] - x, target[1] - y)
        if dist < self.stop_radius:
            return np.array([0.0, 0.0], dtype=np.float32)

        # 2. Reactive obstacle avoidance overrides target-seeking.
        front = obs[_IDX_LIDAR_FRONT]
        left = obs[_IDX_LIDAR_LEFT]
        right = obs[_IDX_LIDAR_RIGHT]

        if front < self.avoid_threshold:
            angular = self.avoid_angular_vel if left > right else -self.avoid_angular_vel
            return np.array([0.05, angular], dtype=np.float32)
        if left < self.avoid_threshold:
            return np.array([0.05, -self.avoid_angular_vel], dtype=np.float32)
        if right < self.avoid_threshold:
            return np.array([0.05, self.avoid_angular_vel], dtype=np.float32)

        # 3. Proportional control toward target.
        yaw = obs[_IDX_YAW]
        dx = target[0] - x
        dy = target[1] - y
        target_heading = np.arctan2(dy, dx)
        heading_error = np.arctan2(
            np.sin(target_heading - yaw),
            np.cos(target_heading - yaw),
        )

        if abs(heading_error) < self.heading_deadband:
            angular = 0.0
        else:
            angular = float(np.clip(
                self.angular_gain * heading_error,
                -self.max_angular,
                self.max_angular,
            ))

        linear = self.linear_vel
        if abs(heading_error) > np.pi / 4:
            linear *= 0.3

        return np.array([linear, angular], dtype=np.float32)
