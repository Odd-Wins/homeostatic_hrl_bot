"""Gymnasium environment wrapper for the homeostatic HRL thesis (ROS2 + Gazebo + TurtleBot3)."""

import math
import subprocess
import threading
import time
from typing import Any, Callable, Optional

import gymnasium as gym
import numpy as np
import rclpy
from gymnasium import spaces
from geometry_msgs.msg import Twist, TwistStamped
from nav_msgs.msg import Odometry
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import LaserScan


# Type alias: reward_fn(prev_obs, action, next_obs, info) -> float
RewardFn = Callable[[np.ndarray, np.ndarray, np.ndarray, dict], float]


class HomeostaticBotEnv(gym.Env):
    """TurtleBot3 Waffle in energy_world — 12-D state, 2-D continuous action, 10 Hz."""

    metadata = {"render_modes": []}

    # Arena / task geometry
    ARENA_HALF = 4.5                           # walls at ±5, keep 0.5 m buffer
    CHARGER_POS = np.array([4.0, 4.0], dtype=np.float32)
    CHARGER_RADIUS = 0.5                       # within this distance → charging
    GOAL_REACHED_RADIUS = 0.3                  # within this distance → success

    # Robot kinematics (TurtleBot3 Waffle spec)
    MAX_LINEAR_VEL = 0.26
    MAX_ANGULAR_VEL = 1.82

    # Control / episode
    CONTROL_PERIOD = 0.1                       # 10 Hz
    MAX_EPISODE_STEPS = 1200                   # 120 s @ 10 Hz

    # Battery constants (ported from battery_node.py — Phase 3)
    INIT_SOC = 100.0
    INIT_SOH = 100.0
    DRAIN_RATE_MOVING = 5.0                    # %/s while moving
    DRAIN_RATE_IDLE = 0.01                     # %/s while idle
    CHARGE_RATE = 10.0                         # %/s while on charger
    HUANG_ALPHA = 0.05                         # SOH(n) = 1 - α · n^β
    HUANG_BETA = 1.2

    # Gazebo / ROS
    WORLD_NAME = "energy_world"
    ROBOT_NAME = "turtlebot3_waffle"

    def __init__(
        self,
        reward_fn: Optional[RewardFn] = None,
        seed: Optional[int] = None,
        robot_name: Optional[str] = None,
        world_name: Optional[str] = None,
    ):
        super().__init__()

        self._reward_fn: RewardFn = reward_fn or (lambda p, a, n, i: 0.0)
        if robot_name is not None:
            self.ROBOT_NAME = robot_name
        if world_name is not None:
            self.WORLD_NAME = world_name

        # Observation bounds — generous; SB3 + VecNormalize don't need tight.
        obs_low = np.array(
            [-5, -5, -math.pi, -1.0, -2.0, 0, 0, 0, 0, 0, 0, 0],
            dtype=np.float32,
        )
        obs_high = np.array(
            [5, 5, math.pi, 1.0, 2.0, 100, 100, 10, 10, 10, 15, 15],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)

        self.action_space = spaces.Box(
            low=np.array([-self.MAX_LINEAR_VEL, -self.MAX_ANGULAR_VEL], dtype=np.float32),
            high=np.array([self.MAX_LINEAR_VEL, self.MAX_ANGULAR_VEL], dtype=np.float32),
            dtype=np.float32,
        )

        # ROS2 setup
        if not rclpy.ok():
            rclpy.init()
        self._node = Node("homeostatic_env")
        # /cmd_vel uses TwistStamped on Gazebo Harmonic — Twist gets silently dropped.
        self._cmd_pub = self._node.create_publisher(TwistStamped, "/cmd_vel", 10)
        self._node.create_subscription(Odometry, "/odom", self._odom_cb, 10)
        self._node.create_subscription(LaserScan, "/scan", self._scan_cb, 10)

        # Latest sensor state (populated by callbacks in the spin thread)
        self._latest_pose = None
        self._latest_twist = None
        self._latest_scan: Optional[LaserScan] = None

        self._executor = MultiThreadedExecutor()
        self._executor.add_node(self._node)
        self._spin_thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._spin_thread.start()

        # Internal state
        self._soc = self.INIT_SOC
        self._soh = self.INIT_SOH
        self._charge_cycles = 0
        self._was_charging = False                # edge detection for cycle count
        self._goal = np.zeros(2, dtype=np.float32)
        self._step_count = 0
        self._prev_obs: Optional[np.ndarray] = None

        self._rng = np.random.default_rng(seed)

    # ----- ROS callbacks -------------------------------------------------------
    def _odom_cb(self, msg: Odometry) -> None:
        self._latest_pose = msg.pose.pose
        self._latest_twist = msg.twist.twist

    def _scan_cb(self, msg: LaserScan) -> None:
        self._latest_scan = msg

    def _wait_for_sensors(self, timeout: float = 5.0) -> None:
        """Block until /odom and /scan have each produced at least one message."""
        start = time.time()
        while self._latest_pose is None or self._latest_scan is None:
            if time.time() - start > timeout:
                raise RuntimeError(
                    "Timed out waiting for /odom and /scan. Is Gazebo running "
                    "and is the TurtleBot3 spawned? Check `ros2 topic list`."
                )
            time.sleep(0.05)

    # ----- Gazebo teleport (Harmonic workaround — no native model reset) ------
    def _teleport_robot(self, x: float, y: float, yaw: float) -> bool:
        """Call `gz service` to teleport robot. Silent failures usually mean ROBOT_NAME mismatch."""
        qz = math.sin(yaw / 2.0)
        qw = math.cos(yaw / 2.0)
        req = (
            f'name: "{self.ROBOT_NAME}", '
            f"position: {{x: {x}, y: {y}, z: 0.01}}, "
            f"orientation: {{x: 0, y: 0, z: {qz}, w: {qw}}}"
        )
        cmd = [
            "gz", "service",
            "-s", f"/world/{self.WORLD_NAME}/set_pose",
            "--reqtype", "gz.msgs.Pose",
            "--reptype", "gz.msgs.Boolean",
            "--timeout", "1000",
            "--req", req,
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=2.0, text=True)
            return "data: true" in result.stdout
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    # ----- Gym API -------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Stop the robot first so it doesn't keep moving through the teleport.
        self._cmd_pub.publish(TwistStamped())

        # Teleport back to origin with random yaw.
        start_yaw = float(self._rng.uniform(-math.pi, math.pi))
        ok = self._teleport_robot(0.0, 0.0, start_yaw)
        if not ok:
            self._node.get_logger().warn(
                "Teleport call returned non-success. Check ROBOT_NAME matches Gazebo model."
            )
        time.sleep(0.2)  # let physics settle after teleport

        # Sample goal inside arena, away from start and charger.
        for _ in range(50):
            gx = float(self._rng.uniform(-self.ARENA_HALF, self.ARENA_HALF))
            gy = float(self._rng.uniform(-self.ARENA_HALF, self.ARENA_HALF))
            if (
                np.linalg.norm([gx, gy]) > 1.5
                and np.linalg.norm([gx - self.CHARGER_POS[0], gy - self.CHARGER_POS[1]]) > 1.0
            ):
                break
        self._goal = np.array([gx, gy], dtype=np.float32)

        # Battery state. Training = 100% SOH by default; eval passes options.
        self._soc = self.INIT_SOC
        self._soh = float(options["initial_soh"]) if options and "initial_soh" in options else self.INIT_SOH
        self._charge_cycles = 0
        self._was_charging = False
        self._step_count = 0

        self._wait_for_sensors()
        obs = self._compute_obs()
        self._prev_obs = obs.copy()
        info = {"goal": self._goal.copy(), "initial_soh": self._soh}
        return obs, info

    def step(self, action: np.ndarray):
        # 1. Publish action
        twist = TwistStamped()
        twist.header.stamp = self._node.get_clock().now().to_msg()
        twist.header.frame_id = "base_link"
        twist.twist.linear.x = float(np.clip(action[0], -self.MAX_LINEAR_VEL, self.MAX_LINEAR_VEL))
        twist.twist.angular.z = float(np.clip(action[1], -self.MAX_ANGULAR_VEL, self.MAX_ANGULAR_VEL))
        self._cmd_pub.publish(twist)

        # 2. Wait one control period for physics + sensors to advance
        time.sleep(self.CONTROL_PERIOD)

        # 3. Update battery given the action we just took
        self._tick_battery(twist.twist.linear.x, twist.twist.angular.z)

        self._step_count += 1

        # 4. Observe + check termination
        obs = self._compute_obs()

        pos = np.array([self._latest_pose.position.x, self._latest_pose.position.y])
        dist_to_goal = float(np.linalg.norm(pos - self._goal))
        reached_goal = dist_to_goal < self.GOAL_REACHED_RADIUS
        battery_dead = self._soc <= 0.0
        time_limit = self._step_count >= self.MAX_EPISODE_STEPS

        terminated = bool(reached_goal or battery_dead)   # task-terminal
        truncated = bool(time_limit and not terminated)   # time-limit only

        info = {
            "soc": self._soc,
            "soh": self._soh,
            "dist_to_goal": dist_to_goal,
            "dist_to_charger": float(np.linalg.norm(pos - self.CHARGER_POS)),
            "reached_goal": reached_goal,
            "battery_dead": battery_dead,
            "charge_cycles": self._charge_cycles,
            "step_count": self._step_count,
        }

        # 5. Compute reward via injected function
        reward = float(self._reward_fn(self._prev_obs, np.asarray(action), obs, info))

        self._prev_obs = obs.copy()
        return obs, reward, terminated, truncated, info

    # ----- Battery dynamics (ported from battery_node.py) ---------------------
    def _tick_battery(self, lin_vel: float, ang_vel: float) -> None:
        pos = np.array([self._latest_pose.position.x, self._latest_pose.position.y])
        near_charger = float(np.linalg.norm(pos - self.CHARGER_POS)) < self.CHARGER_RADIUS
        dt = self.CONTROL_PERIOD

        if near_charger:
            # Charging: SOC always refills to 100% regardless of SOH.
            self._soc = min(self.INIT_SOC, self._soc + self.CHARGE_RATE * dt)
            # Edge-triggered cycle counter: one cycle per charger entry.
            if not self._was_charging:
                self._charge_cycles += 1
                # Huang et al. (2022) capacity fade.
                predicted_soh = (
                    1.0 - self.HUANG_ALPHA * (self._charge_cycles ** self.HUANG_BETA)
                ) * 100.0
                # min() ensures SOH only ratchets down — never un-degrades during eval.
                self._soh = max(0.0, min(self._soh, predicted_soh))
            self._was_charging = True
        else:
            # Draining
            moving = abs(lin_vel) > 0.01 or abs(ang_vel) > 0.01
            base_rate = self.DRAIN_RATE_MOVING if moving else self.DRAIN_RATE_IDLE
            # Power fade: drain × 100/SOH (Cui et al. 2023, Maures et al. 2020).
            # Verified Phase 3: SOH=39.4% → 2.54× drain matched prediction.
            effective_rate = base_rate * (100.0 / self._soh) if self._soh > 1e-6 else base_rate
            self._soc = max(0.0, self._soc - effective_rate * dt)
            self._was_charging = False

    # ----- Observation construction -------------------------------------------
    def _compute_obs(self) -> np.ndarray:
        pose = self._latest_pose
        twist = self._latest_twist
        scan = self._latest_scan

        x = pose.position.x
        y = pose.position.y
        q = pose.orientation
        # Quaternion → yaw (rotation around z).
        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y ** 2 + q.z ** 2),
        )

        lin_vel = twist.linear.x if twist is not None else 0.0
        ang_vel = twist.angular.z if twist is not None else 0.0

        front, left, right = self._sector_mins(scan)

        pos = np.array([x, y], dtype=np.float32)
        dist_goal = float(np.linalg.norm(pos - self._goal))
        dist_charger = float(np.linalg.norm(pos - self.CHARGER_POS))

        return np.array(
            [x, y, yaw, lin_vel, ang_vel,
             self._soc, self._soh,
             front, left, right,
             dist_goal, dist_charger],
            dtype=np.float32,
        )

    @staticmethod
    def _sector_mins(scan: LaserScan) -> tuple[float, float, float]:
        """Min range per sector: front ±30°, left 30..150°, right -30..-150°."""
        ranges = np.array(scan.ranges, dtype=np.float32)
        ranges[~np.isfinite(ranges)] = scan.range_max
        ranges = np.clip(ranges, scan.range_min, scan.range_max)

        n = len(ranges)
        angles = scan.angle_min + np.arange(n) * scan.angle_increment
        # Wrap to [-π, π]
        angles = ((angles + math.pi) % (2 * math.pi)) - math.pi

        front_mask = (angles >= -math.pi / 6) & (angles <= math.pi / 6)
        left_mask = (angles > math.pi / 6) & (angles <= 5 * math.pi / 6)
        right_mask = (angles < -math.pi / 6) & (angles >= -5 * math.pi / 6)

        def _min(mask):
            return float(ranges[mask].min()) if mask.any() else float(scan.range_max)

        return _min(front_mask), _min(left_mask), _min(right_mask)

    # ----- Cleanup -------------------------------------------------------------
    def close(self) -> None:
        try:
            self._cmd_pub.publish(TwistStamped())  # stop the robot
        except Exception:
            pass
        try:
            self._executor.shutdown()
            self._node.destroy_node()
        except Exception:
            pass
