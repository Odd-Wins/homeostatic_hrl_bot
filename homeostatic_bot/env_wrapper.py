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
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import LaserScan


# Type alias: reward_fn(prev_obs, action, next_obs, info) -> float
RewardFn = Callable[[np.ndarray, np.ndarray, np.ndarray, dict], float]


class HomeostaticBotEnv(gym.Env):
    """TurtleBot3 Waffle in energy_world — 12-D state, 2-D continuous action, 10 Hz.

    Operates in two modes:
      goal_conditioned=False (default): 12-D Box observation, used by threshold baseline
        and standalone evaluation. Reward injected via reward_fn.
      goal_conditioned=True: Dict observation (observation, achieved_goal, desired_goal)
        compatible with SB3's HerReplayBuffer. Reward computed internally with
        compute_reward() so HER can recompute on goal relabeling. reward_fn ignored.
    """

    metadata = {"render_modes": []}

    # Arena / task geometry
    ARENA_HALF = 4.5
    CHARGER_POS = np.array([4.0, 4.0], dtype=np.float32)
    CHARGER_RADIUS = 0.5
    GOAL_REACHED_RADIUS = 0.3
    # Option C: when set to a list of (x, y) tuples, reset() picks goals uniformly
    # from this fixed set instead of random-sampling-with-rejection. None = methodology
    # default (random sampling). Override per-instance from train_flat.py only.
    # See Phase4_Implementation_Manual_Supplement2.md § 5.
    GOAL_SET = None
    # Multi-goal: number of delivery goals per episode. When > 1, the env
    # generates a queue of goals and advances to the next on each completion.
    # Episode terminates when all goals are reached or battery dies / time limit.
    NUM_GOALS = 1

    # Robot kinematics (TurtleBot3 Waffle spec)
    MAX_LINEAR_VEL = 0.26
    MAX_ANGULAR_VEL = 1.82

    # Control / episode
    CONTROL_PERIOD = 0.1
    MAX_EPISODE_STEPS = 1200

    # Battery defaults (override per-instance for policy eval / training)
    INIT_SOC = 100.0
    INIT_SOH = 100.0
    DRAIN_RATE_MOVING = 5.0
    DRAIN_RATE_IDLE = 0.01
    CHARGE_RATE = 10.0
    HUANG_ALPHA = 0.05
    HUANG_BETA = 1.2

    # Built-in reward weights (used in goal_conditioned mode for compute_reward()).
    # These match HomeostaticReward defaults and stay in sync deliberately.
    SETPOINT = 80.0
    GOAL_BONUS = 10.0
    DEATH_PENALTY = 10.0
    STEP_COST = 0.1
    COLLISION_PENALTY = 1.0
    COLLISION_THRESHOLD = 0.25

    # Gazebo / ROS
    WORLD_NAME = "energy_world"
    ROBOT_NAME = "waffle"
    CLOCK_TIMEOUT_S = 3.0

    def __init__(
        self,
        reward_fn: Optional[RewardFn] = None,
        seed: Optional[int] = None,
        robot_name: Optional[str] = None,
        world_name: Optional[str] = None,
        goal_conditioned: bool = False,
        
    ):
        super().__init__()

        self.goal_conditioned = goal_conditioned
        self._reward_fn: RewardFn = reward_fn or (lambda p, a, n, i: 0.0)
        if robot_name is not None:
            self.ROBOT_NAME = robot_name
        if world_name is not None:
            self.WORLD_NAME = world_name

        # Define observation and action spaces depending on mode.
        flat_low = np.array(
            [-5, -5, -math.pi, -1.0, -2.0, 0, 0, 0, 0, 0, 0, 0],
            dtype=np.float32,
        )
        flat_high = np.array(
            [5, 5, math.pi, 1.0, 2.0, 100, 100, 10, 10, 10, 15, 15],
            dtype=np.float32,
        )

        if self.goal_conditioned:
            # Dict obs format expected by SB3 HER:
            #   observation: the 12-D vector
            #   achieved_goal: current robot position (x, y)
            #   desired_goal: target goal position (x, y)
            goal_low = np.array([-5, -5], dtype=np.float32)
            goal_high = np.array([5, 5], dtype=np.float32)
            self.observation_space = spaces.Dict({
                "observation": spaces.Box(flat_low, flat_high, dtype=np.float32),
                "achieved_goal": spaces.Box(goal_low, goal_high, dtype=np.float32),
                "desired_goal": spaces.Box(goal_low, goal_high, dtype=np.float32),
            })
        else:
            self.observation_space = spaces.Box(flat_low, flat_high, dtype=np.float32)

        self.action_space = spaces.Box(
            low=np.array([-self.MAX_LINEAR_VEL, -self.MAX_ANGULAR_VEL], dtype=np.float32),
            high=np.array([self.MAX_LINEAR_VEL, self.MAX_ANGULAR_VEL], dtype=np.float32),
            dtype=np.float32,
        )

        # ROS2 setup
        if not rclpy.ok():
            rclpy.init()
        self._node = Node("homeostatic_env")
        self._cmd_pub = self._node.create_publisher(TwistStamped, "/cmd_vel", 10)

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self._node.create_subscription(Odometry, "/odom", self._odom_cb, sensor_qos)
        self._node.create_subscription(LaserScan, "/scan", self._scan_cb, sensor_qos)
        self._node.create_subscription(Clock, "/clock", self._clock_cb, sensor_qos)

        self._latest_pose = None
        self._latest_twist = None
        self._latest_scan: Optional[LaserScan] = None
        self._latest_sim_time: Optional[float] = None

        self._executor = MultiThreadedExecutor()
        self._executor.add_node(self._node)
        self._spin_thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._spin_thread.start()

        # Internal state
        self._soc = self.INIT_SOC
        self._soh = self.INIT_SOH
        self._charge_cycles = 0
        self._was_charging = False
        self._goal = np.zeros(2, dtype=np.float32)
        self._step_count = 0
        self._prev_obs: Optional[np.ndarray] = None
        self._prev_soc: float = self.INIT_SOC

        self._rng = np.random.default_rng(seed)
        self._use_sim_time: Optional[bool] = None

    # ----- ROS callbacks -------------------------------------------------------
    def _odom_cb(self, msg: Odometry) -> None:
        self._latest_pose = msg.pose.pose
        self._latest_twist = msg.twist.twist

    def _scan_cb(self, msg: LaserScan) -> None:
        self._latest_scan = msg

    def _clock_cb(self, msg: Clock) -> None:
        self._latest_sim_time = msg.clock.sec + msg.clock.nanosec * 1e-9

    def _wait_for_sensors(self, timeout: float = 5.0) -> None:
        start = time.time()
        while self._latest_pose is None or self._latest_scan is None:
            if time.time() - start > timeout:
                raise RuntimeError(
                    "Timed out waiting for /odom and /scan. Is Gazebo running "
                    "and is the TurtleBot3 spawned? Check `ros2 topic list`."
                )
            time.sleep(0.05)

    def _decide_clock_source(self) -> None:
        if self._use_sim_time is not None:
            return
        start = time.time()
        while self._latest_sim_time is None and (time.time() - start) < self.CLOCK_TIMEOUT_S:
            time.sleep(0.05)
        self._use_sim_time = self._latest_sim_time is not None
        if self._use_sim_time:
            self._node.get_logger().info(
                "Using /clock for sim-time sync. Gazebo can run faster than real-time."
            )
        else:
            self._node.get_logger().warn(
                "/clock not publishing — falling back to wall-clock sleep."
            )

    def _wait_one_period(self) -> None:
        if self._use_sim_time:
            target_time = self._latest_sim_time + self.CONTROL_PERIOD
            wall_start = time.time()
            while self._latest_sim_time < target_time:
                if time.time() - wall_start > 5.0:
                    self._node.get_logger().warn(
                        "Sim-time not advancing — falling back to sleep."
                    )
                    time.sleep(self.CONTROL_PERIOD)
                    return
                time.sleep(0.001)
        else:
            time.sleep(self.CONTROL_PERIOD)

    # ----- Gazebo teleport ----------------------------------------------------
    def _teleport_robot(self, x: float, y: float, yaw: float) -> bool:
        qz = math.sin(yaw / 2.0)
        qw = math.cos(yaw / 2.0)
        req = (
            f'name: "{self.ROBOT_NAME}", '
            f"position: {{x: {x}, y: {y}, z: 0.01}}, "
            f"orientation: {{x: 0, y: 0, z: {qz}, w: {qw}}}"
        )
        cmd = [
            "gz", "service",
            "-s", f"/world/{self.WORLD_NAME}/set_pose/blocking",
            "--reqtype", "gz.msgs.Pose",
            "--reptype", "gz.msgs.Boolean",
            "--timeout", "2000",
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

        # 1. Stop the robot before teleporting.
        stop_cmd = TwistStamped()
        for _ in range(5):
            self._cmd_pub.publish(stop_cmd)
            time.sleep(0.05)
        time.sleep(0.2)

        # 2. Teleport to origin with retry.
        #    Note: /odom reports integrated odometry, NOT true Gazebo pose. After
        #    set_pose, /odom may lag or report stale values. We trust the service
        #    response and do NOT verify via _latest_pose — that caused false-negative
        #    retry loops in previous versions.
        start_yaw = float(self._rng.uniform(-math.pi, math.pi))
        ok = False
        for attempt in range(3):
            ok = self._teleport_robot(0.0, 0.0, start_yaw)
            if ok:
                break
            self._node.get_logger().warn(
                f"Teleport attempt {attempt + 1}/3 failed. Retrying..."
            )
            time.sleep(0.5)
        if not ok:
            self._node.get_logger().error(
                "All teleport attempts failed. Check ROBOT_NAME matches Gazebo model "
                f"(current: '{self.ROBOT_NAME}'). Run `gz model --list` to verify."
            )
        time.sleep(0.3)

        # Goal selection — generate NUM_GOALS delivery targets.
        self._goal_queue: list[np.ndarray] = []
        for _ in range(self.NUM_GOALS):
            if self.GOAL_SET is not None:
                idx = int(self._rng.integers(0, len(self.GOAL_SET)))
                gx, gy = float(self.GOAL_SET[idx][0]), float(self.GOAL_SET[idx][1])
            else:
                for _ in range(50):
                    gx = float(self._rng.uniform(-self.ARENA_HALF, self.ARENA_HALF))
                    gy = float(self._rng.uniform(-self.ARENA_HALF, self.ARENA_HALF))
                    if (
                        np.linalg.norm([gx, gy]) > 1.5
                        and np.linalg.norm([gx - self.CHARGER_POS[0], gy - self.CHARGER_POS[1]]) > 1.0
                    ):
                        break
            self._goal_queue.append(np.array([gx, gy], dtype=np.float32))
        self._goals_completed = 0
        self._goal = self._goal_queue[0].copy()

        self._soc = self.INIT_SOC
        self._soh = float(options["initial_soh"]) if options and "initial_soh" in options else self.INIT_SOH
        self._charge_cycles = 0
        self._was_charging = False
        self._step_count = 0
        self._prev_soc = self._soc

        self._wait_for_sensors()
        self._decide_clock_source()

        flat_obs = self._compute_flat_obs()
        self._prev_obs = flat_obs.copy()

        info = {
            "goal": self._goal.copy(),
            "goal_queue": [g.copy() for g in self._goal_queue],
            "goals_total": self.NUM_GOALS,
            "initial_soh": self._soh,
            "use_sim_time": self._use_sim_time,
        }

        if self.goal_conditioned:
            return self._wrap_goal_dict(flat_obs), info
        return flat_obs, info

    def step(self, action: np.ndarray):
        # 1. Publish action
        twist = TwistStamped()
        twist.header.stamp = self._node.get_clock().now().to_msg()
        twist.header.frame_id = "base_link"
        twist.twist.linear.x = float(np.clip(action[0], -self.MAX_LINEAR_VEL, self.MAX_LINEAR_VEL))
        twist.twist.angular.z = float(np.clip(action[1], -self.MAX_ANGULAR_VEL, self.MAX_ANGULAR_VEL))
        self._cmd_pub.publish(twist)

        # 2. Sim-time wait
        self._wait_one_period()

        # 3. Battery update — capture prev SOC for drive computation
        prev_soc = self._soc
        self._tick_battery(twist.twist.linear.x, twist.twist.angular.z)

        self._step_count += 1

        # 4. Observe + termination
        flat_obs = self._compute_flat_obs()
        pos = np.array([self._latest_pose.position.x, self._latest_pose.position.y], dtype=np.float32)
        dist_to_goal = float(np.linalg.norm(pos - self._goal))
        reached_current_goal = dist_to_goal < self.GOAL_REACHED_RADIUS
        battery_dead = self._soc <= 0.0
        time_limit = self._step_count >= self.MAX_EPISODE_STEPS

        # Multi-goal: advance to next goal when current one is reached.
        reached_goal = False
        if reached_current_goal:
            self._goals_completed += 1
            if self._goals_completed >= self.NUM_GOALS:
                reached_goal = True  # all deliveries done — episode success
            else:
                # Advance to next goal and update obs with new distance.
                self._goal = self._goal_queue[self._goals_completed].copy()
                flat_obs = self._compute_flat_obs()

        terminated = bool(reached_goal or battery_dead)
        truncated = bool(time_limit and not terminated)

        # Lidar minimum (used both for collision penalty and stored in info for HER)
        lidar_min = float(min(flat_obs[7], flat_obs[8], flat_obs[9]))

        # 5. Reward — different path depending on mode
        if self.goal_conditioned:
            # Compute the homeostatic reward internally so HER can recompute
            # consistent rewards on relabeled transitions.
            reward = float(self._compute_full_reward(
                prev_soc=prev_soc,
                next_soc=self._soc,
                lidar_min=lidar_min,
                achieved_pos=pos,
                desired_goal=self._goal,
                battery_dead=battery_dead,
            ))
            # Components needed for HER's compute_reward to recompute on relabeling.
            non_goal_reward = reward - (self.GOAL_BONUS if reached_goal else 0.0)
        else:
            reward = float(self._reward_fn(self._prev_obs, np.asarray(action), flat_obs,
                                           {"reached_goal": reached_current_goal,
                                            "battery_dead": battery_dead}))
            non_goal_reward = 0.0  # unused outside goal_conditioned mode

        info = {
            "soc": self._soc,
            "soh": self._soh,
            "dist_to_goal": float(np.linalg.norm(pos - self._goal)),
            "dist_to_charger": float(np.linalg.norm(pos - self.CHARGER_POS)),
            "reached_goal": reached_goal,
            "reached_current_goal": reached_current_goal,
            "battery_dead": battery_dead,
            "charge_cycles": self._charge_cycles,
            "step_count": self._step_count,
            "goals_completed": self._goals_completed,
            "goals_total": self.NUM_GOALS,
            # Fields HER's compute_reward needs to reconstruct the reward on relabel:
            "non_goal_reward": non_goal_reward,
            "lidar_min": lidar_min,
            "prev_soc": prev_soc,
            "next_soc": self._soc,
        }

        self._prev_obs = flat_obs.copy()

        if self.goal_conditioned:
            return self._wrap_goal_dict(flat_obs), reward, terminated, truncated, info
        return flat_obs, reward, terminated, truncated, info

    # ----- Reward computation (used in goal_conditioned mode) -----------------
    def _compute_full_reward(
        self,
        prev_soc: float,
        next_soc: float,
        lidar_min: float,
        achieved_pos: np.ndarray,
        desired_goal: np.ndarray,
        battery_dead: bool,
    ) -> float:
        """Full homeostatic reward — used internally in goal_conditioned mode."""
        drive_before = abs(prev_soc - self.SETPOINT)
        drive_after = abs(next_soc - self.SETPOINT)
        reward = drive_before - drive_after - self.STEP_COST
        if lidar_min < self.COLLISION_THRESHOLD:
            reward -= self.COLLISION_PENALTY
        if float(np.linalg.norm(achieved_pos - desired_goal)) < self.GOAL_REACHED_RADIUS:
            reward += self.GOAL_BONUS
        if battery_dead:
            reward -= self.DEATH_PENALTY
        return reward

    def compute_reward(self, achieved_goal, desired_goal, info):
        """Recompute reward for HER-relabeled transitions.

        SB3's HerReplayBuffer calls this with batched arrays and a list of info
        dicts. For each transition, the non-goal-bonus part of the reward
        (drive change, step cost, collision, death) is preserved from the
        original transition (via info["non_goal_reward"]). The goal bonus is
        recomputed from the relabeled desired_goal.
        """
        achieved = np.asarray(achieved_goal, dtype=np.float32).reshape(-1, 2)
        desired = np.asarray(desired_goal, dtype=np.float32).reshape(-1, 2)
        distance = np.linalg.norm(achieved - desired, axis=-1)
        new_goal_bonus = self.GOAL_BONUS * (distance < self.GOAL_REACHED_RADIUS).astype(np.float32)

        if isinstance(info, dict):
            info_list = [info]
        else:
            info_list = list(info)
        non_goal = np.array(
            [d.get("non_goal_reward", 0.0) for d in info_list], dtype=np.float32
        )

        rewards = non_goal + new_goal_bonus
        if rewards.size == 1 and not isinstance(info, list):
            return float(rewards[0])
        return rewards

    # ----- Battery dynamics ---------------------------------------------------
    def _tick_battery(self, lin_vel: float, ang_vel: float) -> None:
        pos = np.array([self._latest_pose.position.x, self._latest_pose.position.y])
        near_charger = float(np.linalg.norm(pos - self.CHARGER_POS)) < self.CHARGER_RADIUS
        dt = self.CONTROL_PERIOD

        if near_charger:
            self._soc = min(self.INIT_SOC, self._soc + self.CHARGE_RATE * dt)
            if not self._was_charging:
                self._charge_cycles += 1
                predicted_soh = (
                    1.0 - self.HUANG_ALPHA * (self._charge_cycles ** self.HUANG_BETA)
                ) * 100.0
                self._soh = max(0.0, min(self._soh, predicted_soh))
            self._was_charging = True
        else:
            moving = abs(lin_vel) > 0.01 or abs(ang_vel) > 0.01
            base_rate = self.DRAIN_RATE_MOVING if moving else self.DRAIN_RATE_IDLE
            effective_rate = base_rate * (100.0 / self._soh) if self._soh > 1e-6 else base_rate
            self._soc = max(0.0, self._soc - effective_rate * dt)
            self._was_charging = False

    # ----- Observation construction -------------------------------------------
    def _compute_flat_obs(self) -> np.ndarray:
        pose = self._latest_pose
        twist = self._latest_twist
        scan = self._latest_scan

        x = pose.position.x
        y = pose.position.y
        q = pose.orientation
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

    def _wrap_goal_dict(self, flat_obs: np.ndarray) -> dict:
        """Convert flat 12-D obs to HER-compatible Dict observation."""
        return {
            "observation": flat_obs,
            "achieved_goal": np.array([flat_obs[0], flat_obs[1]], dtype=np.float32),
            "desired_goal": self._goal.copy().astype(np.float32),
        }

    @staticmethod
    def _sector_mins(scan: LaserScan) -> tuple[float, float, float]:
        """Min range per sector: front ±30°, left 30..150°, right -30..-150°."""
        ranges = np.array(scan.ranges, dtype=np.float32)
        ranges[~np.isfinite(ranges)] = scan.range_max
        ranges = np.clip(ranges, scan.range_min, scan.range_max)

        n = len(ranges)
        angles = scan.angle_min + np.arange(n) * scan.angle_increment
        angles = ((angles + math.pi) % (2 * math.pi)) - math.pi

        front_mask = (angles >= -math.pi / 6) & (angles <= math.pi / 6)
        left_mask = (angles > math.pi / 6) & (angles <= 5 * math.pi / 6)
        right_mask = (angles < -math.pi / 6) & (angles >= -5 * math.pi / 6)

        def _min(mask):
            return float(ranges[mask].min()) if mask.any() else float(scan.range_max)

        return _min(front_mask), _min(left_mask), _min(right_mask)

    def close(self) -> None:
        try:
            self._cmd_pub.publish(TwistStamped())
        except Exception:
            pass
        try:
            self._executor.shutdown()
            self._node.destroy_node()
        except Exception:
            pass
