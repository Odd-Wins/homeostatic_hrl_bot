# Development Log - Homeostatic HRL Thesis

## Week 1-2: ROS2 & Gazebo Setup
**Date:** Feb 24 - March 9

- Installed ROS2 Jazzy, Gazebo Harmonic
- Created homeostatic_bot package
- Built patrol_node (controls TurtleBot3)
- Created custom world with charging station
- Set up GitHub repository
- Used TwistStamped instead of Twist (Gazebo requirement)
- Created simple arena instead of complex world (controlled experiments)

---

## Week 3-4: Energy Simulation
**Date:** March 10 - March 23

- Created battery_node.py with SOC drain and charging at station
- Used ROS2 battery node instead of LinearBatteryPlugin for control over degradation model
- Updated battery_node for detection using AprilTag
- Added Huang et al. (2022) degradation model: SOH = 1 - alpha * n^beta

---

## Week 5-6: Docking & Visual Servoing
**Date:** March 24 - April 6

- Built docking_controller.py with AprilTag-based TF2 visual servoing
- Proportional control with dead zone and obstacle avoidance
- Publishes /charging/detected for battery_node integration

---

## Week 7-9: Environment Wrapper (Phase 3-4)
**Date:** April 7 - April 27

- Built env_wrapper.py (Gymnasium interface to ROS2 + Gazebo)
- 12-D observation: [x, y, yaw, lin_vel, ang_vel, SOC, SOH, lidar_front, lidar_left, lidar_right, dist_goal, dist_charger]
- 2-D continuous action: [linear_vel, angular_vel]
- Goal-conditioned mode for HER compatibility
- Sim-time synchronisation via /clock subscription (2.3x training speedup)

---

## Week 10-11: Flat RL Attempts (Phase 4)
**Date:** April 28 - May 12

- Flat TD3: 500k timesteps, 416 episodes — failed to learn goal-reaching
- Flat TD3 + HER: 6 runs with different configs - all failed
- Conclusion: flat RL insufficient for joint navigation + energy management

---

## Week 12-14: Hierarchical RL (Phase 5)
**Date:** May 13 - June 6

- Implemented Options framework: DQN meta-controller with 2 discrete actions (GOTO_GOAL, GOTO_CHARGER)
- Hand-coded NavigationController for low-level proportional control with obstacle avoidance
- Homeostatic drive-reduction reward (Keramati & Gutkin 2014)
- Stage 1: goal-reaching at SOH=100%
- Stage 2: SOH domain randomisation across {100, 80, 60, 40}% with SOC randomisation (40-100%)
- Action masking: GOTO_CHARGER unavailable when already charged at station
- Bridge watchdog: auto-restarts ros_gz_bridge on sim-time stall
- Ground truth pose subscription to replace drifting /odom after teleport
- Final model: 2026-06-06_11-15-39

---

## Week 15-16: Evaluation (Phase 6)
**Date:** June 8 - June 24

- Evaluation grid: 4 SOH x 3-4 SOC x 50 episodes x 5 seeds per policy
- Three policies evaluated:
  - HHRL (homeostatic drive-reduction)
  - HRL No-DR (ablation - sparse reward only, same architecture)
  - Fixed-threshold baseline (charge at SOC < 30%)
- Per-step CSV logging via RunLogger + EpisodeLogger
- Total: 10,000 evaluation episodes across all conditions

---

## Week 17-18: Thesis Writeup & Code Cleanup
**Date:** June 25 - July 19
- Thesis written 
- Training data archived
- Demo tools: tkinter GUI (demo_gui.py), battery RViz2 display (battery_display.py), screen recording script (demo_hrl.py)

