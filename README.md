# Homeostatic HRL for Battery-Aware Robot Navigation

Hierarchical Reinforcement Learning agent that uses homeostatic drive-reduction to manage battery energy while completing multi-goal (2)delivery tasks. Built on ROS2 Jazzy + Gazebo Harmonic with a TurtleBot3 Waffle.

## Architecture

```
DQN Meta-Controller (2 actions)
├── GOTO_GOAL    → NavigationController → /cmd_vel
└── GOTO_CHARGER → NavigationController → /cmd_vel
```

- **Meta-controller:** DQN with [64, 64] MLP, trained via Options framework
- **Low-level:** Proportional controller with reactive obstacle avoidance
- **Reward:** Homeostatic drive-reduction (Keramati & Gutkin 2014) — reward proportional to SOC moving toward 80% setpoint
- **Battery model:** Huang et al. (2022) capacity fade + 1/SOH power fade

## Prerequisites

- Ubuntu 24.04
- ROS2 Jazzy
- Gazebo Harmonic (gz-sim 8.x)
- TurtleBot3 packages for Jazzy
- Python 3.12+

### Python Dependencies

```bash
pip install gymnasium numpy stable-baselines3 tensorboard
```

### ROS2 Packages

```bash
sudo apt install ros-jazzy-turtlebot3-gazebo ros-jazzy-ros-gz-sim ros-jazzy-ros-gz-bridge
```

## Build

```bash
cd ~/ros2_ws
colcon build --packages-select homeostatic_bot --symlink-install
source install/setup.bash
```

Set TurtleBot3 model (required):
```bash
export TURTLEBOT3_MODEL=waffle
```

## Launch Gazebo World

```bash
ros2 launch homeostatic_bot energy_world.launch.py
```

This launches Gazebo with the custom energy_world (arena with obstacles and charging station at [4.0, 4.0]), spawns the TurtleBot3, and bridges ground-truth pose.

## Running

### Training

```bash
# HRL agent (Options-framework DQN)
ros2 run homeostatic_bot train_hrl

# Ablation (no drive-reduction reward)
ros2 run homeostatic_bot train_hrl_no_homeostatic

# Flat TD3 + HER (negative result)
ros2 run homeostatic_bot train_flat
```

Training logs to `~/thesis_logs/<experiment_name>/<timestamp>/` with TensorBoard, monitor CSV, and model checkpoints.

### Evaluation

```bash
# HHRL policy
ros2 run homeostatic_bot eval_hrl

# Ablation policy
ros2 run homeostatic_bot eval_hrl_no_homeostatic

# Fixed-threshold baseline (no trained model needed)
ros2 run homeostatic_bot threshold_baseline
```

Evaluation runs a grid of SOC x SOH conditions x 50 episodes x 5 seeds. Results logged as per-step CSVs and episode summaries.

### Demo (GUI)

```bash
# 1 - Launch Gazebo (GUI mode)
ros2 launch homeostatic_bot energy_world.launch.py

# 2 - Battery display in RViz2
ros2 run homeostatic_bot battery_display

# 3 - Interactive demo with sliders
python3 -m homeostatic_bot.demo_gui
```

### Tests

```bash
# Reward function unit test (no Gazebo needed)
ros2 run homeostatic_bot test_homeostatic_reward

# Environment smoke test (requires Gazebo running)
ros2 run homeostatic_bot test_env_smoke
```

## File Structure

```
homeostatic_bot/
├── env_wrapper.py              Gymnasium env (12-D obs, 2-D action, 10 Hz)
├── hrl_meta_env.py             Options-framework meta-environment
├── navigation_controller.py    Proportional nav with obstacle avoidance
├── homeostatic_reward.py       Drive-reduction reward function
├── hrl_no_homeostatic_reward.py  Ablation reward (sparse only)
├── battery_node.py             ROS2 battery simulation node
├── docking_controller.py       AprilTag visual servoing
├── train_hrl.py                HRL DQN training script
├── train_hrl_no_homeostatic.py Ablation training script
├── train_flat.py               Flat TD3 + HER training script
├── eval_hrl.py                 HRL evaluation grid
├── eval_hrl_no_homeostatic.py  Ablation evaluation grid
├── threshold_baseline.py       Fixed-threshold rule-based policy + eval
├── logger.py                   Per-step CSV + episode summary logging
├── demo_gui.py                 tkinter GUI for live demo
├── demo_hrl.py                 Screen recording demo script
├── battery_display.py          RViz2 battery status markers
├── test_homeostatic_reward.py  Reward function unit tests
└── test_env_smoke.py           End-to-end environment smoke test

launch/
└── energy_world.launch.py      Gazebo + TurtleBot3 + bridge launch

worlds/
└── energy_world.sdf            Custom arena with obstacles and charger

models/
├── turtlebot3_waffle/          Modified model (camera disabled for training)
└── apriltag_0/                 AprilTag for docking controller
```

## Configuration

Key parameters are defined as constants at the top of each training/eval script:

| Parameter | Value | Description |
|-----------|-------|-------------|
| DRAIN_RATE_MOVING | 0.5 %/s | Battery drain while moving |
| DRAIN_RATE_IDLE | 0.005 %/s | Battery drain while idle |
| CHARGE_RATE | 5.0 %/s | Charging rate at station |
| MAX_OPTION_STEPS | 300 | Max base-env steps per option |
| CHARGER_SOC_TARGET | 80.0% | GOTO_CHARGER terminates at this SOC |
| SOH_LEVELS | [100, 80, 60, 40] | Domain randomisation during training |
| NUM_GOALS | 2 | Delivery goals per episode |

## Battery Model

Power fade: `drain_multiplier = 100 / SOH`
- SOH=100%: 1.00x drain (baseline)
- SOH=80%: 1.25x drain
- SOH=60%: 1.67x drain
- SOH=40%: 2.50x drain

Capacity fade: `SOH = (1 - alpha * cycles^beta) * 100%`
