"""Demo script - runs a single HHRL episode with GUI Gazebo.
Tweaked - Forces low SOC and degraded SOH to demonstrate charging behavior.
Run alongside Gazebo GUI, battery_display node, and RViz2.

Usage:
    1: ros2 launch homeostatic_bot energy_world.launch.py
    2: ros2 run homeostatic_bot battery_display
    3: rviz2  (add MarkerArray on /battery_display)
    4: ros2 run homeostatic_bot demo_hrl
"""

import subprocess
import time
from pathlib import Path

from stable_baselines3 import DQN

from homeostatic_bot.env_wrapper import HomeostaticBotEnv
from homeostatic_bot.homeostatic_reward import HomeostaticReward
from homeostatic_bot.hrl_meta_env import HRLMetaEnv


GOAL_COLORS = {
    1: ("1.0 0.0 0.0", "Goal 1 (RED)"),     # Red
    2: ("0.0 0.0 1.0", "Goal 2 (BLUE)"),    # Blue
}


def spawn_goal_marker(goal_x: float, goal_y: float, goal_num: int) -> None:
    """Spawn a visual sphere in Gazebo to mark the goal position."""
    color = GOAL_COLORS.get(goal_num, ("1.0 1.0 0.0",))[0]
    sdf = f"""<?xml version="1.0"?>
<sdf version="1.9">
  <model name="goal_{goal_num}">
    <static>true</static>
    <link name="link">
      <visual name="visual">
        <geometry><sphere><radius>0.15</radius></sphere></geometry>
        <material>
          <ambient>{color} 1.0</ambient>
          <diffuse>{color} 1.0</diffuse>
        </material>
      </visual>
    </link>
  </model>
</sdf>"""
    cmd = [
        "gz", "service",
        "-s", "/world/energy_world/create",
        "--reqtype", "gz.msgs.EntityFactory",
        "--reptype", "gz.msgs.Boolean",
        "--timeout", "3000",
        "--req",
        f'sdf: "{sdf.replace(chr(10), " ").replace(chr(34), chr(92)+chr(34))}" '
        f'pose: {{position: {{x: {goal_x}, y: {goal_y}, z: 0.3}}}} '
        f'name: "goal_{goal_num}"',
    ]
    try:
        subprocess.run(cmd, capture_output=True, timeout=5.0)
    except Exception:
        pass


def remove_goal_marker(goal_num: int) -> None:
    """Remove a goal marker from Gazebo."""
    cmd = [
        "gz", "service",
        "-s", "/world/energy_world/remove",
        "--reqtype", "gz.msgs.Entity",
        "--reptype", "gz.msgs.Boolean",
        "--timeout", "3000",
        "--req", f'name: "goal_{goal_num}" type: MODEL',
    ]
    try:
        subprocess.run(cmd, capture_output=True, timeout=5.0)
    except Exception:
        pass


# ================================================
# CONFIG — adjustable for differnrt demo scenarios
# ================================================

MODEL_PATH = str(
    Path.home() / "thesis_logs" / "hrl_dqn" / "2026-06-06_11-15-39" / "final_model.zip"
)

# Scenario: degraded battery - should trigger charging behavior
INITIAL_SOC = 40.0       # Low enough to require charging
INITIAL_SOH = 80.0       # Moderately degraded - 1.25x drain
SEED = 15  # Goals far apart - forces charging between deliveries

# Battery rates - same as training
DRAIN_RATE_MOVING = 0.5
DRAIN_RATE_IDLE = 0.005
CHARGE_RATE = 5.0

# Option parameters - same as training
MAX_OPTION_STEPS = 300
CHARGER_SOC_TARGET = 80.0
MAX_EPISODE_STEPS = 1200
NUM_GOALS = 2

# Slow down for demo visibility
STEP_DELAY = 0.0  # seconds extra delay per base-env step (0 = real-time)


def main():
    print("=" * 60)
    print("  HRL Demo — Screen Recording Mode")
    print("=" * 60)

    print(f"\nLoading model: {MODEL_PATH}")
    model = DQN.load(MODEL_PATH)

    base_env = HomeostaticBotEnv(
        reward_fn=HomeostaticReward(),
        seed=SEED,
        goal_conditioned=False,
    )
    base_env.DRAIN_RATE_MOVING = DRAIN_RATE_MOVING
    base_env.DRAIN_RATE_IDLE = DRAIN_RATE_IDLE
    base_env.CHARGE_RATE = CHARGE_RATE
    base_env.NUM_GOALS = NUM_GOALS
    base_env.MAX_EPISODE_STEPS = MAX_EPISODE_STEPS

    meta_env = HRLMetaEnv(
        base_env=base_env,
        max_option_steps=MAX_OPTION_STEPS,
        charger_soc_target=CHARGER_SOC_TARGET,
    )

    options = {
        "initial_soc": INITIAL_SOC,
        "initial_soh": INITIAL_SOH,
    }

    print(f"\nScenario:")
    print(f"  Initial SOC: {INITIAL_SOC}%")
    print(f"  Initial SOH: {INITIAL_SOH}% (drain multiplier: {100/INITIAL_SOH:.2f}x)")
    print(f"  Goals: {NUM_GOALS}")
    print(f"  Charge rate: {CHARGE_RATE} %/s")
    print(f"\nStarting in 3 seconds — start your screen recording now!")
    time.sleep(3)

    obs, info = meta_env.reset(options=options)
    goal = info["goal"]
    goal_queue = info.get("goal_queue", [goal])

    # Spawn red spheres at goal locations in Gazebo
    for i, g in enumerate(goal_queue):
        spawn_goal_marker(float(g[0]), float(g[1]), i + 1)
        print(f"  Spawned goal {i+1} marker at ({g[0]:+.2f}, {g[1]:+.2f})")

    print(f"\n{'='*60}")
    print(f"  Episode started")
    print(f"  Goal 1: ({goal[0]:+.2f}, {goal[1]:+.2f})")
    if len(goal_queue) > 1:
        print(f"  Goal 2: ({goal_queue[1][0]:+.2f}, {goal_queue[1][1]:+.2f})")
    print(f"  SOC: {INITIAL_SOC}%, SOH: {INITIAL_SOH}%")
    print(f"{'='*60}\n")

    total_reward = 0.0
    meta_step = 0
    goals_delivered = 0
    option_names = {0: "GOTO_GOAL", 1: "GOTO_CHARGER"}

    while True:
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)

        soc_before = obs[5]
        print(f"  Meta-step {meta_step + 1}: {option_names[action]}")
        print(f"    SOC before: {soc_before:.1f}%")

        obs, reward, terminated, truncated, info = meta_env.step(action)
        total_reward += reward
        meta_step += 1

        soc_after = info.get("soc", obs[5])
        option_steps = info.get("option_steps", 0)
        print(f"    SOC after:  {soc_after:.1f}%")
        print(f"    Option ran: {option_steps} steps")
        print(f"    Reward:     {reward:+.1f}")
        print(f"    d_goal:     {info.get('dist_to_goal', 0):.2f}m")

        if info.get("reached_current_goal", False):
            goals_delivered += 1
            remove_goal_marker(goals_delivered)
            print(f"    >>> DELIVERY {goals_delivered} COMPLETE! (marker removed) <<<")
        if info.get("reached_goal", False):
            print(f"    >>> ALL DELIVERIES COMPLETE! <<<")

        print()

        if terminated or truncated:
            break

    outcome = (
        "ALL GOALS DELIVERED" if info.get("reached_goal", False)
        else "BATTERY DEAD" if info.get("battery_dead", False)
        else "TIME LIMIT"
    )

    print(f"{'='*60}")
    print(f"  Episode finished: {outcome}")
    print(f"  Meta-steps: {meta_step}")
    print(f"  Total reward: {total_reward:+.1f}")
    print(f"  Final SOC: {info.get('soc', 0):.1f}%")
    print(f"  Charging visits: {info.get('charging_visits', 0)}")
    print(f"{'='*60}")

    
    meta_env.close()


if __name__ == "__main__":
    main()
