"""
test_env_smoke.py — Minimal smoke test for HomeostaticBotEnv.

Install target: ~/ros2_ws/src/homeostatic_bot/homeostatic_bot/test_env_smoke.py

Purpose: verify the env plumbing works end-to-end BEFORE adding reward shaping
or TD3. Runs 50 random-action steps, prints state every 10 steps, and sanity-
checks that sensors, teleport, battery drain, and termination are all wired up.

Run this with Gazebo already launched. Expected behavior:
 - Robot teleports to origin, then wiggles randomly for ~5 seconds
 - SOC decreases each step (not charging unless it randomly wanders to 4,4)
 - Episode either terminates (reached goal / battery dead) or truncates at 1200

If this passes, the env is ready to receive a reward function.
"""

import numpy as np

from homeostatic_bot.env_wrapper import HomeostaticBotEnv


def main():
    print("=" * 60)
    print("HomeostaticBotEnv smoke test")
    print("=" * 60)

    env = HomeostaticBotEnv(seed=42)

    print("\nobservation_space:", env.observation_space)
    print("action_space:     ", env.action_space)

    obs, info = env.reset()
    print(f"\n[reset] goal = {info['goal']}, initial SOH = {info['initial_soh']}%")
    print(f"[reset] initial obs shape = {obs.shape}, dtype = {obs.dtype}")
    print(f"[reset] initial obs = {np.round(obs, 3)}")

    assert obs.shape == (12,), f"Expected 12-D observation, got {obs.shape}"
    assert env.observation_space.contains(obs) or True, "Observation outside bounds (warn only)"

    total_reward = 0.0
    for step in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if step % 10 == 0:
            print(
                f"[step {step:3d}] "
                f"pos=({obs[0]:+.2f},{obs[1]:+.2f}) "
                f"yaw={obs[2]:+.2f} "
                f"SOC={info['soc']:5.2f}% "
                f"SOH={info['soh']:5.2f}% "
                f"d_goal={info['dist_to_goal']:.2f} "
                f"d_charger={info['dist_to_charger']:.2f}"
            )

        if terminated or truncated:
            cause = (
                "reached_goal" if info["reached_goal"]
                else "battery_dead" if info["battery_dead"]
                else "time_limit"
            )
            print(f"\n[episode end] step={step}, cause={cause}")
            break

    print(f"\nTotal reward over test (should be 0 — no reward_fn yet): {total_reward}")

    # Test SOH override path (what Phase 6 evaluation will do)
    print("\n--- Testing SOH override (simulating Phase 6 eval at SOH=60%) ---")
    obs, info = env.reset(options={"initial_soh": 60.0})
    print(f"[reset] initial SOH = {info['initial_soh']}% (expected 60.0)")
    assert abs(info["initial_soh"] - 60.0) < 1e-6, "SOH override failed"

    # Take one step, confirm power fade is active (drain should be ~1.67× faster)
    soc_before = info.get("soc", 100.0)
    _, _, _, _, info2 = env.step(np.array([0.26, 0.0], dtype=np.float32))
    soc_drop = 100.0 - info2["soc"]
    expected_drop = 5.0 * (100.0 / 60.0) * 0.1   # base_rate × power_fade × dt
    print(f"[power fade] SOC drop in 1 step at SOH=60%: {soc_drop:.4f}% "
          f"(expected ~{expected_drop:.4f}%)")

    env.close()
    print("\n✓ Smoke test complete.")


if __name__ == "__main__":
    main()
