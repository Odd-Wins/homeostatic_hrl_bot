"""End-to-end smoke test for HomeostaticBotEnv (requires Gazebo running)."""

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

    # One step at SOH=60%, confirm power fade gives ~1.67× drain.
    _, _, _, _, info2 = env.step(np.array([0.26, 0.0], dtype=np.float32))
    soc_drop = 100.0 - info2["soc"]
    expected_drop = 5.0 * (100.0 / 60.0) * 0.1   # base × power_fade × dt
    print(f"[power fade] SOC drop in 1 step at SOH=60%: {soc_drop:.4f}% "
          f"(expected ~{expected_drop:.4f}%)")

    env.close()
    print("\n✓ Smoke test complete.")


if __name__ == "__main__":
    main()
