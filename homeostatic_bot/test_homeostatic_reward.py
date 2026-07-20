"""Unit tests for HomeostaticReward (no Gazebo required)."""

import numpy as np

from homeostatic_bot.homeostatic_reward import (
    HomeostaticReward,
    make_default_reward,
)


def make_obs(soc: float = 100.0, lidar=(10.0, 10.0, 10.0)) -> np.ndarray:
    obs = np.zeros(12, dtype=np.float32)
    obs[5] = soc                
    obs[6] = 100.0              
    obs[7] = lidar[0]           
    obs[8] = lidar[1]           
    obs[9] = lidar[2]           
    return obs


def assert_close(actual: float, expected: float, label: str, tol: float = 1e-5):
    ok = abs(actual - expected) < tol
    mark = "✓" if ok else "✗"
    print(f"  {mark} {label}: got {actual:+.4f}, expected {expected:+.4f}")
    assert ok, f"FAILED: {label}"


def main():
    print("=" * 60)
    print("HomeostaticReward unit tests")
    print("=" * 60)

    r = make_default_reward()

    # ----- Test 1: drive reduction TOWARD setpoint --------------------
    print("\n[1] SOC 90 → 85 (moving toward 80% setpoint)")
    print("    drive_before=|90-80|=10, drive_after=|85-80|=5")
    print("    reward = 10 - 5 - 0.1 = +4.9")
    reward = r(make_obs(soc=90.0), np.zeros(2), make_obs(soc=85.0), {})
    assert_close(reward, 4.9, "drive toward setpoint")

    # ----- Test 2: drive reduction AWAY from setpoint -----------------
    print("\n[2] SOC 85 → 90 (moving away from 80% setpoint)")
    print("    drive_before=5, drive_after=10")
    print("    reward = 5 - 10 - 0.1 = -5.1")
    reward = r(make_obs(soc=85.0), np.zeros(2), make_obs(soc=90.0), {})
    assert_close(reward, -5.1, "drive away from setpoint")

    # ----- Test 3: SOC crosses setpoint symmetrically -----------------
    print("\n[3] SOC 85 → 75 (crosses 80% setpoint, symmetric)")
    print("    drive_before=5, drive_after=5 (same drive on both sides)")
    print("    reward = 0 - 0.1 = -0.1")
    reward = r(make_obs(soc=85.0), np.zeros(2), make_obs(soc=75.0), {})
    assert_close(reward, -0.1, "drive across setpoint")

    # ----- Test 4: goal bonus -----------------------------------------
    print("\n[4] Goal reached, SOC 80 → 80 (no drive change)")
    print("    reward = 0 - 0.1 + 10 = +9.9")
    reward = r(make_obs(soc=80.0), np.zeros(2), make_obs(soc=80.0),
               {"reached_goal": True})
    assert_close(reward, 9.9, "goal bonus")

    # ----- Test 5: death penalty --------------------------------------
    print("\n[5] Battery dies, SOC 5 → 0")
    print("    drive_before=75, drive_after=80")
    print("    reward = 75 - 80 - 0.1 - 10 = -15.1")
    reward = r(make_obs(soc=5.0), np.zeros(2), make_obs(soc=0.0),
               {"battery_dead": True})
    assert_close(reward, -15.1, "death penalty")

    # ----- Test 6: collision penalty triggered ------------------------
    print("\n[6] Front lidar 0.20 m (below 0.25 m threshold)")
    print("    reward = 0 - 0.1 - 1 = -1.1")
    reward = r(make_obs(soc=80.0), np.zeros(2),
               make_obs(soc=80.0, lidar=(0.20, 5.0, 5.0)), {})
    assert_close(reward, -1.1, "collision penalty")

    # ----- Test 7: collision threshold boundary (strict <) ------------
    print("\n[7] All lidar at exactly 0.25 m — NOT a collision (strict <)")
    print("    reward = -0.1 (only step cost)")
    reward = r(make_obs(soc=80.0), np.zeros(2),
               make_obs(soc=80.0, lidar=(0.25, 0.25, 0.25)), {})
    assert_close(reward, -0.1, "collision boundary")

    # ----- Test 8: combined good outcome ------------------------------
    print("\n[8] Reach goal at SOC 90 → 85 (good run)")
    print("    reward = 5 (drive) + 10 (goal) - 0.1 (step) = +14.9")
    reward = r(make_obs(soc=90.0), np.zeros(2), make_obs(soc=85.0),
               {"reached_goal": True})
    assert_close(reward, 14.9, "combined goal + drive")

    # ----- Test 9: combined worst case --------------------------------
    print("\n[9] Hit wall AND battery dies in same step")
    print("    drive_before=70, drive_after=80")
    print("    reward = -10 (drive) - 0.1 (step) - 1 (collide) - 10 (die)")
    print("           = -21.1")
    reward = r(make_obs(soc=10.0), np.zeros(2),
               make_obs(soc=0.0, lidar=(0.15, 5.0, 5.0)),
               {"battery_dead": True})
    assert_close(reward, -21.1, "death + collision combined")

    # ----- Test 10: ablation — disable collision penalty --------------
    print("\n[10] Phase 6 ablation pattern: disable collision penalty")
    print("    reward = -0.1 (step cost only, no collision term)")
    r_custom = HomeostaticReward(collision_penalty=0.0)
    reward = r_custom(make_obs(soc=80.0), np.zeros(2),
                      make_obs(soc=80.0, lidar=(0.10, 0.10, 0.10)), {})
    assert_close(reward, -0.1, "ablation: collision disabled")

    # ----- Test 11: ablation — different setpoint ---------------------
    print("\n[11] Ablation pattern: setpoint = 60% (sanity check)")
    print("    SOC 70 → 65, drive 10 → 5, reward = 5 - 0.1 = +4.9")
    r_custom = HomeostaticReward(setpoint=60.0)
    reward = r_custom(make_obs(soc=70.0), np.zeros(2),
                      make_obs(soc=65.0), {})
    assert_close(reward, 4.9, "ablation: setpoint=60")

    print("\n" + "=" * 60)
    print("✓ All 11 tests passed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
