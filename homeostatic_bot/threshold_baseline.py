"""Fixed-threshold rule-based baseline """

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from homeostatic_bot.env_wrapper import HomeostaticBotEnv
from homeostatic_bot.logger import RunLogger


# Observation indices - keep in sync with env_wrapper._compute_obs()
_IDX_X = 0
_IDX_Y = 1
_IDX_YAW = 2
_IDX_SOC = 5
_IDX_LIDAR_FRONT = 7
_IDX_LIDAR_LEFT = 8
_IDX_LIDAR_RIGHT = 9


@dataclass
class ThresholdPolicy:
    #If SOC < threshold, drive to charger. Otherwise drive to goal. Reactive obstacle avoidance.
    # Charging logic
    charge_threshold: float = 30.0       # Below this SOC, divert to charger.

    # Proportional controller (tuned for smooth tracking)
    linear_vel: float = 0.15
    angular_gain: float = 0.6            # Lowered from 1.0 to reduce overshoot oscillation.
    heading_deadband: float = 0.1        # rad (~5.7°) — stop correcting when close enough.
    max_angular: float = 1.82

    # Obstacle avoidance
    avoid_threshold: float = 0.4
    avoid_angular_vel: float = 1.0

    # Targets
    charger_pos: np.ndarray = field(
        default_factory=lambda: np.array([4.0, 4.0], dtype=np.float32)
    )

    # Toggle to print decisions to console (logger always captures everything)
    debug: bool = False

    def __call__(self, obs: np.ndarray, goal: np.ndarray) -> np.ndarray:
        #Compute action from observation. Returns [linear_vel, angular_vel]

        # 1. Decide target - charger or goal?
        soc = obs[_IDX_SOC]
        if soc < self.charge_threshold:
            target = self.charger_pos
        else:
            target = goal

        # 2. Reactive obstacle avoidance overrides target-seeking.
        front = obs[_IDX_LIDAR_FRONT]
        left = obs[_IDX_LIDAR_LEFT]
        right = obs[_IDX_LIDAR_RIGHT]

        if front < self.avoid_threshold:
            angular = self.avoid_angular_vel if left > right else -self.avoid_angular_vel
            if self.debug:
                print(f"  [AVOID front] F={front:.2f} L={left:.2f} R={right:.2f} → ang={angular:+.2f}", flush=True)
            return np.array([0.05, angular], dtype=np.float32)
        if left < self.avoid_threshold:
            if self.debug:
                print(f"  [AVOID left]  F={front:.2f} L={left:.2f} R={right:.2f} → ang=-{self.avoid_angular_vel:.2f}", flush=True)
            return np.array([0.05, -self.avoid_angular_vel], dtype=np.float32)
        if right < self.avoid_threshold:
            if self.debug:
                print(f"  [AVOID right] F={front:.2f} L={left:.2f} R={right:.2f} → ang=+{self.avoid_angular_vel:.2f}", flush=True)
            return np.array([0.05, self.avoid_angular_vel], dtype=np.float32)

        # 3. No obstacle pressure - proportional control toward target.
        x, y, yaw = obs[_IDX_X], obs[_IDX_Y], obs[_IDX_YAW]
        dx = target[0] - x
        dy = target[1] - y
        target_heading = np.arctan2(dy, dx)
        # Heading error wrapped to [-π, π]
        heading_error = np.arctan2(
            np.sin(target_heading - yaw),
            np.cos(target_heading - yaw),
        )

        # Heading deadband: Prevents oscillation
        # around the target heading when proportional gain alone would over/undershoot.
        if abs(heading_error) < self.heading_deadband:
            angular = 0.0
        else:
            angular = float(np.clip(
                self.angular_gain * heading_error,
                -self.max_angular,
                self.max_angular,
            ))

        # Slow forward velocity when heading error is large - turn first, then drive.
        linear = self.linear_vel
        if abs(heading_error) > np.pi / 4:
            linear *= 0.3

        if self.debug:
            print(f"  [pol] pos=({x:+.2f},{y:+.2f}) yaw={yaw:+.2f} "
                  f"tgt=({target[0]:+.2f},{target[1]:+.2f}) "
                  f"hdg_err={heading_error:+.2f} "
                  f"lin={linear:.2f} ang={angular:+.2f} "
                  f"SOC={soc:.1f}", flush=True)

        return np.array([linear, angular], dtype=np.float32)


def run_episode(
    env: HomeostaticBotEnv,
    policy: ThresholdPolicy,
    run_logger: RunLogger,
    episode_id: str,
    initial_soh: Optional[float] = None,
    initial_soc: Optional[float] = None,
    verbose: bool = True,
) -> dict:
    #Run one episode with the threshold policy. Logs every step + episode summary

    options = {}
    if initial_soh is not None:
        options["initial_soh"] = initial_soh
    if initial_soc is not None:
        options["initial_soc"] = initial_soc
    options = options or None
    obs, info = env.reset(options=options)
    goal = info["goal"]

    ep_logger = run_logger.start_episode(
        episode_id=episode_id,
        initial_soh=info["initial_soh"],
        goal=(float(goal[0]), float(goal[1])),
    )

    if verbose:
        print(f"  goal = ({goal[0]:+.2f}, {goal[1]:+.2f}), initial SOH = {info['initial_soh']:.0f}%")

    while True:
        action = policy(obs, goal)
        next_obs, reward, terminated, truncated, info = env.step(action)

        ep_logger.log_step(
            obs=next_obs,
            action=action,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
            charger_radius=env.CHARGER_RADIUS,
        )

        obs = next_obs

        if verbose and ep_logger._steps % 100 == 0:
            print(
                f"  [step {ep_logger._steps:4d}] SOC={info['soc']:5.1f}% "
                f"d_goal={info['dist_to_goal']:.2f} "
                f"d_charger={info['dist_to_charger']:.2f}"
            )

        if terminated or truncated:
            break

    outcome = (
        "reached_goal" if info["reached_goal"]
        else "battery_dead" if info["battery_dead"]
        else "time_limit"
    )

    summary = ep_logger.finish(outcome=outcome)
    return summary


def main():
    #Run the threshold baseline across all SOH levels — Phase 6 evaluation grid

    # =================================================================
    # CONFIG
    # =================================================================
    SOH_LEVELS = [100.0, 80.0, 60.0, 40.0]
    SOC_LEVELS = [80.0, 60.0, 40.0]  
    EPISODES_PER_CONDITION = 50
    SEEDS = [42,1963, 1949, 456, 789]  
    NUM_GOALS = 2

    DRAIN_RATE_MOVING = 0.5
    DRAIN_RATE_IDLE = 0.005
    CHARGE_RATE = 5.0
    MAX_EPISODE_STEPS = 1200

    print("=" * 70)
    print("Threshold baseline - Phase 6 evaluation grid (2-goal, SOC sweep)")
    print("=" * 70)

    from homeostatic_bot.homeostatic_reward import HomeostaticReward

    policy = ThresholdPolicy(debug=False)

    print(f"\nPolicy: charge_threshold={policy.charge_threshold}%, "
          f"linear_vel={policy.linear_vel} m/s, "
          f"angular_gain={policy.angular_gain}, "
          f"heading_deadband={policy.heading_deadband:.2f} rad, "
          f"avoid_threshold={policy.avoid_threshold} m")
    print(f"Battery: drain_moving={DRAIN_RATE_MOVING} %/s, "
          f"drain_idle={DRAIN_RATE_IDLE} %/s, "
          f"charge_rate={CHARGE_RATE} %/s")
    print(f"Goals per episode: {NUM_GOALS}")
    print(f"SOH levels: {SOH_LEVELS}")
    print(f"SOC levels: {SOC_LEVELS}")
    print(f"Episodes per condition per seed: {EPISODES_PER_CONDITION}")
    print(f"Seeds: {SEEDS}")
    print(f"Total episodes: {len(SOH_LEVELS) * len(SOC_LEVELS) * EPISODES_PER_CONDITION * len(SEEDS)}\n")

    all_results: dict[tuple, list[dict]] = {}
    for soc in SOC_LEVELS:
        for soh in SOH_LEVELS:
            all_results[(soc, soh)] = []

    for seed in SEEDS:
        print(f"\n{'='*40} Seed {seed} {'='*40}")

        env = HomeostaticBotEnv(reward_fn=HomeostaticReward(), seed=seed)
        env.DRAIN_RATE_MOVING = DRAIN_RATE_MOVING
        env.DRAIN_RATE_IDLE = DRAIN_RATE_IDLE
        env.CHARGE_RATE = CHARGE_RATE
        env.NUM_GOALS = NUM_GOALS
        env.MAX_EPISODE_STEPS = MAX_EPISODE_STEPS

        run_logger = RunLogger(
            experiment_name="threshold_baseline_2goal",
            config={
                "policy": "ThresholdPolicy(charge_threshold=30%)",
                "reward": "HomeostaticReward(default)",
                "env_seed": seed,
                "num_goals": NUM_GOALS,
                "soh_levels": SOH_LEVELS,
                "soc_levels": SOC_LEVELS,
                "episodes_per_condition": EPISODES_PER_CONDITION,
                "battery_rates": {
                    "drain_moving": DRAIN_RATE_MOVING,
                    "drain_idle": DRAIN_RATE_IDLE,
                    "charge_rate": CHARGE_RATE,
                },
            },
        )

        for soc in SOC_LEVELS:
            for soh in SOH_LEVELS:
                print(f"\n--- SOC = {soc}%, SOH = {soh}%, Seed = {seed} ---")
                for ep in range(EPISODES_PER_CONDITION):
                    episode_id = f"seed{seed}_soc{int(soc)}_soh{int(soh)}_ep{ep}"
                    result = run_episode(
                        env=env,
                        policy=policy,
                        run_logger=run_logger,
                        episode_id=episode_id,
                        initial_soh=soh,
                        initial_soc=soc,
                        verbose=(ep < 2),
                    )
                    all_results[(soc, soh)].append(result)

                    if ep < 2 or (ep + 1) % 10 == 0:
                        print(
                            f"  ep {ep}: {result['outcome']:<14} "
                            f"steps={result['steps']:>5}  "
                            f"SOC={result['final_soc']:>5.1f}%  "
                            f"reward={result['total_reward']:>+8.1f}  "
                            f"charges={result['charging_visits']}"
                        )

        run_logger.close()
        env.close()

    # Summary table
    print("\n" + "=" * 70)
    print("Summary (aggregated across all seeds)")
    print("=" * 70)
    print(f"{'SOC':>5}  {'SOH':>5}  {'N':>4}  {'Success%':>8}  {'MeanSteps':>10}  "
          f"{'MeanSOC':>8}  {'MeanReward':>11}  {'MeanCharges':>12}")
    for soc in SOC_LEVELS:
        for soh in SOH_LEVELS:
            rs = all_results[(soc, soh)]
            n = len(rs)
            if n == 0:
                continue
            successes = sum(1 for r in rs if r['outcome'] == 'reached_goal')
            mean_steps = sum(r['steps'] for r in rs) / n
            mean_soc = sum(r['final_soc'] for r in rs) / n
            mean_reward = sum(r['total_reward'] for r in rs) / n
            mean_charges = sum(r['charging_visits'] for r in rs) / n
            print(
                f"{soc:>4}%  {soh:>4}%  {n:>4}  {successes/n*100:>7.1f}%  {mean_steps:>10.1f}  "
                f"{mean_soc:>7.1f}%  {mean_reward:>+11.1f}  {mean_charges:>12.1f}"
            )

    print("\n✓ Threshold baseline (2-goal) complete.")


if __name__ == "__main__":
    main()
