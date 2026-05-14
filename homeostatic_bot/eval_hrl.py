"""Evaluation script for the trained HRL agent across SOH degradation levels.

Loads a trained DQN model and runs it on the HRLMetaEnv, logging every base-env
step via the step_callback mechanism. Mirrors the threshold_baseline.py evaluation
pattern for Phase 6 comparison.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from stable_baselines3 import DQN

from homeostatic_bot.env_wrapper import HomeostaticBotEnv
from homeostatic_bot.homeostatic_reward import HomeostaticReward
from homeostatic_bot.hrl_meta_env import HRLMetaEnv
from homeostatic_bot.logger import RunLogger


# =============================================================================
# CONFIG
# =============================================================================

MODEL_PATH = str(
    Path.home() / "thesis_logs" / "hrl_dqn" / "2026-05-08_15-54-59" / "final_model.zip"
)
SEED = 42
SOH_LEVELS = [100.0, 80.0, 60.0, 40.0]
EPISODES_PER_SOH = 1  # increase to 50 for Phase 6 full eval

# Battery rates — same as training and threshold baseline.
DRAIN_RATE_MOVING = 0.5
DRAIN_RATE_IDLE = 0.005
CHARGE_RATE = 2.0

# Option parameters — must match training config.
MAX_OPTION_STEPS = 150
CHARGER_SOC_TARGET = 80.0


def run_episode(
    model: DQN,
    meta_env: HRLMetaEnv,
    base_env: HomeostaticBotEnv,
    run_logger: RunLogger,
    episode_id: str,
    initial_soh: Optional[float] = None,
    verbose: bool = True,
) -> dict:
    """Run one episode with the trained HRL agent. Logs every base-env step."""

    options = {"initial_soh": initial_soh} if initial_soh is not None else None
    obs, info = meta_env.reset(options=options)
    goal = info["goal"]

    ep_logger = run_logger.start_episode(
        episode_id=episode_id,
        initial_soh=info["initial_soh"],
        goal=(float(goal[0]), float(goal[1])),
    )

    # Step callback: called on every base-env step inside the option loop.
    def step_cb(base_obs, action, reward, terminated, truncated, step_info, option_action):
        ep_logger.log_step(
            obs=base_obs,
            action=action,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=step_info,
            charger_radius=base_env.CHARGER_RADIUS,
        )

    # Attach callback for this episode.
    meta_env._step_callback = step_cb

    if verbose:
        print(f"  goal = ({goal[0]:+.2f}, {goal[1]:+.2f}), initial SOH = {info['initial_soh']:.0f}%")

    total_reward = 0.0
    meta_steps = 0
    option_log = []

    while True:
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)

        soc_before = info.get("soc", obs[5] if hasattr(obs, '__len__') else 0)
        obs, reward, terminated, truncated, info = meta_env.step(action)
        total_reward += reward
        meta_steps += 1

        option_log.append({
            "meta_step": meta_steps,
            "action": "GOTO_GOAL" if action == 0 else "GOTO_CHARGER",
            "option_steps": info.get("option_steps", 0),
            "soc_before": soc_before,
            "soc_after": info.get("soc", 0),
            "reward": reward,
        })

        if verbose and meta_steps % 10 == 0:
            print(
                f"  [meta {meta_steps:4d}] action={option_log[-1]['action']:<14} "
                f"opt_steps={info.get('option_steps', 0):3d} "
                f"SOC={info.get('soc', 0):5.1f}% "
                f"d_goal={info.get('dist_to_goal', 0):.2f}"
            )

        if terminated or truncated:
            break

    # Detach callback.
    meta_env._step_callback = None

    outcome = (
        "reached_goal" if info.get("reached_goal", False)
        else "battery_dead" if info.get("battery_dead", False)
        else "time_limit"
    )

    summary = ep_logger.finish(outcome=outcome)
    summary["meta_steps"] = meta_steps
    summary["total_meta_reward"] = round(total_reward, 4)

    if verbose:
        print(f"\n  Option decisions:")
        for entry in option_log[:20]:  # show first 20
            print(
                f"    [{entry['meta_step']:3d}] {entry['action']:<14} "
                f"steps={entry['option_steps']:3d}  "
                f"SOC: {entry['soc_before']:.1f}% -> {entry['soc_after']:.1f}%  "
                f"r={entry['reward']:+.1f}"
            )
        if len(option_log) > 20:
            print(f"    ... ({len(option_log) - 20} more decisions)")

    return summary


def main():
    print("=" * 70)
    print("HRL DQN Evaluation — Phase 6 degradation grid")
    print("=" * 70)

    # Load trained model.
    print(f"\nLoading model from: {MODEL_PATH}")
    model = DQN.load(MODEL_PATH)
    print(f"Model loaded successfully.")

    # Create base env (flat mode with homeostatic reward).
    base_env = HomeostaticBotEnv(
        reward_fn=HomeostaticReward(),
        seed=SEED,
        goal_conditioned=False,
    )
    base_env.DRAIN_RATE_MOVING = DRAIN_RATE_MOVING
    base_env.DRAIN_RATE_IDLE = DRAIN_RATE_IDLE
    base_env.CHARGE_RATE = CHARGE_RATE

    # Wrap in meta-env (no callback yet — set per-episode in run_episode).
    meta_env = HRLMetaEnv(
        base_env=base_env,
        max_option_steps=MAX_OPTION_STEPS,
        charger_soc_target=CHARGER_SOC_TARGET,
    )

    run_logger = RunLogger(
        experiment_name="hrl_dqn_eval",
        config={
            "model_path": MODEL_PATH,
            "algorithm": "DQN (Options framework HRL)",
            "reward": "HomeostaticReward(default)",
            "env_seed": SEED,
            "soh_levels": SOH_LEVELS,
            "episodes_per_soh": EPISODES_PER_SOH,
            "max_option_steps": MAX_OPTION_STEPS,
            "charger_soc_target": CHARGER_SOC_TARGET,
            "battery_rates": {
                "drain_moving": DRAIN_RATE_MOVING,
                "drain_idle": DRAIN_RATE_IDLE,
                "charge_rate": CHARGE_RATE,
            },
        },
    )

    print(f"\nBattery: drain_moving={DRAIN_RATE_MOVING} %/s, "
          f"drain_idle={DRAIN_RATE_IDLE} %/s, "
          f"charge_rate={CHARGE_RATE} %/s")
    print(f"Option config: max_steps={MAX_OPTION_STEPS}, "
          f"charger_target={CHARGER_SOC_TARGET}%")
    print(f"SOH levels: {SOH_LEVELS}")
    print(f"Episodes per SOH: {EPISODES_PER_SOH}\n")

    results = {}
    for soh in SOH_LEVELS:
        soh_results = []
        for ep in range(EPISODES_PER_SOH):
            episode_id = f"soh{int(soh)}_ep{ep}"
            print(f"--- SOH = {soh}%, Episode {ep} ---")
            result = run_episode(
                model=model,
                meta_env=meta_env,
                base_env=base_env,
                run_logger=run_logger,
                episode_id=episode_id,
                initial_soh=soh,
                verbose=True,
            )
            soh_results.append(result)
            print(
                f"  -> outcome: {result['outcome']}, "
                f"steps: {result['steps']}, "
                f"meta_steps: {result['meta_steps']}, "
                f"final SOC: {result['final_soc']:.1f}%, "
                f"reward: {result['total_reward']:+.1f}, "
                f"charging visits: {result['charging_visits']}\n"
            )
        results[soh] = soh_results

    # Summary table.
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"{'SOH':>5}  {'Outcome':<14}  {'Steps':>6}  {'Meta':>6}  "
          f"{'FinalSOC':>9}  {'Reward':>8}  {'Visits':>6}")
    for soh, soh_results in results.items():
        for r in soh_results:
            print(
                f"{soh:>4}%  {r['outcome']:<14}  {r['steps']:>6}  "
                f"{r['meta_steps']:>6}  "
                f"{r['final_soc']:>8.1f}%  {r['total_reward']:>+8.1f}  "
                f"{r['charging_visits']:>6}"
            )

    run_logger.close()
    meta_env.close()
    print("\n✓ HRL evaluation complete.")


if __name__ == "__main__":
    main()
