"""Evaluation script for the ablation HRL agent (no homeostatic reward).

Identical to eval_hrl.py but loads the ablation model trained with
NoHomeostaticReward.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from stable_baselines3 import DQN

from homeostatic_bot.env_wrapper import HomeostaticBotEnv
from homeostatic_bot.hrl_no_homeostatic_reward import NoHomeostaticReward
from homeostatic_bot.hrl_meta_env import HRLMetaEnv
from homeostatic_bot.logger import RunLogger


# =============================================================================
# CONFIG
# =============================================================================

MODEL_PATH = str(
    Path.home() / "thesis_logs" / "hrl_dqn_no_homeostatic" / "2026-06-20_21-46-03" / "final_model.zip"
)
SEED = 42
SOH_LEVELS = [100.0, 80.0, 60.0, 40.0]
SOC_LEVELS = [100.0, 80.0, 60.0, 40.0]
EPISODES_PER_CONDITION = 50
SEEDS = [42, 1963, 1949, 456, 789]

# Battery rates — same as training and other evals.
DRAIN_RATE_MOVING = 0.5
DRAIN_RATE_IDLE = 0.005
CHARGE_RATE = 5.0
MAX_EPISODE_STEPS = 1200

# Option parameters — must match training config.
MAX_OPTION_STEPS = 300
CHARGER_SOC_TARGET = 80.0


def run_episode(
    model: DQN,
    meta_env: HRLMetaEnv,
    base_env: HomeostaticBotEnv,
    run_logger: RunLogger,
    episode_id: str,
    initial_soh: Optional[float] = None,
    initial_soc: Optional[float] = None,
    verbose: bool = True,
) -> dict:

    options = {}
    if initial_soh is not None:
        options["initial_soh"] = initial_soh
    if initial_soc is not None:
        options["initial_soc"] = initial_soc
    options = options or None
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
        for entry in option_log[:20]:
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
    print("ABLATION Eval — HRL DQN (No Homeostatic Reward)")
    print("=" * 70)

    # Load trained ablation model.
    print(f"\nLoading model from: {MODEL_PATH}")
    model = DQN.load(MODEL_PATH)
    print(f"Model loaded successfully.")

    print(f"\nBattery: drain_moving={DRAIN_RATE_MOVING} %/s, "
          f"drain_idle={DRAIN_RATE_IDLE} %/s, "
          f"charge_rate={CHARGE_RATE} %/s")
    print(f"Option config: max_steps={MAX_OPTION_STEPS}, "
          f"charger_target={CHARGER_SOC_TARGET}%")
    print(f"SOH levels: {SOH_LEVELS}")
    print(f"SOC levels: {SOC_LEVELS}")
    print(f"Episodes per condition per seed: {EPISODES_PER_CONDITION}")
    print(f"Seeds: {SEEDS}")
    print(f"Total episodes: {len(SOH_LEVELS) * len(SOC_LEVELS) * EPISODES_PER_CONDITION * len(SEEDS)}\n")

    # Results keyed by (soc, soh)
    all_results: dict[tuple, list[dict]] = {}
    for soc in SOC_LEVELS:
        for soh in SOH_LEVELS:
            all_results[(soc, soh)] = []

    for seed in SEEDS:
        print(f"\n{'='*40} Seed {seed} {'='*40}")

        base_env = HomeostaticBotEnv(
            reward_fn=NoHomeostaticReward(),
            seed=seed,
            goal_conditioned=False,
        )
        base_env.DRAIN_RATE_MOVING = DRAIN_RATE_MOVING
        base_env.DRAIN_RATE_IDLE = DRAIN_RATE_IDLE
        base_env.CHARGE_RATE = CHARGE_RATE
        base_env.NUM_GOALS = 2
        base_env.MAX_EPISODE_STEPS = MAX_EPISODE_STEPS

        meta_env = HRLMetaEnv(
            base_env=base_env,
            max_option_steps=MAX_OPTION_STEPS,
            charger_soc_target=CHARGER_SOC_TARGET,
        )

        run_logger = RunLogger(
            experiment_name="hrl_dqn_no_homeostatic_eval",
            config={
                "model_path": MODEL_PATH,
                "algorithm": "DQN (Options framework HRL — NO homeostatic reward)",
                "reward": "NoHomeostaticReward (sparse only)",
                "env_seed": seed,
                "num_goals": 2,
                "soh_levels": SOH_LEVELS,
                "soc_levels": SOC_LEVELS,
                "episodes_per_condition": EPISODES_PER_CONDITION,
                "max_option_steps": MAX_OPTION_STEPS,
                "charger_soc_target": CHARGER_SOC_TARGET,
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
                        model=model,
                        meta_env=meta_env,
                        base_env=base_env,
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
                            f"meta={result['meta_steps']:>3}  "
                            f"SOC={result['final_soc']:>5.1f}%  "
                            f"reward={result['total_reward']:>+8.1f}  "
                            f"charges={result['charging_visits']}"
                        )

        run_logger.close()
        meta_env.close()

    # Summary table.
    print("\n" + "=" * 70)
    print("Summary (aggregated across all seeds)")
    print("=" * 70)
    print(f"{'SOC':>5}  {'SOH':>5}  {'N':>4}  {'Success%':>8}  {'MeanSteps':>10}  "
          f"{'MeanMeta':>9}  {'MeanSOC':>8}  {'MeanReward':>11}  {'MeanCharges':>12}")
    for soc in SOC_LEVELS:
        for soh in SOH_LEVELS:
            rs = all_results[(soc, soh)]
            n = len(rs)
            if n == 0:
                continue
            successes = sum(1 for r in rs if r['outcome'] == 'reached_goal')
            mean_steps = sum(r['steps'] for r in rs) / n
            mean_meta = sum(r['meta_steps'] for r in rs) / n
            mean_soc = sum(r['final_soc'] for r in rs) / n
            mean_reward = sum(r['total_reward'] for r in rs) / n
            mean_charges = sum(r['charging_visits'] for r in rs) / n
            print(
                f"{soc:>4}%  {soh:>4}%  {n:>4}  {successes/n*100:>7.1f}%  {mean_steps:>10.1f}  "
                f"{mean_meta:>9.1f}  {mean_soc:>7.1f}%  {mean_reward:>+11.1f}  "
                f"{mean_charges:>12.1f}"
            )

    print("\n✓ Ablation evaluation complete.")


if __name__ == "__main__":
    main()
