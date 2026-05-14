"""TD3 + HER training script for the flat homeostatic baseline (Phase 4)."""

from datetime import datetime
from pathlib import Path

import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer

from homeostatic_bot.env_wrapper import HomeostaticBotEnv


# =============================================================================
# CONFIG
# =============================================================================

# Smoke test default for HER implementation, first ran on 50k timesteps and 10k chekpnt 

TOTAL_TIMESTEPS = 500_000
CHECKPOINT_FREQ = 50_000
SEED = 42

# Battery rates — slow values for policy evaluation.
DRAIN_RATE_MOVING = 0.5
DRAIN_RATE_IDLE = 0.005
CHARGE_RATE = 2.0

# === Option C: relaxed flat-baseline task ===
# Cardinal-direction goals with clear straight-line paths from spawn, larger goal radius.
# Asymmetric protocol: only the flat baseline uses these; threshold and HRL keep
# methodology defaults (random goals at 0.3 m radius). Rationale and defense framing
# documented in Phase4_Implementation_Manual_Supplement2.md § 5.
USE_OPTION_C = True

if USE_OPTION_C:
    GOAL_RADIUS = 0.5
    GOAL_SET = [
        (3.0, 0.0),    # east of spawn — clear straight-line path
        (-3.0, 0.0),   # west of spawn — clear straight-line path
        (1.5, 3.0),    # north (offset from (0, 3) to clear obstacle 1 at (0, 2))
        (0.0, -3.0),   # south of spawn — clear straight-line path
    ]
else:
    GOAL_RADIUS = 0.3   # methodology default
    GOAL_SET = None     # random sampling per methodology (Section 1.3)

# TD3 hyperparameters (Fujimoto et al. 2018 + SB3 defaults).
LEARNING_RATE = 3e-4
BUFFER_SIZE = 100_000
BATCH_SIZE = 256
GAMMA = 0.99
TAU = 0.005
LEARNING_STARTS = 25_000
TRAIN_FREQ = 1
GRADIENT_STEPS = 1
POLICY_DELAY = 2

ACTION_NOISE_SIGMA_FRACTION = 0.2 # was 0.1
NET_ARCH = [400, 300]

# HER hyperparameters (Andrychowicz et al. 2017).
N_SAMPLED_GOAL = 4                         # number of relabeled goals per real goal
GOAL_SELECTION_STRATEGY = "future"         # most common HER strategy


def main():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_root = Path.home() / "thesis_logs" / "flat_td3_her" / timestamp
    log_root.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = log_root / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    tb_dir = log_root / "tensorboard"

    # Goal-conditioned env (returns Dict obs and implements compute_reward()).
    env = HomeostaticBotEnv(seed=SEED, goal_conditioned=True)
    env.DRAIN_RATE_MOVING = DRAIN_RATE_MOVING
    env.DRAIN_RATE_IDLE = DRAIN_RATE_IDLE
    env.CHARGE_RATE = CHARGE_RATE
    env.GOAL_REACHED_RADIUS = GOAL_RADIUS
    env.GOAL_SET = GOAL_SET

    env = Monitor(env, filename=str(log_root / "monitor"))

    # Action noise — Gaussian, scaled per-dimension to action range.
    action_high = env.action_space.high
    action_noise = NormalActionNoise(
        mean=np.zeros_like(action_high),
        sigma=ACTION_NOISE_SIGMA_FRACTION * action_high,
    )

    policy_kwargs = dict(net_arch=NET_ARCH)

    # MultiInputPolicy is required for Dict observation spaces (i.e. goal-conditioned).
    model = TD3(
        policy="MultiInputPolicy",
        env=env,
        learning_rate=LEARNING_RATE,
        buffer_size=BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        tau=TAU,
        learning_starts=LEARNING_STARTS,
        train_freq=TRAIN_FREQ,
        gradient_steps=GRADIENT_STEPS,
        policy_delay=POLICY_DELAY,
        action_noise=action_noise,
        policy_kwargs=policy_kwargs,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(
            n_sampled_goal=N_SAMPLED_GOAL,
            goal_selection_strategy=GOAL_SELECTION_STRATEGY,
        ),
        seed=SEED,
        verbose=1,
        tensorboard_log=str(tb_dir),
    )

    print(f"\n{'=' * 70}")
    print(f"TD3 + HER Training — Flat Homeostatic Baseline")
    print(f"{'=' * 70}")
    print(f"Log dir:         {log_root}")
    print(f"Total timesteps: {TOTAL_TIMESTEPS:,}")
    print(f"Checkpoints:     every {CHECKPOINT_FREQ:,} steps")
    print(f"Seed:            {SEED}")
    print(f"Device:          {model.device}")
    print(f"Battery rates:   drain={DRAIN_RATE_MOVING}/{DRAIN_RATE_IDLE} %/s, charge={CHARGE_RATE} %/s")
    print(f"Option C:        {USE_OPTION_C}, goal radius={GOAL_RADIUS} m")
    if GOAL_SET is None:
        print(f"Goal sampling:   random uniform with rejection (methodology default)")
    else:
        print(f"Goal set:        {GOAL_SET}")
    print(f"Network:         {NET_ARCH}, lr={LEARNING_RATE}, batch={BATCH_SIZE}")
    print(f"HER:             n_sampled_goal={N_SAMPLED_GOAL}, strategy={GOAL_SELECTION_STRATEGY}")
    print(f"TensorBoard:     run `tensorboard --logdir {tb_dir}` in another terminal")
    print(f"{'=' * 70}\n")

    checkpoint_cb = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=str(checkpoint_dir),
        name_prefix="td3_her",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=checkpoint_cb,
            log_interval=10,
        )
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Saving model state before exit...")

    final_model_path = log_root / "final_model"
    model.save(str(final_model_path))
    print(f"\nFinal model saved to: {final_model_path}.zip")
    print(f"Monitor log:          {log_root}/monitor.monitor.csv")
    print(f"TensorBoard logs:     {tb_dir}")

    env.close()
    print("\n✓ Training complete.")


if __name__ == "__main__":
    main()
