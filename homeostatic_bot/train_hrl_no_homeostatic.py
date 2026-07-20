"""Ablation: HRL DQN training WITHOUT homeostatic drive-reduction reward."""

import signal
from datetime import datetime
from pathlib import Path

import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from homeostatic_bot.env_wrapper import HomeostaticBotEnv
from homeostatic_bot.hrl_no_homeostatic_reward import NoHomeostaticReward
from homeostatic_bot.hrl_meta_env import HRLMetaEnv


# =============================================================================
# CONFIG - Stage 1 (goal-reaching, SOH=100%)
# Change to Stage 2 after Stage 1 completes.
# =============================================================================

TOTAL_TIMESTEPS = 20_000
CHECKPOINT_FREQ = 1_000
SEED = 42

RESUME_FROM = str(Path.home() / "thesis_logs" / "hrl_dqn_no_homeostatic" / "2026-06-18_19-55-22" / "final_model.zip")

# Battery rates 
DRAIN_RATE_MOVING = 0.5
DRAIN_RATE_IDLE = 0.005
CHARGE_RATE = 5.0

# Stage 2: SOH randomized across levels
SOH_LEVELS = [100.0, 80.0, 60.0, 40.0]
# Stage 2: SOC randomized
SOC_RANGE = (40.0, 100.0)
# Stage 2: longer episodes
MAX_EPISODE_STEPS = 1200

# Option parameters - identical to main training
MAX_OPTION_STEPS = 300
CHARGER_SOC_TARGET = 80.0

# DQN hyperparameters - identical to main training
LEARNING_RATE = 1e-3
BUFFER_SIZE = 50_000
BATCH_SIZE = 64
GAMMA = 0.99
EXPLORATION_FRACTION = 0.3
EXPLORATION_FINAL_EPS = 0.10
LEARNING_STARTS = 500
TARGET_UPDATE_INTERVAL = 100
NET_ARCH = [64, 64]


def main():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_root = Path.home() / "thesis_logs" / "hrl_dqn_no_homeostatic" / timestamp
    log_root.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = log_root / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    tb_dir = log_root / "tensorboard"

    rng = np.random.default_rng(SEED)

    # Base env with NO HOMEOSTATIC reward.
    base_env = HomeostaticBotEnv(
        reward_fn=NoHomeostaticReward(),
        seed=SEED,
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
        soh_levels=SOH_LEVELS,
        soh_rng=rng if SOH_LEVELS else None,
        soc_range=SOC_RANGE,
    )
    meta_env = Monitor(meta_env, filename=str(log_root / "monitor"))

    if RESUME_FROM and Path(RESUME_FROM).exists():
        model = DQN.load(
            RESUME_FROM,
            env=meta_env,
            learning_rate=LEARNING_RATE,
            buffer_size=BUFFER_SIZE,
            batch_size=BATCH_SIZE,
            gamma=GAMMA,
            exploration_fraction=EXPLORATION_FRACTION,
            exploration_final_eps=EXPLORATION_FINAL_EPS,
            learning_starts=LEARNING_STARTS,
            target_update_interval=TARGET_UPDATE_INTERVAL,
            seed=SEED,
            verbose=1,
            tensorboard_log=str(tb_dir),
        )
        resume_msg = f"Resumed from: {RESUME_FROM}"
    else:
        model = DQN(
            policy="MlpPolicy",
            env=meta_env,
            learning_rate=LEARNING_RATE,
            buffer_size=BUFFER_SIZE,
            batch_size=BATCH_SIZE,
            gamma=GAMMA,
            exploration_fraction=EXPLORATION_FRACTION,
            exploration_final_eps=EXPLORATION_FINAL_EPS,
            learning_starts=LEARNING_STARTS,
            target_update_interval=TARGET_UPDATE_INTERVAL,
            policy_kwargs=dict(net_arch=NET_ARCH),
            seed=SEED,
            verbose=1,
            tensorboard_log=str(tb_dir),
        )
        resume_msg = "Training from scratch"

    print(f"\n{'=' * 70}")
    print(f"ABLATION: HRL DQN — NO Homeostatic Drive-Reduction")
    print(f"{'=' * 70}")
    print(f"Log dir:           {log_root}")
    print(f"{resume_msg}")
    print(f"Reward:            NoHomeostaticReward (goal_bonus + step_cost only)")
    print(f"Total meta-steps:  {TOTAL_TIMESTEPS:,}")
    print(f"Checkpoints:       every {CHECKPOINT_FREQ:,} meta-steps")
    print(f"Seed:              {SEED}")
    print(f"Device:            {model.device}")
    print(f"Battery rates:     drain={DRAIN_RATE_MOVING}/{DRAIN_RATE_IDLE} %/s, charge={CHARGE_RATE} %/s")
    print(f"SOH levels:        {SOH_LEVELS}")
    print(f"Max episode steps: {MAX_EPISODE_STEPS}")
    print(f"Option config:     max_steps={MAX_OPTION_STEPS}, charger_target={CHARGER_SOC_TARGET}%")
    if SOC_RANGE is not None:
        print(f"SOC range:         {SOC_RANGE[0]:.0f}% – {SOC_RANGE[1]:.0f}%")
    print(f"DQN:               net={NET_ARCH}, lr={LEARNING_RATE}, batch={BATCH_SIZE}")
    print(f"Exploration:       fraction={EXPLORATION_FRACTION}, final_eps={EXPLORATION_FINAL_EPS}")
    print(f"TensorBoard:       tensorboard --logdir {tb_dir}")
    print(f"{'=' * 70}\n")

    checkpoint_cb = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=str(checkpoint_dir),
        name_prefix="hrl_no_homeo",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    def _sigterm_handler(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, _sigterm_handler)

    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=checkpoint_cb,
            log_interval=10,
            reset_num_timesteps=True,
        )
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Saving model state before exit...")

    final_model_path = log_root / "final_model"
    model.save(str(final_model_path))
    print(f"\nFinal model saved to: {final_model_path}.zip")
    print(f"Monitor log:          {log_root}/monitor.monitor.csv")
    print(f"TensorBoard logs:     {tb_dir}")

    meta_env.close()
    print("\n✓ Ablation training complete.")


if __name__ == "__main__":
    main()
