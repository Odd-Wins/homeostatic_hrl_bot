"""DQN training script for the hierarchical homeostatic RL agent (Phase 5).

High-level DQN learns {GOTO_GOAL, GOTO_CHARGER} decisions from accumulated
homeostatic drive-reduction reward. Low-level navigation is hand-coded
(NavigationController). Architecture follows Options framework
(Sutton, Precup & Singh 1999).
"""

from datetime import datetime
from pathlib import Path

import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from homeostatic_bot.env_wrapper import HomeostaticBotEnv
from homeostatic_bot.homeostatic_reward import HomeostaticReward
from homeostatic_bot.hrl_meta_env import HRLMetaEnv


# =============================================================================
# CONFIG
# =============================================================================

TOTAL_TIMESTEPS = 20_000       # meta-steps
CHECKPOINT_FREQ = 5_000
SEED = 42

# Resume from Iteration 6 checkpoint (goal-reaching already learned).
# Set to None to train from scratch.
RESUME_FROM = str(
    Path.home() / "thesis_logs" / "hrl_dqn" / "2026-05-12_22-54-27" / "final_model.zip"
)

# Battery rates — same as threshold baseline and flat TD3.
DRAIN_RATE_MOVING = 0.5
DRAIN_RATE_IDLE = 0.005
CHARGE_RATE = 2.0

# SOH randomization — train across degradation levels so the agent
# experiences conditions where charging is necessary.
SOH_LEVELS = [100.0, 80.0, 60.0, 40.0]

# Option parameters
MAX_OPTION_STEPS = 300         # ~30s of sim-time per option max
CHARGER_SOC_TARGET = 80.0     # GOTO_CHARGER terminates when SOC >= setpoint

# Episode limit — reduced from 1200 (120s) to 600 (60s). Successful 2-goal
# episodes take ~40-60s; failed episodes burning the full 120s generate
# mostly useless data. Shorter episodes = more diverse training per hour.
MAX_EPISODE_STEPS = 600

# DQN hyperparameters — small network for 2-action discrete problem.
LEARNING_RATE = 1e-3
BUFFER_SIZE = 50_000
BATCH_SIZE = 64
GAMMA = 0.99                  # Required for homeostatic stability (Keramati & Gutkin 2014, Eq. 5)
EXPLORATION_FRACTION = 0.3     # more exploration since agent must discover charging benefit at low SOH
EXPLORATION_FINAL_EPS = 0.10   # higher than default — 2 actions need more final exploration (RL Zoo LunarLander)
LEARNING_STARTS = 500
TARGET_UPDATE_INTERVAL = 100
NET_ARCH = [64, 64]


def main():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_root = Path.home() / "thesis_logs" / "hrl_dqn" / timestamp
    log_root.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = log_root / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    tb_dir = log_root / "tensorboard"

    rng = np.random.default_rng(SEED)

    # Base env in flat mode with homeostatic reward.
    base_env = HomeostaticBotEnv(
        reward_fn=HomeostaticReward(),
        seed=SEED,
        goal_conditioned=False,
    )
    base_env.DRAIN_RATE_MOVING = DRAIN_RATE_MOVING
    base_env.DRAIN_RATE_IDLE = DRAIN_RATE_IDLE
    base_env.CHARGE_RATE = CHARGE_RATE
    base_env.NUM_GOALS = 2  # multi-goal: 2 deliveries per episode
    base_env.MAX_EPISODE_STEPS = MAX_EPISODE_STEPS

    # Wrap in options-framework meta-env with SOH randomization.
    meta_env = HRLMetaEnv(
        base_env=base_env,
        max_option_steps=MAX_OPTION_STEPS,
        charger_soc_target=CHARGER_SOC_TARGET,
        soh_levels=SOH_LEVELS,
        soh_rng=rng,
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
    print(f"HRL DQN Training — Hierarchical Homeostatic Agent (Phase 5)")
    print(f"{'=' * 70}")
    print(f"Log dir:           {log_root}")
    print(f"{resume_msg}")
    print(f"Total meta-steps:  {TOTAL_TIMESTEPS:,}")
    print(f"Checkpoints:       every {CHECKPOINT_FREQ:,} meta-steps")
    print(f"Seed:              {SEED}")
    print(f"Device:            {model.device}")
    print(f"Battery rates:     drain={DRAIN_RATE_MOVING}/{DRAIN_RATE_IDLE} %/s, charge={CHARGE_RATE} %/s")
    print(f"SOH levels:        {SOH_LEVELS}")
    print(f"Max episode steps: {MAX_EPISODE_STEPS}")
    print(f"Option config:     max_steps={MAX_OPTION_STEPS}, charger_target={CHARGER_SOC_TARGET}%")
    print(f"DQN:               net={NET_ARCH}, lr={LEARNING_RATE}, batch={BATCH_SIZE}")
    print(f"Exploration:       fraction={EXPLORATION_FRACTION}, final_eps={EXPLORATION_FINAL_EPS}")
    print(f"TensorBoard:       tensorboard --logdir {tb_dir}")
    print(f"{'=' * 70}\n")

    checkpoint_cb = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=str(checkpoint_dir),
        name_prefix="hrl_dqn",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

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
    print("\n✓ HRL training complete.")


if __name__ == "__main__":
    main()
