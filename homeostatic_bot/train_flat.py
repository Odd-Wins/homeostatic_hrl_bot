"""TD3 training script for the flat homeostatic baseline (Phase 4)."""

from datetime import datetime
from pathlib import Path

import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise

from homeostatic_bot.env_wrapper import HomeostaticBotEnv
from homeostatic_bot.homeostatic_reward import HomeostaticReward


# =============================================================================
# CONFIG - change TOTAL_TIMESTEPS to scale from validation to full training.
# =============================================================================

# Training duration. Default = quick smoke test to validate code end-to-end.
# After confirming the smoke test learns *something* (rollout reward trends up),
# change to 500_000 for the full Phase 4 baseline run.
TOTAL_TIMESTEPS = 500_000          # SMOKE TEST - will change to 500_000 for full run
SEED = 42

# Checkpoint cadence. Smoke-test default saves at 5k for visibility.
# For full run, set to 50_000.
CHECKPOINT_FREQ = 50_000           # SMOKE TEST - will change to 50_000 for full run

# Battery rates - slow values for policy evaluation, same as threshold baseline.
# Gives ~200s mission time at SOH=100, room for the agent to learn navigation
# AND charging behavior without immediate survival pressure.
DRAIN_RATE_MOVING = 0.5
DRAIN_RATE_IDLE = 0.005
CHARGE_RATE = 2.0

# TD3 hyperparameters (Fujimoto et al. 2018 + SB3 defaults).
LEARNING_RATE = 3e-4              # SB3 default
BUFFER_SIZE = 100_000             # smaller than SB3's 1M default - adequate for 500k training
BATCH_SIZE = 256                  # SB3 default
GAMMA = 0.99                      # discount factor -  standard for episodic tasks
TAU = 0.005                       # target network soft-update rate
LEARNING_STARTS = 25_000          # pure random actions before training (warmup)
TRAIN_FREQ = 1                    # train after every env step
GRADIENT_STEPS = 1                # gradient steps per training call
POLICY_DELAY = 2                  # actor updates lag critic by 2 - TD3 signature

# Action noise - Gaussian, scaled to 10% of action range per dimension.
# Action space is asymmetric ([-0.26, 0.26] linear, [-1.82, 1.82] angular)
# so a single sigma would over-noise the small-magnitude action.
ACTION_NOISE_SIGMA_FRACTION = 0.1

# Network architecture -  TD3 paper default ([400, 300] for both actor and critic).
NET_ARCH = [400, 300]


def main():
    # -------------------------------------------------------------------------
    # 1. Set up output directories. Logs live outside the repo at
    #    ~/thesis_logs/flat_td3/<timestamp>/
    # -------------------------------------------------------------------------
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_root = Path.home() / "thesis_logs" / "flat_td3" / timestamp
    log_root.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = log_root / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    tb_dir = log_root / "tensorboard"

    # -------------------------------------------------------------------------
    # 2. Create env with policy-evaluation drain rates.
    # -------------------------------------------------------------------------
    env = HomeostaticBotEnv(reward_fn=HomeostaticReward(), seed=SEED)
    env.DRAIN_RATE_MOVING = DRAIN_RATE_MOVING
    env.DRAIN_RATE_IDLE = DRAIN_RATE_IDLE
    env.CHARGE_RATE = CHARGE_RATE

    # Monitor wrapper records (episode_reward, episode_length) to a CSV
    # at <log_root>/monitor.csv. SB3 also uses this for TensorBoard logging.
    env = Monitor(env, filename=str(log_root / "monitor"))

    # -------------------------------------------------------------------------
    # 3. Action noise-  Gaussian, scaled per-dimension to the action range.
    # -------------------------------------------------------------------------
    action_high = env.action_space.high                       # [0.26, 1.82]
    action_noise = NormalActionNoise(
        mean=np.zeros_like(action_high),
        sigma=ACTION_NOISE_SIGMA_FRACTION * action_high,      # [0.026, 0.182]
    )

    # -------------------------------------------------------------------------
    # 4. Create TD3 model.
    # -------------------------------------------------------------------------
    policy_kwargs = dict(net_arch=NET_ARCH)
    model = TD3(
        policy="MlpPolicy",
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
        seed=SEED,
        verbose=1,
        tensorboard_log=str(tb_dir),
    )

    # -------------------------------------------------------------------------
    # 5. Banner + run summary.
    # -------------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print(f"TD3 Training — Flat Homeostatic Baseline")
    print(f"{'=' * 70}")
    print(f"Log dir:         {log_root}")
    print(f"Total timesteps: {TOTAL_TIMESTEPS:,}")
    print(f"Checkpoints:     every {CHECKPOINT_FREQ:,} steps → {checkpoint_dir}")
    print(f"Seed:            {SEED}")
    print(f"Device:          {model.device}")
    print(f"Battery rates:   drain={DRAIN_RATE_MOVING}/{DRAIN_RATE_IDLE} %/s, charge={CHARGE_RATE} %/s")
    print(f"Network:         {NET_ARCH}, lr={LEARNING_RATE}, batch={BATCH_SIZE}")
    print(f"TensorBoard:     run `tensorboard --logdir {tb_dir}` in another terminal")
    print(f"{'=' * 70}\n")

    # -------------------------------------------------------------------------
    # 6. Periodic checkpointing - saves the model every CHECKPOINT_FREQ steps.
    #    save_replay_buffer=False because the buffer is huge and not needed
    #    for inference / evaluation.
    # -------------------------------------------------------------------------
    checkpoint_cb = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=str(checkpoint_dir),
        name_prefix="td3_homeo",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    # -------------------------------------------------------------------------
    # 7. TRAIN. KeyboardInterrupt is caught so a Ctrl-C still saves the model.
    # -------------------------------------------------------------------------
    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=checkpoint_cb,
            log_interval=10,                      # log every 10 episodes to console+TB
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Saving model state before exit...")

    # -------------------------------------------------------------------------
    # 8. Save final model.
    # -------------------------------------------------------------------------
    final_model_path = log_root / "final_model"
    model.save(str(final_model_path))
    print(f"\nFinal model saved to: {final_model_path}.zip")
    print(f"Monitor log:          {log_root}/monitor.monitor.csv")
    print(f"TensorBoard logs:     {tb_dir}")

    env.close()
    print("\n✓ Training complete.")


if __name__ == "__main__":
    main()
