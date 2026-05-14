"""Options-framework meta-environment for hierarchical homeostatic RL.

Wraps HomeostaticBotEnv. Each step() executes a complete option (GOTO_GOAL
or GOTO_CHARGER) using NavigationController, accumulating homeostatic reward
over the option's duration. The high-level DQN learns *when* to charge based
on the accumulated drive-reduction signal.

Architecture follows Sutton, Precup & Singh (1999) Options framework, adapted
for energy-aware HRL per Ramezani & Atashgah (2024).
"""

from typing import Any, Callable, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from homeostatic_bot.env_wrapper import HomeostaticBotEnv
from homeostatic_bot.navigation_controller import NavigationController


# Callback signature for per-base-step logging during evaluation.
# (obs, action, reward, terminated, truncated, info, option_action)
StepCallback = Callable[[np.ndarray, np.ndarray, float, bool, bool, dict, int], None]


class HRLMetaEnv(gym.Env):
    """Meta-environment where each step() runs a complete navigation option.

    Action space: Discrete(2) — 0=GOTO_GOAL, 1=GOTO_CHARGER
    Observation space: Box(12-D) — same as HomeostaticBotEnv flat mode
    """

    metadata = {"render_modes": []}

    GOTO_GOAL = 0
    GOTO_CHARGER = 1

    def __init__(
        self,
        base_env: HomeostaticBotEnv,
        max_option_steps: int = 200,
        charger_soc_target: float = 80.0,
        goal_stop_radius: float = 0.3,
        charger_stop_radius: float = 0.5,
        step_callback: Optional[StepCallback] = None,
        soh_levels: Optional[list[float]] = None,
        soh_rng: Optional[np.random.Generator] = None,
    ):
        super().__init__()
        self._base_env = base_env
        self._max_option_steps = max_option_steps
        self._charger_soc_target = charger_soc_target
        self._step_callback = step_callback
        self._soh_levels = soh_levels
        self._soh_rng = soh_rng or np.random.default_rng()

        # Two controllers with different stop radii.
        self._goal_controller = NavigationController(stop_radius=goal_stop_radius)
        self._charger_controller = NavigationController(stop_radius=charger_stop_radius)

        # Spaces mirror the base env flat mode.
        self.observation_space = base_env.observation_space
        self.action_space = spaces.Discrete(2)

        self._goal: Optional[np.ndarray] = None
        self._latest_obs: Optional[np.ndarray] = None

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        # Randomize SOH per episode if soh_levels is configured.
        if self._soh_levels is not None:
            soh = float(self._soh_rng.choice(self._soh_levels))
            options = options or {}
            options["initial_soh"] = soh
        obs, info = self._base_env.reset(seed=seed, options=options)
        self._goal = info["goal"].copy()
        self._latest_obs = obs.copy()
        return obs, info

    def _is_charged_at_charger(self, obs) -> bool:
        """Check if robot is already at charger with SOC at target (action masking)."""
        if self._latest_obs is None:
            return False
        pos = np.array([obs[0], obs[1]], dtype=np.float32)
        at_charger = float(np.linalg.norm(pos - self._base_env.CHARGER_POS)) < self._charger_controller.stop_radius
        soc_at_target = obs[5] >= self._charger_soc_target  # obs[5] = SOC
        return at_charger and soc_at_target

    def step(self, action: int):
        # Action masking: if already at charger with SOC at target,
        # force GOTO_GOAL. This implements the Options framework initiation
        # set — GOTO_CHARGER is not available when already fully charged
        # at the charging station. You can't "go to" where you already are.
        if action == self.GOTO_CHARGER and self._is_charged_at_charger(self._latest_obs):
            action = self.GOTO_GOAL

        if action == self.GOTO_GOAL:
            target = self._base_env._goal.copy()  # read current goal (may change mid-episode in multi-goal)
            controller = self._goal_controller
        else:
            target = self._base_env.CHARGER_POS
            controller = self._charger_controller

        accumulated_reward = 0.0
        option_steps = 0
        terminated = False
        truncated = False
        info: dict[str, Any] = {}
        obs = self._latest_obs

        for _ in range(self._max_option_steps):
            low_action = controller(obs, target)
            obs, reward, terminated, truncated, info = self._base_env.step(low_action)
            accumulated_reward += reward
            option_steps += 1

            if self._step_callback is not None:
                self._step_callback(obs, low_action, reward, terminated, truncated, info, action)

            # Check option-specific termination.
            if terminated or truncated:
                break
            if action == self.GOTO_GOAL and info.get("reached_current_goal", False):
                # Current delivery reached — option ends. The base env has
                # already advanced to the next goal in the queue.
                self._goal = self._base_env._goal.copy()
                break
            if action == self.GOTO_CHARGER:
                at_charger = info["dist_to_charger"] < self._charger_controller.stop_radius
                soc_at_target = info["soc"] >= self._charger_soc_target
                if at_charger and soc_at_target:
                    break

        self._latest_obs = obs.copy()

        # Augment info with option metadata.
        info["option_action"] = action
        info["option_steps"] = option_steps
        info["option_target"] = "goal" if action == self.GOTO_GOAL else "charger"

        return obs, accumulated_reward, terminated, truncated, info

    def close(self) -> None:
        self._base_env.close()
