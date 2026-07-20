"""Logging utilities for episode runs (per-step CSV + episode summary + run metadata)."""

import csv
import json
import os
import subprocess
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


# Default log root 
DEFAULT_LOG_ROOT = Path.home() / "thesis_logs"

STEP_COLUMNS = [
    # Timing
    "step",
    "wall_time_s",          # seconds since episode start (real wall clock)
    # Position and orientation (from observation)
    "x",
    "y",
    "yaw",
    "lin_vel",
    "ang_vel",
    # Battery (from observation)
    "soc",
    "soh",
    # Lidar sectors (from observation)
    "lidar_front",
    "lidar_left",
    "lidar_right",
    # Distances (from observation)
    "dist_to_goal_obs",     # from obs vector
    "dist_to_charger_obs",
    # Action commanded 
    "action_lin",
    "action_ang",
    # Reward and termination
    "reward",
    "terminated",
    "truncated",
    # From info dict 
    "soc_info",
    "soh_info",
    "dist_to_goal_info",
    "dist_to_charger_info",
    "reached_goal",
    "battery_dead",
    "charge_cycles",
    # Convenience flags computed at log time
    "is_charging",          # within charger radius this step
]

EPISODE_COLUMNS = [
    "episode_id",
    "start_time",
    "initial_soh",
    "goal_x",
    "goal_y",
    "outcome",              # reached_goal | battery_dead | time_limit
    "steps",
    "duration_s",
    "final_soc",
    "final_soh",
    "total_reward",
    "charging_visits",
    "min_dist_to_goal",
    "min_lidar",            # closest the robot got to any obstacle
]


class RunLogger:
    #Manages a single 'run' (one invocation of an experiment script).

    def __init__(
        self,
        experiment_name: str,
        config: Optional[dict] = None,
        log_root: Optional[Path] = None,
    ):
        log_root = Path(log_root) if log_root else DEFAULT_LOG_ROOT
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.run_dir = log_root / experiment_name / timestamp
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.experiment_name = experiment_name
        self._episode_summaries: list[dict] = []
        self._current_episode: Optional[EpisodeLogger] = None

        # Save run-level metadata up front so even a crashed run leaves a trace.
        self._write_metadata(config or {})

        print(f"[RunLogger] Logging to: {self.run_dir}")

    def _write_metadata(self, config: dict) -> None:
        git_commit = "unknown"
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True, text=True, timeout=2.0,
            )
            if result.returncode == 0:
                git_commit = result.stdout.strip()
        except Exception:
            pass

        # If config has dataclass fields, convert them to plain dicts for JSON.
        config_serializable = self._make_json_safe(config)

        meta = {
            "experiment_name": self.experiment_name,
            "run_started": datetime.now().isoformat(),
            "git_commit": git_commit,
            "config": config_serializable,
        }
        with open(self.run_dir / "run_metadata.json", "w") as f:
            json.dump(meta, f, indent=2, default=str)

    @staticmethod
    def _make_json_safe(obj: Any) -> Any:
        """Recursively convert dataclasses, numpy arrays, etc. to JSON-friendly types."""
        if is_dataclass(obj) and not isinstance(obj, type):
            return RunLogger._make_json_safe(asdict(obj))
        if isinstance(obj, dict):
            return {str(k): RunLogger._make_json_safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [RunLogger._make_json_safe(v) for v in obj]
        if hasattr(obj, "tolist"):       # numpy arrays
            return obj.tolist()
        if isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        return str(obj)

    def start_episode(
        self,
        episode_id: str,
        initial_soh: float,
        goal: tuple,
    ) -> "EpisodeLogger":
        #Begin a new episode. Returns an EpisodeLogger for per-step writes
        if self._current_episode is not None and not self._current_episode._finished:
            self._current_episode.finish(outcome="abandoned")
        self._current_episode = EpisodeLogger(
            run_dir=self.run_dir,
            episode_id=episode_id,
            initial_soh=initial_soh,
            goal=goal,
        )
        self._current_episode._parent = self
        return self._current_episode

    def record_episode_summary(self, summary: dict) -> None:
        """Called by EpisodeLogger.finish(); appends to in-memory list."""
        self._episode_summaries.append(summary)

    def close(self) -> None:
        """Write the episode summary CSV. Call once after all episodes finish."""
        if not self._episode_summaries:
            print("[RunLogger] No episodes recorded — nothing to summarize.")
            return

        summary_path = self.run_dir / "episode_summary.csv"
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=EPISODE_COLUMNS)
            writer.writeheader()
            for row in self._episode_summaries:
                # Fill missing columns with empty string so CSV stays well-formed.
                writer.writerow({col: row.get(col, "") for col in EPISODE_COLUMNS})

        print(f"[RunLogger] Wrote {len(self._episode_summaries)} episode summaries to {summary_path}")
        print(f"[RunLogger] Run complete: {self.run_dir}")


class EpisodeLogger:
    #Per-step CSV writer for one episode. Returned by RunLogger.start_episode()

    def __init__(self, run_dir: Path, episode_id: str, initial_soh: float, goal: tuple):
        self.run_dir = run_dir
        self.episode_id = episode_id
        self.initial_soh = initial_soh
        self.goal = goal
        self._start_time = time.time()
        self._finished = False

        # Aggregates accumulated as steps come in.
        self._total_reward = 0.0
        self._steps = 0
        self._charging_visits = 0
        self._was_charging = False
        self._min_dist_to_goal = float("inf")
        self._min_lidar = float("inf")
        self._last_info: dict = {}

        # Open the per-step CSV and write header.
        self._path = run_dir / f"steps_{episode_id}.csv"
        self._file = open(self._path, "w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=STEP_COLUMNS)
        self._writer.writeheader()
        self._file.flush()

        # Reference to parent so finish() can call back. Set by start_episode.
        self._parent: Optional[RunLogger] = None

    def log_step(
        self,
        obs,
        action,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
        charger_radius: float = 0.5,
    ) -> None:
        #Write one row to the per-step CSV. Flushed immediately for crash safety
        if self._finished:
            raise RuntimeError(f"log_step called on finished episode {self.episode_id}")

        self._steps += 1
        self._total_reward += float(reward)
        self._last_info = dict(info)

        # Update aggregates.
        d_goal = float(info.get("dist_to_goal", 0.0))
        if d_goal < self._min_dist_to_goal:
            self._min_dist_to_goal = d_goal
        lidar_min = float(min(obs[7], obs[8], obs[9]))
        if lidar_min < self._min_lidar:
            self._min_lidar = lidar_min

        # Edge-triggered charger visit count.
        d_charger = float(info.get("dist_to_charger", 0.0))
        is_charging = d_charger < charger_radius
        if is_charging and not self._was_charging:
            self._charging_visits += 1
        self._was_charging = is_charging

        row = {
            "step": self._steps,
            "wall_time_s": round(time.time() - self._start_time, 4),
            "x": float(obs[0]),
            "y": float(obs[1]),
            "yaw": float(obs[2]),
            "lin_vel": float(obs[3]),
            "ang_vel": float(obs[4]),
            "soc": float(obs[5]),
            "soh": float(obs[6]),
            "lidar_front": float(obs[7]),
            "lidar_left": float(obs[8]),
            "lidar_right": float(obs[9]),
            "dist_to_goal_obs": float(obs[10]),
            "dist_to_charger_obs": float(obs[11]),
            "action_lin": float(action[0]),
            "action_ang": float(action[1]),
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "soc_info": float(info.get("soc", "")) if "soc" in info else "",
            "soh_info": float(info.get("soh", "")) if "soh" in info else "",
            "dist_to_goal_info": d_goal,
            "dist_to_charger_info": d_charger,
            "reached_goal": bool(info.get("reached_goal", False)),
            "battery_dead": bool(info.get("battery_dead", False)),
            "charge_cycles": int(info.get("charge_cycles", 0)),
            "is_charging": is_charging,
        }
        self._writer.writerow(row)
        self._file.flush()

    def finish(self, outcome: str) -> dict:
        #Close the per-step CSV and return the episode summary dict
        if self._finished:
            return {}
        self._finished = True
        self._file.close()

        summary = {
            "episode_id": self.episode_id,
            "start_time": datetime.fromtimestamp(self._start_time).isoformat(),
            "initial_soh": self.initial_soh,
            "goal_x": self.goal[0],
            "goal_y": self.goal[1],
            "outcome": outcome,
            "steps": self._steps,
            "duration_s": round(time.time() - self._start_time, 3),
            "final_soc": float(self._last_info.get("soc", 0.0)),
            "final_soh": float(self._last_info.get("soh", 0.0)),
            "total_reward": round(self._total_reward, 4),
            "charging_visits": self._charging_visits,
            "min_dist_to_goal": round(self._min_dist_to_goal, 4),
            "min_lidar": round(self._min_lidar, 4),
        }
        if self._parent is not None:
            self._parent.record_episode_summary(summary)
        return summary
