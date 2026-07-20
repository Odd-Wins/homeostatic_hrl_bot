"""tkinter GUI for launching HHRL demo episodes with configurable parameters.
Sliders for SOC, SOH, and no. goals, a Run button, and a live log area
terminal showing meta-step decisions and battery drain. Runs alongside Gazebo GUI.
Usage:
   1: ros2 launch homeostatic_bot energy_world.launch.py  (GUI mode)
   2: python3 -m homeostatic_bot.demo_gui
"""

import subprocess
import threading
import time
import tkinter as tk
from tkinter import ttk
from pathlib import Path

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from stable_baselines3 import DQN

from homeostatic_bot.env_wrapper import HomeostaticBotEnv
from homeostatic_bot.homeostatic_reward import HomeostaticReward
from homeostatic_bot.hrl_meta_env import HRLMetaEnv


MODEL_PATH = str(
    Path.home() / "thesis_logs" / "hrl_dqn" / "2026-06-06_11-15-39" / "final_model.zip"
)

OPTION_NAMES = {0: "GOTO_GOAL", 1: "GOTO_CHARGER"}

GOAL_COLORS = {
    1: ("1.0 0.0 0.0", "Goal 1 (RED)"),
    2: ("0.0 0.0 1.0", "Goal 2 (BLUE)"),
}


def spawn_goal_marker(goal_x: float, goal_y: float, goal_num: int) -> None:
    color = GOAL_COLORS.get(goal_num, ("1.0 1.0 0.0",))[0]
    sdf = f"""<?xml version="1.0"?>
<sdf version="1.9">
  <model name="goal_{goal_num}">
    <static>true</static>
    <link name="link">
      <visual name="visual">
        <geometry><sphere><radius>0.15</radius></sphere></geometry>
        <material>
          <ambient>{color} 1.0</ambient>
          <diffuse>{color} 1.0</diffuse>
        </material>
      </visual>
    </link>
  </model>
</sdf>"""
    cmd = [
        "gz", "service",
        "-s", "/world/energy_world/create",
        "--reqtype", "gz.msgs.EntityFactory",
        "--reptype", "gz.msgs.Boolean",
        "--timeout", "3000",
        "--req",
        f'sdf: "{sdf.replace(chr(10), " ").replace(chr(34), chr(92)+chr(34))}" '
        f'pose: {{position: {{x: {goal_x}, y: {goal_y}, z: 0.3}}}} '
        f'name: "goal_{goal_num}"',
    ]
    try:
        subprocess.run(cmd, capture_output=True, timeout=5.0)
    except Exception:
        pass


def remove_goal_marker(goal_num: int) -> None:
    cmd = [
        "gz", "service",
        "-s", "/world/energy_world/remove",
        "--reqtype", "gz.msgs.Entity",
        "--reptype", "gz.msgs.Boolean",
        "--timeout", "3000",
        "--req", f'name: "goal_{goal_num}" type: MODEL',
    ]
    try:
        subprocess.run(cmd, capture_output=True, timeout=5.0)
    except Exception:
        pass


class DemoGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("HRL Homeostatic Demo")
        root.geometry("700x650")
        root.resizable(False, False)

        self.model = None
        self.running = False

        # ROS 2 publisher for battery_display node
        rclpy.init()
        self._ros_node = rclpy.create_node("demo_gui_battery_pub")
        self._battery_pub = self._ros_node.create_publisher(
            Float32MultiArray, "/battery_status", 10
        )

        # --- Controls frame ---
        ctrl = ttk.LabelFrame(root, text="Scenario Configuration", padding=10)
        ctrl.pack(fill="x", padx=10, pady=5)

        # SOC slider
        ttk.Label(ctrl, text="Initial SOC (%)").grid(row=0, column=0, sticky="w")
        self.soc_var = tk.DoubleVar(value=40.0)
        self.soc_slider = tk.Scale(ctrl, from_=10, to=100, variable=self.soc_var,
                                    orient="horizontal", length=300, showvalue=False,
                                    sliderlength=15, resolution=5)
        self.soc_slider.grid(row=0, column=1, padx=5)
        self.soc_label = ttk.Label(ctrl, text="40%", width=8)
        self.soc_label.grid(row=0, column=2)

        # SOH slider
        ttk.Label(ctrl, text="Initial SOH (%)").grid(row=1, column=0, sticky="w")
        self.soh_var = tk.DoubleVar(value=80.0)
        self.soh_slider = tk.Scale(ctrl, from_=20, to=100, variable=self.soh_var,
                                    orient="horizontal", length=300, showvalue=False,
                                    sliderlength=15, resolution=5)
        self.soh_slider.grid(row=1, column=1, padx=5)
        self.soh_label = ttk.Label(ctrl, text="80%", width=8)
        self.soh_label.grid(row=1, column=2)

        # Goals dropdown
        ttk.Label(ctrl, text="Number of Goals").grid(row=2, column=0, sticky="w")
        self.goals_var = tk.IntVar(value=2)
        goals_combo = ttk.Combobox(ctrl, textvariable=self.goals_var, values=[1, 2, 3],
                                    state="readonly", width=5)
        goals_combo.grid(row=2, column=1, sticky="w", padx=5)

        # Seed
        ttk.Label(ctrl, text="Seed").grid(row=3, column=0, sticky="w")
        self.seed_var = tk.IntVar(value=15)
        seed_entry = ttk.Entry(ctrl, textvariable=self.seed_var, width=8)
        seed_entry.grid(row=3, column=1, sticky="w", padx=5)

        # Update labels on slider move
        def _update_soc(v):
            self.soc_label.configure(text=f"{int(float(v))}%")
        def _update_soh(v):
            val = float(v)
            self.soh_label.configure(text=f"{int(val)}%")
            drain = 100.0 / val if val > 0 else 999
            self.drain_config_label.configure(text=f"Drain multiplier: {drain:.2f}x")
        self.soc_slider.configure(command=_update_soc)
        self.soh_slider.configure(command=_update_soh)

        # Drain multiplier preview (updates with SOH slider)
        self.drain_config_label = ttk.Label(ctrl, text="Drain multiplier: 1.25x")
        self.drain_config_label.grid(row=4, column=0, columnspan=3, sticky="w", pady=(5, 0))

        # --- Battery display frame ---
        batt_frame = ttk.LabelFrame(root, text="Battery Status", padding=10)
        batt_frame.pack(fill="x", padx=10, pady=5)

        # SOC bar
        ttk.Label(batt_frame, text="SOC:").grid(row=0, column=0, sticky="w")
        self.soc_bar = ttk.Progressbar(batt_frame, length=400, mode="determinate",
                                        maximum=100)
        self.soc_bar.grid(row=0, column=1, padx=5)
        self.soc_bar["value"] = 40
        self.soc_pct = ttk.Label(batt_frame, text="40.0%", width=8)
        self.soc_pct.grid(row=0, column=2)

        # SOH bar
        ttk.Label(batt_frame, text="SOH:").grid(row=1, column=0, sticky="w")
        self.soh_bar = ttk.Progressbar(batt_frame, length=400, mode="determinate",
                                        maximum=100)
        self.soh_bar.grid(row=1, column=1, padx=5)
        self.soh_bar["value"] = 80
        self.soh_pct = ttk.Label(batt_frame, text="80.0%", width=8)
        self.soh_pct.grid(row=1, column=2)

        # Live drain info
        self.drain_label = ttk.Label(batt_frame, text="Drain: 0.50 %/s (1.25x)")
        self.drain_label.grid(row=2, column=0, columnspan=3, sticky="w")

        # --- Buttons ---
        btn_frame = ttk.Frame(root, padding=5)
        btn_frame.pack(fill="x", padx=10)

        self.run_btn = ttk.Button(btn_frame, text="Run Episode", command=self.start_episode)
        self.run_btn.pack(side="left", padx=5)

        self.stop_btn = ttk.Button(btn_frame, text="Stop", command=self.stop_episode,
                                    state="disabled")
        self.stop_btn.pack(side="left", padx=5)

        self.status_label = ttk.Label(btn_frame, text="Ready", foreground="green")
        self.status_label.pack(side="left", padx=20)

        # --- Log area ---
        log_frame = ttk.LabelFrame(root, text="Episode Log", padding=5)
        log_frame.pack(fill="both", expand=True, padx=10, pady=5)

        self.log_text = tk.Text(log_frame, height=15, font=("Consolas", 9),
                                 state="disabled", bg="#1e1e1e", fg="#d4d4d4")
        scrollbar = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        self.log_text.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Tag colors for log
        self.log_text.tag_configure("header", foreground="#569cd6")
        self.log_text.tag_configure("charge", foreground="#4ec9b0")
        self.log_text.tag_configure("goal", foreground="#dcdcaa")
        self.log_text.tag_configure("warning", foreground="#ce9178")
        self.log_text.tag_configure("success", foreground="#6a9955")
        self.log_text.tag_configure("fail", foreground="#f44747")

    def log(self, text: str, tag: str = "") -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", text + "\n", tag)
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def update_battery(self, soc: float, soh: float) -> None:
        self.soc_bar["value"] = max(0, min(100, soc))
        self.soc_pct.configure(text=f"{soc:.1f}%")
        self.soh_bar["value"] = max(0, min(100, soh))
        self.soh_pct.configure(text=f"{soh:.1f}%")
        multiplier = 100.0 / soh if soh > 0 else 999
        effective_drain = 0.5 * multiplier  # nominal drain * multiplier
        self.drain_label.configure(
            text=f"Drain: {effective_drain:.2f} %/s ({multiplier:.2f}x)"
        )
        # Publish for battery_display node
        msg = Float32MultiArray()
        msg.data = [float(soc), float(soh)]
        self._battery_pub.publish(msg)

    def start_episode(self) -> None:
        if self.running:
            return
        self.running = True
        self.run_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.status_label.configure(text="Running...", foreground="orange")

        # Clear log
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")

        # Run in background thread
        thread = threading.Thread(target=self._run_episode, daemon=True)
        thread.start()

    def stop_episode(self) -> None:
        self.running = False

    def _run_episode(self) -> None:
        try:
            soc = self.soc_var.get()
            soh = self.soh_var.get()
            num_goals = self.goals_var.get()
            seed = self.seed_var.get()

            self.root.after(0, self.update_battery, soc, soh)
            self.root.after(0, self.log,
                f"{'='*55}", "header")
            self.root.after(0, self.log,
                f"  SOC={soc:.0f}%  SOH={soh:.0f}%  Goals={num_goals}  Seed={seed}",
                "header")
            self.root.after(0, self.log,
                f"{'='*55}", "header")

            # Load model on first run
            if self.model is None:
                self.root.after(0, self.log, "  Loading model...", "")
                self.model = DQN.load(MODEL_PATH)
                self.root.after(0, self.log, "  Model loaded.", "")

            base_env = HomeostaticBotEnv(
                reward_fn=HomeostaticReward(),
                seed=seed,
                goal_conditioned=False,
            )
            base_env.DRAIN_RATE_MOVING = 0.5
            base_env.DRAIN_RATE_IDLE = 0.005
            base_env.CHARGE_RATE = 5.0
            base_env.NUM_GOALS = num_goals
            base_env.MAX_EPISODE_STEPS = 1200

            meta_env = HRLMetaEnv(
                base_env=base_env,
                max_option_steps=300,
                charger_soc_target=80.0,
            )

            options = {"initial_soc": soc, "initial_soh": soh}
            obs, info = meta_env.reset(options=options)
            goal_queue = info.get("goal_queue", [info["goal"]])

            for i, g in enumerate(goal_queue):
                spawn_goal_marker(float(g[0]), float(g[1]), i + 1)
                self.root.after(0, self.log,
                    f"  Goal {i+1} at ({g[0]:+.2f}, {g[1]:+.2f})", "goal")

            self.root.after(0, self.log, "", "")

            total_reward = 0.0
            meta_step = 0
            goals_delivered = 0

            while self.running:
                action, _ = self.model.predict(obs, deterministic=True)
                action = int(action)
                name = OPTION_NAMES[action]

                soc_before = obs[5]
                tag = "charge" if action == 1 else "goal"
                self.root.after(0, self.log,
                    f"  Step {meta_step+1}: {name}  (SOC {soc_before:.1f}%)", tag)

                obs, reward, terminated, truncated, info = meta_env.step(action)
                total_reward += reward
                meta_step += 1

                soc_after = info.get("soc", obs[5])
                soh_now = info.get("soh", obs[6]) if "soh" in info else soh
                steps = info.get("option_steps", 0)

                self.root.after(0, self.update_battery, soc_after, soh_now)
                self.root.after(0, self.log,
                    f"    SOC: {soc_before:.1f}% -> {soc_after:.1f}%  "
                    f"({steps} steps)  r={reward:+.1f}", "")

                if info.get("reached_current_goal", False):
                    goals_delivered += 1
                    remove_goal_marker(goals_delivered)
                    self.root.after(0, self.log,
                        f"    >>> DELIVERY {goals_delivered} COMPLETE <<<", "success")

                if info.get("reached_goal", False):
                    self.root.after(0, self.log,
                        f"    >>> ALL GOALS DELIVERED <<<", "success")

                if terminated or truncated:
                    break

            # Episode finished
            if info.get("reached_goal", False):
                outcome = "ALL GOALS DELIVERED"
                otag = "success"
            elif info.get("battery_dead", False):
                outcome = "BATTERY DEAD"
                otag = "fail"
            elif not self.running:
                outcome = "STOPPED BY USER"
                otag = "warning"
            else:
                outcome = "TIME LIMIT"
                otag = "warning"

            self.root.after(0, self.log, "", "")
            self.root.after(0, self.log, f"{'='*55}", "header")
            self.root.after(0, self.log, f"  {outcome}", otag)
            self.root.after(0, self.log,
                f"  Steps: {meta_step}  Reward: {total_reward:+.1f}  "
                f"SOC: {info.get('soc', 0):.1f}%", "")
            self.root.after(0, self.log, f"{'='*55}", "header")

            meta_env.close()

        except Exception as e:
            self.root.after(0, self.log, f"  ERROR: {e}", "fail")

        finally:
            self.running = False
            self.root.after(0, lambda: self.run_btn.configure(state="normal"))
            self.root.after(0, lambda: self.stop_btn.configure(state="disabled"))
            self.root.after(0, lambda: self.status_label.configure(
                text="Ready", foreground="green"))


def main():
    root = tk.Tk()
    gui = DemoGUI(root)
    root.mainloop()
    gui._ros_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
