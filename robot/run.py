#!/usr/bin/env python3
"""RUN A TRAINED POLICY ON THE REAL ROBOT.

    python robot/run.py policies/walk_v9.zip

--------------------------------------------------------------------------
BEFORE YOU RUN THIS
--------------------------------------------------------------------------
  1. Watch the policy in simulation first:
         python training/watch.py robot/policies/walk_v9.zip
     If it does not walk there, it will not walk here.

  2. Put the robot on a flat, open surface with nothing to fall off.

  3. Power the servos. USB alone powers the ESP32 but NOT the servos -- the
     robot will happily talk to you and refuse to move. Switch the battery on.

  4. Keep a hand near it. Ctrl-C stops the loop and rests the servos.

--------------------------------------------------------------------------
THE DEPLOYMENT LOOP
--------------------------------------------------------------------------
Compare this to the training loop in `training/train.py`. Half of it is gone:

    while running:
        action = policy(observation)     <- still here
        observation = env.step(action)   <- still here
        # collect the rewards            <- gone, nothing is learning
        # estimate advantages            <- gone
        # update the network             <- gone

That is all deployment is. The network's weights are frozen; it is now just a
function from 246 numbers to 8 numbers, evaluated 20 times a second.

Note `deterministic=True`. During training the policy SAMPLES from a
distribution around its best guess -- that randomness is how it explores. Here
we want its best guess every time. Leaving exploration on would make a
perfectly good policy stagger.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from env import BittleRobotEnv  # noqa: E402


def main(argv=None):
    a = argparse.ArgumentParser(description="Run a trained policy on a real Bittle.")
    a.add_argument("model", type=Path, help="Path to a trained .zip policy.")
    a.add_argument("--steps", type=int, default=400, help="Control steps (400 = 20 seconds).")
    a.add_argument("--port", help="Serial port. Auto-detected if omitted.")
    a.add_argument("--margin", type=float, default=15.0,
                   help="Degrees of clearance kept inside the measured joint "
                        "limits. Lower it only once you trust the policy.")
    args = a.parse_args(argv)

    # Be forgiving about where the policy path is written from: the repo root,
    # inside robot/, or an absolute path all work.
    model_path = args.model
    if not model_path.exists():
        for base in (HERE, HERE.parent):
            if (base / args.model).exists():
                model_path = base / args.model
                break
        else:
            raise SystemExit(f"No such policy: {args.model}")
    args.model = model_path

    from stable_baselines3 import PPO

    print(f"loading {args.model.name}")
    policy = PPO.load(str(args.model))

    env = BittleRobotEnv(port=args.port, safety_margin_deg=args.margin)
    step_times, stale_count = [], 0
    try:
        print("standing up...")
        obs, _ = env.reset()
        print(f"walking for {args.steps} steps "
              f"({args.steps * 0.05:.0f} s).  Ctrl-C to stop.\n")

        for step in range(args.steps):
            action, _ = policy.predict(obs, deterministic=True)
            obs, _reward, _term, _trunc, info = env.step(action)

            step_times.append(info["step_time"])
            stale_count += info["imu_stale"]

            if step % 20 == 0:
                pitch, roll = obs[1], obs[0]  # quaternion y, x components
                print(f"step {step:>4}  {info['step_time']*1000:>5.1f} ms  "
                      f"tilt[{roll:+.2f} {pitch:+.2f}]  "
                      f"joints {np.asarray(info['joint_deg'], int).tolist()}"
                      f"{'  STALE IMU' if info['imu_stale'] else ''}",
                      flush=True)

    except KeyboardInterrupt:
        print("\nstopped by user")
    finally:
        # ALWAYS rest the servos, on any exit path. Leaving them holding a pose
        # cooks them, and a crashed script that leaves a robot straining is how
        # you lose a gearbox.
        print("resting servos...")
        env.close()

    if step_times:
        times = np.array(step_times) * 1000
        print(f"\n{len(times)} steps   "
              f"loop {times.mean():.1f} ms mean / {times.max():.1f} ms worst   "
              f"(target {1000 * 0.05:.0f} ms)")
        print(f"stale IMU reads: {stale_count} / {len(times)}")
        if times.mean() > 60:
            print("  -> loop is running slow; the policy is acting at the wrong "
                  "rate and the gait will not match simulation.")
        if stale_count > len(times) * 0.1:
            print("  -> lots of stale sensor data; check the USB cable and close "
                  "any serial monitor holding the port.")


if __name__ == "__main__":
    main()
