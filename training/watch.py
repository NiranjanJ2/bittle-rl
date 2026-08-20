#!/usr/bin/env python3
"""Watch a trained policy walk in the simulator.

    python training/watch.py runs/walk/best_model.zip            # live window
    python training/watch.py runs/walk/best_model.zip --gif w.gif  # save a gif

Do this BEFORE putting a policy on the real robot. A policy that looks wrong
here will look worse there, and the robot cannot fall over in a GIF.

It also prints the reward breakdown, which is the fastest way to understand
what a policy is actually optimizing. If a gait looks strange, one term in
that table is almost always paying far more than you intended -- that is
reward hacking, and it is the single most common failure in applied RL.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from env import DEFAULT_URDF, BittleSimEnv  # noqa: E402


def main(argv=None):
    a = argparse.ArgumentParser(description="Replay a trained policy in simulation.")
    a.add_argument("model", type=Path, help="Path to a trained .zip policy.")
    a.add_argument("--steps", type=int, default=200, help="Steps to run (200 = one episode).")
    a.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    a.add_argument("--gif", type=Path, help="Save frames to this GIF instead of opening a window.")
    args = a.parse_args(argv)

    if not args.model.exists():
        raise SystemExit(f"No such policy: {args.model}")

    from stable_baselines3 import PPO

    env = BittleSimEnv(args.urdf, gui=args.gif is None)
    frames, totals = [], {}
    try:
        # Load ONLY the policy. The critic that trained it is not needed to act
        # -- it existed to judge actions, and judging is over.
        model = PPO.load(str(args.model))
        obs, _ = env.reset()
        total_reward = 0.0
        step = 0

        for step in range(args.steps):
            # deterministic=True: always the policy's best guess. During
            # training actions are sampled randomly around that guess, which is
            # how the agent explores. At deployment you want no dice rolls.
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            for k, v in info["reward_parts"].items():
                totals[k] = totals.get(k, 0.0) + v

            if args.gif:
                frames.append(_grab_frame(env))
            if terminated or truncated:
                break

        steps_taken = step + 1
        print(f"\n{steps_taken} steps   total reward {total_reward:+.1f}   "
              f"{'FELL' if terminated else 'survived'}")
        print(f"final speed {info['speed']*100:+.1f} cm/s   height {info['height']*100:.1f} cm\n")

        print(f"{'reward term':<16}{'total':>10}{'per step':>11}")
        print("-" * 37)
        for name, value in sorted(totals.items(), key=lambda kv: -abs(kv[1])):
            if abs(value) > 1e-9:
                print(f"{name:<16}{value:>+10.1f}{value / steps_taken:>+11.3f}")

        if args.gif:
            _save_gif(frames, args.gif)
            print(f"\nsaved {args.gif}")
    finally:
        env.close()


def _grab_frame(env):
    """Render one camera frame, following the robot."""
    p, c = env.p, env.client
    (x, y, _), _ = p.getBasePositionAndOrientation(env.robot, physicsClientId=c)
    view = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=[x, y, 0.05], distance=0.42,
        yaw=50, pitch=-25, roll=0, upAxisIndex=2, physicsClientId=c)
    proj = p.computeProjectionMatrixFOV(fov=60, aspect=4 / 3, nearVal=0.01, farVal=2.0,
                                        physicsClientId=c)
    w, h, rgb, _, _ = p.getCameraImage(480, 360, view, proj,
                                       renderer=p.ER_TINY_RENDERER, physicsClientId=c)
    return np.reshape(np.array(rgb, dtype=np.uint8), (h, w, 4))[:, :, :3]


def _save_gif(frames, path: Path):
    try:
        import imageio.v2 as imageio
    except ImportError:
        raise SystemExit("GIF export needs imageio:  pip install imageio")
    # 20 fps matches the control rate, so the GIF plays at real-time speed.
    imageio.mimsave(str(path), frames, fps=20, loop=0)


if __name__ == "__main__":
    main()
