"""THE ENVIRONMENT, AGAIN — but this time it is a real robot.

This file implements the SAME interface as `training/env.py`:

    obs, info        = env.reset()
    obs, reward, ... = env.step(action)
                       env.close()

Put the two files side by side. They import the same constants from
`contract.py`, call the same `apply_action`, and assemble the observation with
the same `build_observation`. The policy cannot tell them apart, which is the
entire point -- a network trained against one runs unmodified against the
other.

--------------------------------------------------------------------------
WHAT IS DIFFERENT, AND WHY IT MATTERS
--------------------------------------------------------------------------
1. THERE IS NO REWARD. `step()` returns 0.0 forever. Reward is a training
   signal; nothing is learning here. On hardware you could not compute it
   anyway -- it needs the robot's true height and velocity, which nothing on
   board can measure. This is worth sitting with: the reward function you
   spend weeks tuning does not exist at deployment.

2. TIME IS REAL AND CANNOT BE PAUSED. The simulator advances exactly 50 ms per
   step because it is told to. Here, 50 ms is 50 ms, and if the loop overruns,
   the robot has already carried on moving during the overrun. `step()` paces
   itself and reports what it actually achieved.

3. THE ROBOT CANNOT BE RESET. `reset()` in simulation teleports the robot to
   the start. Here, it ramps the servos into a standing pose -- and if the
   robot has fallen over, that means a human picks it up. There is no such
   thing as a free episode.

4. SENSOR READINGS GO STALE. If no new IMU reading has arrived, the previous
   one is held rather than substituting zeros. Zeros would be a lie the policy
   has never seen -- an all-zero orientation is not a pose a robot can be in,
   and it would confidently act on it.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import gymnasium as gym
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from contract import (  # noqa: E402
    CONTROL_DT,
    JOINT_HISTORY_LEN,
    NUM_JOINTS,
    OBSERVATION_SIZE,
    RESET_POSE_DEG,
    REST_POSE_DEG,
    apply_action,
    build_body_state,
    build_observation,
    normalize_joint_deg,
)
from link import RobotLink  # noqa: E402

# Extra degrees of clearance kept inside the measured joint limits. The limits
# in contract.py are where the leg touches the shell; stopping short of that
# means a servo never grinds against plastic. Set to 0 for full range once you
# trust the policy.
SAFETY_MARGIN_DEG = 15.0

# Refuse to keep driving on IMU data older than this. At ~40 Hz a healthy link
# delivers a reading every 25 ms; a tenth of a second means something is wrong.
MAX_IMU_AGE_S = 0.1

# The reset ramp: how far to move per micro-step, and how long to pause.
RAMP_STEP_DEG = 5.0
RAMP_DT = 1.0 / 40.0

# Within one 50 ms control step, stream this many interpolated sub-targets
# rather than one jump. See step() for why.
MICRO_STEPS = 8
MICRO_WINDOW = 0.8  # use 80% of the period, leaving room for the loop itself


class BittleRobotEnv(gym.Env):
    """A real Bittle, wearing the same interface as the simulator."""

    def __init__(self, port: str | None = None, safety_margin_deg: float = SAFETY_MARGIN_DEG):
        self.link = RobotLink(port)
        self.margin = float(safety_margin_deg)
        self.joint_deg = RESET_POSE_DEG.copy()
        self.body_state = np.zeros(6)
        self.joint_history = np.zeros(JOINT_HISTORY_LEN * NUM_JOINTS)
        self.step_count = 0
        self.yaw_zero: float | None = None

        # Identical to the simulator's spaces. If these ever disagree, the
        # policy silently misreads its own inputs.
        self.action_space = gym.spaces.Box(-1.0, 1.0, (NUM_JOINTS,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-1.0, 1.0, (OBSERVATION_SIZE,), dtype=np.float32)

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        """Put the robot into the standing pose the policy expects at step 0.

        Two stages, and the order matters. First `d` drops the robot into the
        firmware's rest crouch, so we know exactly where the legs are
        regardless of how the last run ended. Then ramp gradually from there
        to the standing pose.

        Do not skip the ramp. Commanding the stand directly from an unknown
        pose makes every servo slam toward its target at full speed at once,
        which shoves the robot across the floor and can strip a gear.
        """
        super().reset(seed=seed)
        self.step_count = 0
        self.yaw_zero = None

        self.link.rest()
        time.sleep(0.8)  # the firmware's rest motion is slow and blocking
        self._ramp(REST_POSE_DEG, RESET_POSE_DEG)
        time.sleep(0.3)  # let the servos settle before the first sensor read

        self.joint_history = np.tile(normalize_joint_deg(self.joint_deg), JOINT_HISTORY_LEN)
        self._refresh_body_state()
        return build_observation(self.body_state, self.joint_history), {}

    def _ramp(self, start_deg, target_deg) -> None:
        """Walk the servos from one pose to another a few degrees at a time."""
        start, target = np.asarray(start_deg, float), np.asarray(target_deg, float)
        steps = max(1, int(np.ceil(np.max(np.abs(target - start)) / RAMP_STEP_DEG)))
        for i in range(1, steps + 1):
            self.joint_deg = np.round(start + (i / steps) * (target - start))
            self.link.send_joints(self._clip(self.joint_deg))
            time.sleep(RAMP_DT)
        self.joint_deg = target.copy()

    def _clip(self, angles_deg: np.ndarray) -> np.ndarray:
        """Apply the extra safety margin on top of the contract's limits."""
        from contract import LIMIT_HIGH_DEG, LIMIT_LOW_DEG
        return np.clip(angles_deg, LIMIT_LOW_DEG + self.margin, LIMIT_HIGH_DEG - self.margin)

    # ------------------------------------------------------------------
    def step(self, action):
        """One policy decision, executed on real servos over real 50 ms."""
        started = time.time()
        previous = self.joint_deg.copy()

        # Same function the simulator uses. This is the contract doing its job.
        self.joint_deg = self._clip(apply_action(self.joint_deg, action))

        # Stream the move as several small targets instead of one jump.
        #
        # A real servo covers a 5-degree move in well under 50 ms, so a single
        # packet per step means it snaps to the new angle and then sits still
        # for 40 ms -- a stuttering, twitchy gait. The simulator meanwhile
        # tracks the target smoothly across the whole step. Interpolating here
        # closes that gap AND looks dramatically better on video.
        #
        # This changes nothing the policy sees: still one decision per 50 ms,
        # and the history records only the final target.
        for k in range(1, MICRO_STEPS + 1):
            fraction = k / MICRO_STEPS
            self.link.send_joints(np.round(previous + fraction * (self.joint_deg - previous)))
            if k < MICRO_STEPS:
                self._sleep_until(started + fraction * CONTROL_DT * MICRO_WINDOW)

        # Update the policy's memory of where the legs have been.
        if self.step_count % 2 == 0:
            self.joint_history = np.roll(self.joint_history, -NUM_JOINTS)
            self.joint_history[-NUM_JOINTS:] = normalize_joint_deg(self.joint_deg)

        stale = self._refresh_body_state()
        self.step_count += 1

        # Hold the loop to 20 Hz so the policy runs at the rate it was trained
        # for. A policy running at the wrong frequency produces the wrong gait:
        # its actions are per-decision deltas, so at double speed it moves the
        # joints twice as fast as it means to.
        self._sleep_until(started + CONTROL_DT)
        elapsed = time.time() - started

        return (
            build_observation(self.body_state, self.joint_history),
            0.0,          # no reward on hardware -- nothing is learning
            False,        # no automatic termination; a human decides
            False,
            {"step_time": elapsed, "imu_stale": stale, "joint_deg": self.joint_deg.copy()},
        )

    @staticmethod
    def _sleep_until(target_time: float) -> None:
        remaining = target_time - time.time()
        if remaining > 0:
            time.sleep(remaining)

    def _refresh_body_state(self) -> bool:
        """Pull the newest IMU reading. Returns True if it was too old to trust.

        On a stale or missing reading the PREVIOUS state is held. Holding a
        slightly old pose is a small lie; substituting zeros would claim the
        robot is perfectly level, which is a large one.
        """
        reading, age = self.link.read_imu()
        if reading is None or age > MAX_IMU_AGE_S:
            return True

        ax, ay, az, neg_yaw, pitch, roll = reading

        # Zero the yaw against wherever the robot happened to be pointing when
        # the episode started. The IMU has no compass, so its absolute yaw is
        # an arbitrary number that also drifts. (build_body_state discards yaw
        # entirely -- this just keeps the input sane.)
        if self.yaw_zero is None:
            self.yaw_zero = neg_yaw
        self.body_state = build_body_state(ax, ay, az, neg_yaw - self.yaw_zero, pitch, roll)
        return False

    # ------------------------------------------------------------------
    def close(self):
        self.link.close()
