"""THE CONTRACT — the single agreement between simulation and the real robot.

Read this file first. Everything else depends on it.

An RL policy is trained in a simulator and then run on hardware. Those are two
completely different pieces of software. The policy only transfers if they
agree, exactly, on three things:

    1. WHAT AN ACTION MEANS   -- the policy outputs 8 numbers. Which joint is
                                 number 0? Is +1 "bend" or "straighten"? How
                                 many degrees is it worth?
    2. WHAT AN OBSERVATION IS -- the policy reads 246 numbers. Which slot holds
                                 pitch? What are the units? What scaling?
    3. WHERE AN EPISODE STARTS -- the pose the robot is in at step 0.

If sim and hardware disagree on ANY of these, the policy will look perfect in
the simulator and fail on the robot, and you will spend weeks blaming the
reward function. (We did. See README, "The sign bug".)

So those three things are defined ONCE, here, and both
`training/env.py` and `robot/env.py` import them. Neither is allowed to have
its own opinion. That is the whole idea: a shared contract makes a whole class
of sim-to-real bug impossible to write.

Every number below was measured on a real Bittle, not guessed.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

# ============================================================================
# 1. THE JOINTS
# ============================================================================
# A Bittle has 8 leg joints (plus a head servo we never touch). Each leg has an
# UPPER joint at the body and a LOWER joint at the knee:
#
#            front                                    back
#         shoulder_left  ---- torso ----  hip_left        <- upper joints
#              |                              |
#          elbow_left                     knee_left       <- lower joints
#              |                              |
#            paw                            paw
#
# "shoulder/elbow" = front legs, "hip/knee" = rear legs. Petoi's own naming.
#
# THE ORDER MATTERS AND IT IS NOT ALPHABETICAL. It is the order the firmware
# expects on the wire (servo IDs 8..15). The policy's action vector, the
# observation's joint history, and the serial packet all use THIS order.

NUM_JOINTS = 8

JOINT_NAMES = [
    "shoulder_left",   # 0  servo 8   front-left  upper
    "shoulder_right",  # 1  servo 9   front-right upper
    "hip_right",       # 2  servo 10  rear-right  upper
    "hip_left",        # 3  servo 11  rear-left   upper
    "elbow_left",      # 4  servo 12  front-left  lower
    "elbow_right",     # 5  servo 13  front-right lower
    "knee_right",      # 6  servo 14  rear-right  lower
    "knee_left",       # 7  servo 15  rear-left   lower
]

# The simulator loads the robot model from a URDF file. PyBullet numbers the
# joints in the order it walks the model's tree, which is NOT the order above.
# So the sim looks joints up BY NAME using JOINT_NAMES and rebuilds the
# mapping. Never index sim joints positionally.
#
# (This was a real bug: 6 of 8 commands went to the wrong leg for weeks. The
# robot moved, it just moved wrong, so nothing crashed to tell us.)

# ----------------------------------------------------------------------------
# Sign convention
# ----------------------------------------------------------------------------
# The firmware stores a +1/-1 flip per servo, because the physical servos are
# mounted facing opposite ways on the left and right sides. It applies the flip
# itself, in `motion.h`:
#
#     duty = calibratedZeroPosition[i] + angle * rotationDirection[i]
#
# So a command we send is in the PRE-flip frame. The practical consequence:
#
#     the SAME number sent to a left/right pair produces a MIRRORED pose.
#
# That is why the standing pose below is [30, 30, 30, ...] and not
# [30, -30, 30, ...]. The simulator must apply the identical flip before
# driving its model, or sim and reality mirror each other.

ROTATION_DIRECTION = np.array([1, -1, -1, 1, -1, 1, 1, -1], dtype=np.float64)


# ============================================================================
# 2. THE ACTION
# ============================================================================
# The policy outputs 8 floats in [-1, +1]. They are NOT joint angles. They are
# *changes* to the current joint angles:
#
#     new_angle = current_angle + action * STEP_ANGLE_DEG
#
# Why relative and not absolute? Two reasons worth understanding:
#
#   - It bounds how fast the robot can move. At 20 decisions/second and 5
#     degrees per decision, a joint can slew at most 100 deg/s. Real servos
#     physically cannot track faster than about 240 deg/s, so an absolute
#     action space lets the policy learn a gait that the hardware silently
#     fails to execute. Relative actions make that unlearnable.
#   - Smooth motion becomes the default. Outputting 0 means "hold". With an
#     absolute action space, "hold" requires the network to reproduce the exact
#     current angle every step, and small errors turn into jitter.
#
# The cost: the policy needs memory of where the joints are, which is why the
# observation carries a joint history (see below).

STEP_ANGLE_DEG = 5.0
CONTROL_HZ = 20.0
CONTROL_DT = 1.0 / CONTROL_HZ  # 50 ms per decision

# ----------------------------------------------------------------------------
# Limits: where the legs actually stop
# ----------------------------------------------------------------------------
# The firmware ships a permissive software limit (roughly +-200 deg). That is
# NOT the real limit -- long before 200 degrees the leg slams into the robot's
# own plastic shell and the servo stalls, buzzing and drawing current.
#
# These were measured by hand: jog one joint a degree at a time until the leg
# touched the body, write the number down. Both sim and hardware clip HERE, so
# the policy never explores a pose that reality refuses to produce.
#
# (If sim allows a pose that hardware clips, the policy will happily build a
# gait on top of it and that gait cannot transfer. Clipping identically on both
# sides is the fix.)

LIMIT_LOW_DEG = np.array([-125, -125, -35, -35, -60, -60, -45, -45], dtype=np.float64)
LIMIT_HIGH_DEG = np.array([+70, +70, +125, +125, +125, +125, +125, +125], dtype=np.float64)

# The serial packet carries ONE SIGNED BYTE per joint, so nothing outside
# [-128, 127] can physically be transmitted. Clip to it as well, so a widened
# limit can never silently overflow into a wrong angle.
PACKET_LOW_DEG, PACKET_HIGH_DEG = -128.0, 127.0


def apply_action(current_deg: np.ndarray, action: np.ndarray) -> np.ndarray:
    """Turn a policy action into the next joint-angle command, in degrees.

    Both the simulator and the real robot call THIS function. That is the
    point: "what an action means" has exactly one definition.

    Angles are rounded to whole degrees because the serial packet is integer
    bytes -- so the simulator trains on the same quantized angles the robot
    will actually receive, rather than on smooth floats it can never get.
    """
    target = current_deg + np.asarray(action, dtype=np.float64) * STEP_ANGLE_DEG
    target = np.clip(target, LIMIT_LOW_DEG, LIMIT_HIGH_DEG)
    target = np.clip(target, PACKET_LOW_DEG, PACKET_HIGH_DEG)
    return np.round(target)


# ============================================================================
# 3. THE OBSERVATION
# ============================================================================
# 246 numbers, all roughly in [-1, 1], assembled as:
#
#     [ 6 body-state values ] + [ 30 past joint poses x 8 joints ]
#     |___________________|     |_________________________________|
#       where the body is         where the legs have been
#
# What the robot can actually sense is worth being blunt about, because it
# shapes everything:
#
#   - It has an IMU. That gives orientation (pitch/roll) and acceleration.
#   - It has NO joint encoders. The servos cannot report their true position.
#   - It has NO foot contact sensors. It cannot feel the ground.
#   - It has NO camera in this setup, and no idea where it is in the room.
#
# So the "joint history" is a log of what we COMMANDED, not what the legs did.
# If a servo stalls against the floor, the observation never mentions it. The
# policy has to be robust to that, and this is the single biggest reason a
# simulator-trained gait degrades on hardware.
#
# Why keep 30 steps of history instead of just the current pose? Because the
# policy is a plain feed-forward network with no memory of its own. Walking is
# periodic -- to know you are mid-stride you must know what you did recently.
# The history is the memory, stapled onto the input.

JOINT_HISTORY_LEN = 30
BODY_STATE_SIZE = 6
OBSERVATION_SIZE = JOINT_HISTORY_LEN * NUM_JOINTS + BODY_STATE_SIZE  # 246

# Scale factor on acceleration, to bring m/s^2 into roughly [-1, 1].
# Neural networks train badly on inputs of wildly different magnitude: a raw
# value of 9.8 sitting next to a quaternion component of 0.03 means the first
# dominates the initial gradients purely because of units.
ACCEL_SCALE = 0.012


def normalize_joint_deg(angles_deg: np.ndarray) -> np.ndarray:
    """Map joint angles into [-1, 1] for the network, per joint."""
    envelope = np.maximum(np.abs(LIMIT_LOW_DEG), np.abs(LIMIT_HIGH_DEG))
    return np.clip(angles_deg / envelope, -1.0, 1.0)


def build_body_state(
    ax: float, ay: float, az: float,
    neg_yaw_deg: float, pitch_deg: float, roll_deg: float,
) -> np.ndarray:
    """Pack the 6 body-state values from raw IMU numbers.

    Takes exactly the six values the firmware prints. The simulator computes
    fake versions of those same six from physics and calls this identical
    function, so both sides produce a bit-identical observation for the same
    physical pose.

    Returns [quat_x, quat_y, quat_z, quat_w, ax*scale, ay*scale].

    NOTE ON YAW -- it is deliberately thrown away.
    Orientation is passed to the network as a quaternion rather than as
    Euler angles, because Euler angles jump discontinuously (179 deg -> -179
    deg is a tiny physical rotation but a huge input change) and a network
    cannot easily learn across that seam.
    But the quaternion here encodes pitch and roll ONLY. The real IMU's yaw
    drifts about 10 degrees per second with nothing to correct it -- there is
    no compass. Within one episode the robot would come to believe it is
    turning when it is driving straight. The simulator has perfect yaw and
    would never show the policy that lie. So rather than train against a
    signal we cannot trust, we zero it on both sides.
    That is a general sim-to-real move: if a sensor is unreliable on hardware,
    do not let the policy depend on it, even though the simulator could
    provide it perfectly.
    """
    quat_xyzw = Rotation.from_euler(
        "zyx", [0.0, pitch_deg, roll_deg], degrees=True
    ).as_quat()
    accel_xy = np.clip(np.array([ax, ay]) * ACCEL_SCALE, -1.0, 1.0)
    return np.concatenate((quat_xyzw, accel_xy)).astype(np.float64)


def build_observation(body_state: np.ndarray, joint_history: np.ndarray) -> np.ndarray:
    """Assemble the final 246-vector handed to the policy network."""
    return np.hstack((body_state, joint_history)).astype(np.float32)


# ============================================================================
# 4. WHERE AN EPISODE STARTS
# ============================================================================
# Every episode begins with the robot ALREADY STANDING, in the pose below.
#
# This is a deliberate narrowing of the problem. Learning to stand up from
# lying down and learning to walk are two different skills, and asking one
# policy to discover both from scratch makes the early exploration far harder:
# almost every random action sequence ends in a heap on the floor, so the
# policy sees almost no reward signal about walking at all.
#
# On hardware the robot does not stand up by itself either -- `robot/env.py`
# ramps the servos from the rest pose into this stand before handing control
# to the policy. Sim and hardware start from the same place.

STAND_POSE_DEG = np.array([30, 30, 30, 30, 30, 30, 30, 30], dtype=np.float64)
RESET_POSE_DEG = STAND_POSE_DEG.copy()

# Height of the shoulder axles above the floor in that stand, ruler-measured
# on the real robot. The reward uses this as the target height.
STAND_HEIGHT_M = 0.08

# The firmware's own built-in "rest" crouch, which `d` puts the robot into.
# Hardware resets ramp FROM here TO the stand, so the legs never jump.
REST_POSE_DEG = np.array([75, 75, 75, 75, -55, -55, -55, -55], dtype=np.float64)


# ============================================================================
# Self-check: run `python contract.py` to verify the contract is coherent.
# ============================================================================
if __name__ == "__main__":
    assert len(JOINT_NAMES) == NUM_JOINTS == len(set(JOINT_NAMES))
    assert OBSERVATION_SIZE == 246
    assert np.all(LIMIT_LOW_DEG < LIMIT_HIGH_DEG)
    assert set(np.abs(ROTATION_DIRECTION)) == {1.0}

    # The stand pose must be reachable -- a start pose outside the limits
    # would be silently clipped and the robot would start somewhere else.
    assert np.all(STAND_POSE_DEG >= LIMIT_LOW_DEG)
    assert np.all(STAND_POSE_DEG <= LIMIT_HIGH_DEG)

    # An action of 0 must hold position, or "do nothing" is not expressible.
    assert np.array_equal(apply_action(STAND_POSE_DEG, np.zeros(8)), STAND_POSE_DEG)

    # Full-throttle actions must move by exactly one step and stay in bounds.
    up = apply_action(STAND_POSE_DEG, np.ones(8))
    assert np.all(up <= LIMIT_HIGH_DEG) and np.all(up >= LIMIT_LOW_DEG)
    assert np.array_equal(up, np.minimum(STAND_POSE_DEG + STEP_ANGLE_DEG, LIMIT_HIGH_DEG))

    # Actions must never escape the transmittable byte range.
    far = apply_action(np.full(8, 127.0), np.ones(8))
    assert np.all(far >= PACKET_LOW_DEG) and np.all(far <= PACKET_HIGH_DEG)

    # A level, motionless robot: quaternion is identity, no lateral accel.
    level = build_body_state(0.0, 0.0, 9.81, 0.0, 0.0, 0.0)
    assert level.shape == (BODY_STATE_SIZE,)
    assert np.allclose(level, [0, 0, 0, 1, 0, 0], atol=1e-9), level

    # Nose-down pitch must show up in the observation, or the policy is blind
    # to the one thing that matters most for not falling over.
    tilted = build_body_state(0.0, 0.0, 9.81, 0.0, -20.0, 0.0)
    assert abs(tilted[1]) > 0.15, tilted

    obs = build_observation(level, np.tile(normalize_joint_deg(STAND_POSE_DEG), 30))
    assert obs.shape == (OBSERVATION_SIZE,) and obs.dtype == np.float32

    print(f"contract OK  |  {NUM_JOINTS} joints  "
          f"|  obs {OBSERVATION_SIZE}  |  {CONTROL_HZ:.0f} Hz  "
          f"|  {STEP_ANGLE_DEG:.0f} deg/step -> "
          f"{STEP_ANGLE_DEG * CONTROL_HZ:.0f} deg/s max slew")
