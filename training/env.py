"""THE ENVIRONMENT — a simulated Bittle, in PyBullet.

In reinforcement learning the ENVIRONMENT is everything that is not the agent.
It holds the world's state, accepts an action, advances time, and reports back
an observation and a reward. It is the thing being learned about.

The interface is three methods, and it is the same interface `robot/env.py`
implements with a real robot on the other end:

    obs, info          = env.reset()          start a new episode
    obs, reward, ...   = env.step(action)     take one action, advance 50 ms
                         env.close()          shut down

That is the whole API an RL algorithm needs. The agent never knows whether
there is physics or a serial cable behind it -- which is exactly why a policy
trained here can be run there.

Formally this is a Markov Decision Process (MDP): at each step the agent sees a
STATE, picks an ACTION, and receives a REWARD plus a new state. "Markov" means
the state should contain everything needed to decide -- which is why the
observation carries 30 steps of joint history rather than just the current
pose. See `contract.py` for the full state/action definition.

WHY SIMULATE AT ALL? Training takes about 10 million steps. At the robot's real
control rate of 20 Hz that would be 139 hours of continuous walking, on servos
that overheat and a battery that lasts 45 minutes, with a human resetting the
robot after every fall. The simulator runs those same 10 million steps in
around 35 minutes across parallel worlds. The entire difficulty of this field
is that the simulator is not reality, and the gap is where policies die.
"""

from __future__ import annotations

import sys
from pathlib import Path

import gymnasium as gym
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from contract import (  # noqa: E402
    CONTROL_DT,
    JOINT_HISTORY_LEN,
    JOINT_NAMES,
    NUM_JOINTS,
    OBSERVATION_SIZE,
    RESET_POSE_DEG,
    ROTATION_DIRECTION,
    STAND_HEIGHT_M,
    apply_action,
    build_body_state,
    build_observation,
    normalize_joint_deg,
)

DEFAULT_URDF = Path(__file__).resolve().parent / "model" / "bittle.urdf"

# ============================================================================
# SIMULATION SETTINGS
# ============================================================================
GRAVITY = 9.81
PHYSICS_DT = 1.0 / 240.0        # PyBullet's internal tick
SUBSTEPS = 12                   # 12 ticks x 1/240 s = 50 ms = one agent decision
EPISODE_STEPS = 200             # 200 decisions x 50 ms = 10 seconds of robot time

# How hard the simulated servos push, in newton-metres. This is a DELIBERATE
# UNDERESTIMATE of the real P1S servo's stall torque.
#
# An earlier version used a higher value and the sim held poses rigidly. On the
# real floor the servos sagged under the robot's weight, the lower legs drooped
# below where they were commanded, and the knees scraped the ground -- a gait
# that was clean in sim and scraping in reality. Rather than pretend the real
# servos are stronger, we made the simulated ones weaker, so the policy learns
# poses that still work once everything sags a little.
#
# This is the standard sim-to-real move: when reality is worse than your model,
# degrade the model rather than hoping. Same reason the joint speed below is
# capped at the servo's real limit instead of PyBullet's infinite default.
JOINT_TORQUE = 0.14
MAX_JOINT_SPEED_RAD_S = 4.2     # ~240 deg/s, the P1S servo's unloaded top speed


# ============================================================================
# THE REWARD
# ============================================================================
# The reward function is the ONLY place you say what "good" means. The policy
# will maximize exactly what is written here -- not what you meant. Every term
# below exists because a previous version of this file was exploited by a
# policy that found a cheaper way to score.
#
# The goal: walk forward, slowly, while standing up.
#
# Read the history of failures in the README; the short version is that a
# quadruped will always rather sprawl on its belly and drag itself than balance
# on four legs, because balancing is hard and dragging is not.

TARGET_SPEED_MS = 0.08          # 8 cm/s. Slow is deliberate -- see below.
MAX_SPEED_MS = 0.12             # above this, penalized
SPEED_SIGMA = 0.003             # width of the speed reward bell curve
HEIGHT_SIGMA = 0.0006           # width of the height bell curve (~2.4 cm)
SPEED_FILTER = 0.2              # smoothing on measured speed, see step()

# What counts as having fallen over. Crossing either ends the episode.
FALL_TILT_RAD = 0.9             # ~52 degrees of roll or pitch
FALL_HEIGHT_M = 0.045           # body this low means lying down, not walking

# Weights. Tuning these is most of the practical work in applied RL.
W_ALIVE = 0.15                  # small payment per step for not having fallen
W_FORWARD = 2.0                 # move at the target speed
W_HEIGHT = 1.0                  # stay at standing height
W_REVERSE = 1.0                 # do not go backwards
W_OVERSPEED = 1.5               # do not lunge
W_TILT = 2.0                    # keep the body level
W_SPIN = 0.2                    # do not rotate wildly
W_BOUNCE = 0.1                  # do not bob up and down
W_SIDEWAYS = 0.5                # do not crab sideways
W_YAW = 0.1                     # do not turn off course
W_FOOT_SLIP = 0.5               # plant the feet, do not skate
W_THIGH_GROUND = 0.5            # upper legs should not touch the floor
W_SHIN_GROUND = 2.0             # lower legs REALLY should not touch the floor
W_BELLY_GROUND = 2.0            # the torso must never drag
W_EFFORT = 0.03                 # prefer small actions
W_JERK = 0.2                    # prefer smooth actions
W_SMOOTH_1 = 0.2                # penalize joint velocity
W_SMOOTH_2 = 0.05               # penalize joint acceleration
FALL_PENALTY = 100.0            # one-time cost of falling over

# Which parts of the robot are allowed to touch the ground. Only paws.
PAWS = {"left_front_paw", "right_front_paw", "left_back_paw", "right_back_paw"}
THIGHS = {"left_upper_arm", "right_upper_arm", "left_upper_leg", "right_upper_leg"}
SHINS = {"left_lower_arm", "right_lower_arm", "left_lower_leg", "right_lower_leg"}


def compute_reward(*, speed, height, roll, pitch, yaw, spin_xy, bounce, sideways,
                   foot_slip, thigh_ground, shin_ground, belly_ground,
                   effort, jerk, smoothness, fallen) -> tuple[float, dict]:
    """Score one step. Returns (total, breakdown-for-inspection).

    The breakdown is returned so you can see WHICH term the policy is actually
    collecting. When a policy does something strange, print this dict -- the
    answer is almost always that one term is paying far more than intended.

    THE KEY IDEA IN THIS FUNCTION is the `height_gate`. Notice that the forward
    reward is MULTIPLIED by it, not added to it:

        forward_reward = W_FORWARD * height_gate * speed_match

    An earlier version added them. The policy immediately discovered it could
    flop onto its belly, give up the height reward entirely, and shovel itself
    along with its legs -- collecting full forward reward while lying down.
    That scored better than walking, because walking risks falling.

    Multiplying makes the two goals inseparable: at belly height the gate is
    near zero, so forward progress is worth near zero no matter how fast you
    go. You cannot buy speed by giving up posture. This trick -- gating a
    reward on a precondition instead of adding it -- is worth remembering.
    """
    # Bell curve: 1.0 at exactly the target, falling off smoothly either side.
    # Smooth beats a threshold here, because a threshold gives zero gradient
    # information -- the policy learns nothing from "not there yet".
    height_gate = np.exp(-((height - STAND_HEIGHT_M) ** 2) / HEIGHT_SIGMA)
    speed_match = np.exp(-((speed - TARGET_SPEED_MS) ** 2) / SPEED_SIGMA)

    spin_x, spin_y = spin_xy
    parts = {
        "alive": W_ALIVE,
        "forward": W_FORWARD * height_gate * speed_match,
        "height": W_HEIGHT * height_gate,
        "reverse": -W_REVERSE * max(-speed, 0.0),
        "overspeed": -W_OVERSPEED * max(speed - MAX_SPEED_MS, 0.0),
        "tilt": -W_TILT * (roll ** 2 + pitch ** 2),
        "spin": -W_SPIN * (spin_x ** 2 + spin_y ** 2),
        "bounce": -W_BOUNCE * bounce ** 2,
        "sideways": -W_SIDEWAYS * sideways ** 2,
        "yaw": -W_YAW * yaw ** 2,
        "foot_slip": -W_FOOT_SLIP * foot_slip,
        "thigh_ground": -W_THIGH_GROUND * thigh_ground,
        "shin_ground": -W_SHIN_GROUND * shin_ground,
        "belly_ground": -W_BELLY_GROUND * belly_ground,
        "effort": -W_EFFORT * effort,
        "jerk": -W_JERK * jerk,
        "smoothness": -smoothness,
        "fall": -FALL_PENALTY if fallen else 0.0,
    }
    return float(sum(parts.values())), parts


# ============================================================================
# THE ENVIRONMENT
# ============================================================================
class BittleSimEnv(gym.Env):
    """A simulated Bittle that speaks the standard Gymnasium env interface."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, urdf_path: Path | str = DEFAULT_URDF, gui: bool = False):
        import pybullet
        import pybullet_data

        self.p = pybullet
        self._pybullet_data = pybullet_data
        self.urdf_path = Path(urdf_path)
        if not self.urdf_path.exists():
            raise FileNotFoundError(f"Robot model not found: {self.urdf_path}")

        self.client = self.p.connect(self.p.GUI if gui else self.p.DIRECT)
        self.p.configureDebugVisualizer(self.p.COV_ENABLE_GUI, 0, physicsClientId=self.client)

        # ACTION SPACE: 8 numbers in [-1, 1]. See contract.apply_action for
        # what they mean. Bounded because policies output bounded values and
        # unbounded actions would let the network scream at the servos.
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(NUM_JOINTS,), dtype=np.float32
        )
        # OBSERVATION SPACE: 246 numbers, all normalized to roughly [-1, 1].
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(OBSERVATION_SIZE,), dtype=np.float32
        )

        self.robot = None
        self.joint_ids: list[int] = []
        self.link_ids: dict[str, int] = {}
        self.step_count = 0
        self.joint_deg = RESET_POSE_DEG.copy()
        self.prev_action = np.zeros(NUM_JOINTS)
        self.prev_world_velocity = np.zeros(3)
        self.smoothed_speed = 0.0
        self.joint_history = np.zeros(JOINT_HISTORY_LEN * NUM_JOINTS)
        self.recent_poses = np.zeros(3 * NUM_JOINTS)
        self.body_state = np.zeros(6)

    # ------------------------------------------------------------------
    # RESET — begin a new episode
    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.prev_action = np.zeros(NUM_JOINTS)
        self.prev_world_velocity = np.zeros(3)
        self.smoothed_speed = 0.0

        if self.robot is None:
            self._build_world()
        else:
            # Reloading the URDF every episode cost ~140 ms (about 44% of total
            # episode time, because the mesh files must be re-parsed). Teleporting
            # the existing robot back to the start is equivalent and free.
            self.p.resetBasePositionAndOrientation(
                self.robot, [0, 0, 0.085], [0, 0, 0, 1], physicsClientId=self.client
            )
            self.p.resetBaseVelocity(
                self.robot, [0, 0, 0], [0, 0, 0], physicsClientId=self.client
            )

        # Snap the legs into the standing pose. The robot spawns a few
        # millimetres up and settles onto its feet.
        self.joint_deg = RESET_POSE_DEG.copy()
        for joint_id, angle in zip(self.joint_ids, self._to_sim_frame(self.joint_deg)):
            self.p.resetJointState(self.robot, joint_id, angle, physicsClientId=self.client)

        # Fill the history buffer with the standing pose, so step 0 does not
        # look to the policy like it just teleported from zero.
        pose_norm = normalize_joint_deg(self.joint_deg)
        self.joint_history = np.tile(pose_norm, JOINT_HISTORY_LEN)
        self.recent_poses = np.tile(pose_norm, 3)
        self.body_state = build_body_state(*self._read_imu())
        return build_observation(self.body_state, self.joint_history), {}

    def _build_world(self):
        p, c = self.p, self.client
        p.resetSimulation(physicsClientId=c)
        p.setGravity(0, 0, -GRAVITY, physicsClientId=c)
        p.setAdditionalSearchPath(self._pybullet_data.getDataPath(), physicsClientId=c)
        p.loadURDF("plane.urdf", physicsClientId=c)
        self.robot = p.loadURDF(
            str(self.urdf_path), [0, 0, 0.085], [0, 0, 0, 1],
            # Self-collision ON: without it the legs pass straight through the
            # body and each other, and the policy learns gaits that are
            # geometrically impossible on a real robot.
            flags=p.URDF_USE_SELF_COLLISION | p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT,
            physicsClientId=c,
        )

        # Map joint NAME -> PyBullet index. PyBullet numbers joints in its own
        # tree order, which is not our order. Looking up by name is what keeps
        # command 0 going to the joint we think it does. (See contract.py.)
        name_to_id, self.link_ids = {}, {}
        for i in range(p.getNumJoints(self.robot, physicsClientId=c)):
            info = p.getJointInfo(self.robot, i, physicsClientId=c)
            self.link_ids[info[12].decode()] = i
            if info[2] in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
                name_to_id[info[1].decode()] = i
                p.changeDynamics(self.robot, i, maxJointVelocity=MAX_JOINT_SPEED_RAD_S,
                                 physicsClientId=c)

        missing = [n for n in JOINT_NAMES if n not in name_to_id]
        if missing:
            raise RuntimeError(f"Robot model is missing joints: {missing}")
        self.joint_ids = [name_to_id[n] for n in JOINT_NAMES]

    def _to_sim_frame(self, policy_deg: np.ndarray) -> np.ndarray:
        """Apply the firmware's left/right sign flip, then convert to radians.

        The real firmware does this multiplication itself before driving the
        servos. The simulator must do it too, or the two mirror each other.
        This one line is the difference between a gait that transfers and one
        that walks backwards into a wall. See contract.ROTATION_DIRECTION.
        """
        return np.deg2rad(policy_deg * ROTATION_DIRECTION)

    # ------------------------------------------------------------------
    # STEP — one action, 50 ms of simulated time
    # ------------------------------------------------------------------
    def step(self, action):
        action = np.asarray(action, dtype=np.float64)
        p, c = self.p, self.client

        # 1. ACTION -> joint targets, using the shared contract function.
        self.joint_deg = apply_action(self.joint_deg, action)

        # 2. Drive the simulated servos and advance physics 50 ms.
        #    POSITION_CONTROL means "go to this angle with at most this much
        #    torque" -- the same behaviour a hobby servo has.
        p.setJointMotorControlArray(
            self.robot, self.joint_ids, p.POSITION_CONTROL,
            targetPositions=self._to_sim_frame(self.joint_deg),
            forces=np.full(NUM_JOINTS, JOINT_TORQUE),
            physicsClientId=c,
        )
        for _ in range(SUBSTEPS):
            p.stepSimulation(physicsClientId=c)

        # 3. Update the joint history (the policy's memory).
        #    Only every other step, so 30 slots span 3 seconds of motion
        #    instead of 1.5 -- long enough to cover several strides.
        pose_norm = normalize_joint_deg(self.joint_deg)
        if self.step_count % 2 == 0:
            self.joint_history = np.roll(self.joint_history, -NUM_JOINTS)
            self.joint_history[-NUM_JOINTS:] = pose_norm
        self.recent_poses = np.roll(self.recent_poses, -NUM_JOINTS)
        self.recent_poses[-NUM_JOINTS:] = pose_norm

        # 4. Read the (simulated) sensors.
        self.body_state = build_body_state(*self._read_imu())
        (x, y, z), quat = p.getBasePositionAndOrientation(self.robot, physicsClientId=c)
        velocity, spin = p.getBaseVelocity(self.robot, physicsClientId=c)
        roll, pitch, yaw = p.getEulerFromQuaternion(quat)

        # 5. Measure forward speed IN THE ROBOT'S OWN HEADING, not world X.
        #    Otherwise a robot that turns 90 degrees and walks away is still
        #    scored on its world-X progress, and "turn then drift" beats
        #    "walk straight".
        forward = np.cos(yaw) * velocity[0] + np.sin(yaw) * velocity[1]
        sideways = -np.sin(yaw) * velocity[0] + np.cos(yaw) * velocity[1]
        # Low-pass filter: instantaneous speed is spiky from foot impacts, and
        # a spiky reward is a noisy learning signal.
        self.smoothed_speed = (SPEED_FILTER * forward
                               + (1 - SPEED_FILTER) * self.smoothed_speed)

        # 6. Score it.
        slip, thigh, shin, belly = self._contacts()
        now, prev, prev2 = (self.recent_poses[-NUM_JOINTS:],
                            self.recent_poses[NUM_JOINTS:2 * NUM_JOINTS],
                            self.recent_poses[:NUM_JOINTS])
        smoothness = float(np.sum(
            W_SMOOTH_1 * (now - prev) ** 2 + W_SMOOTH_2 * (now - 2 * prev + prev2) ** 2
        ))
        fallen = self._has_fallen(z, roll, pitch)
        reward, parts = compute_reward(
            speed=self.smoothed_speed, height=z, roll=roll, pitch=pitch, yaw=yaw,
            spin_xy=(spin[0], spin[1]), bounce=velocity[2], sideways=sideways,
            foot_slip=slip, thigh_ground=thigh, shin_ground=shin, belly_ground=belly,
            effort=float(np.sum(action ** 2)),
            jerk=float(np.sum((action - self.prev_action) ** 2)),
            smoothness=smoothness, fallen=fallen,
        )
        self.prev_action = action.copy()
        self.step_count += 1

        # TERMINATED means the episode ended because of something real (a fall).
        # TRUNCATED means it just ran out of time. RL algorithms treat these
        # differently: a truncated episode still has future value worth
        # estimating, a terminated one does not.
        return (
            build_observation(self.body_state, self.joint_history),
            reward,
            fallen,
            self.step_count >= EPISODE_STEPS,
            {"reward_parts": parts, "speed": self.smoothed_speed, "height": z},
        )

    def _has_fallen(self, z, roll, pitch) -> bool:
        # A grace period at the start, while the robot settles onto its feet
        # from the spawn a few millimetres above the floor.
        if self.step_count < 10:
            return False
        return bool(abs(roll) > FALL_TILT_RAD or abs(pitch) > FALL_TILT_RAD
                    or z < FALL_HEIGHT_M)

    def _read_imu(self):
        """Produce the same six numbers the real firmware would print.

        This function is pure sim-to-real plumbing and it is subtle.

        The real IMU is an accelerometer: it measures PROPER acceleration, not
        the Newtonian acceleration of the robot. Sitting perfectly still on a
        table it reads +9.81 upward -- because the table is pushing it up --
        not zero. Naively reporting the simulator's kinematic acceleration
        gives 0, 0, 0 at rest, so a stationary robot looks completely
        different to the policy in sim and on hardware.

        So: take the simulator's real acceleration, SUBTRACT gravity, and
        rotate into the robot's body frame. Now both sides read ~9.81 on the
        vertical axis while standing, and tilt shows up on ax/ay identically.
        """
        p, c = self.p, self.client
        _, quat = p.getBasePositionAndOrientation(self.robot, physicsClientId=c)
        velocity, _ = p.getBaseVelocity(self.robot, physicsClientId=c)
        velocity = np.asarray(velocity)

        accel_world = (velocity - self.prev_world_velocity) / CONTROL_DT
        self.prev_world_velocity = velocity

        world_to_body = np.asarray(p.getMatrixFromQuaternion(quat)).reshape(3, 3).T
        accel_body = world_to_body @ (accel_world - np.array([0, 0, -GRAVITY]))

        # The physical IMU chip is soldered onto the board facing backwards
        # relative to the robot model's forward axis. Measured, not assumed:
        # at a 7-degree nose-down tilt the hardware reports ax = -1.3 while the
        # model computes +1.0. One sign flip reconciles them.
        accel_body[0] = -accel_body[0]

        roll, pitch, yaw = p.getEulerFromQuaternion(quat)
        return (float(accel_body[0]), float(accel_body[1]), float(accel_body[2]),
                float(-np.rad2deg(yaw)), float(np.rad2deg(pitch)), float(np.rad2deg(roll)))

    def _contacts(self):
        """Find what is touching the floor. Only paws are supposed to.

        The robot has no contact sensors, so this information is available to
        the REWARD but never to the OBSERVATION. That is allowed and normal:
        during training you may use anything the simulator knows to shape
        behaviour, as long as the policy's *inputs* stay restricted to what the
        real robot can actually sense. Reward is a training-time signal; it
        does not exist at deployment.
        """
        p, c = self.p, self.client
        slip = thigh = shin = belly = 0.0

        # Link -1 is the torso itself.
        if p.getContactPoints(bodyA=self.robot, linkIndexA=-1, physicsClientId=c):
            belly += 1.0
        for name, idx in self.link_ids.items():
            if not p.getContactPoints(bodyA=self.robot, linkIndexA=idx, physicsClientId=c):
                continue
            if name in PAWS:
                # A planted foot should have near-zero horizontal velocity.
                # Anything else is skating, which looks like walking in sim and
                # goes nowhere on a real floor with different friction.
                link_velocity = p.getLinkState(self.robot, idx, computeLinkVelocity=1,
                                               physicsClientId=c)[6]
                slip += float(np.linalg.norm(link_velocity[:2]))
            elif name in SHINS:
                shin += 1.0
            elif name in THIGHS:
                thigh += 1.0
            elif name == "battery":
                belly += 1.0
        return slip, thigh, shin, belly

    def close(self):
        if self.p.isConnected(self.client):
            self.p.disconnect(self.client)


def _probe(**overrides):
    """Neutral reward inputs for the self-check, with fields overridden."""
    base = dict(speed=TARGET_SPEED_MS, height=STAND_HEIGHT_M, roll=0.0, pitch=0.0,
                yaw=0.0, spin_xy=(0.0, 0.0), bounce=0.0, sideways=0.0, foot_slip=0.0,
                thigh_ground=0.0, shin_ground=0.0, belly_ground=0.0, effort=0.0,
                jerk=0.0, smoothness=0.0, fallen=False)
    return {**base, **overrides}


# ============================================================================
# Self-check: run `python training/env.py` to verify the env behaves.
# ============================================================================
if __name__ == "__main__":
    env = BittleSimEnv()
    try:
        obs, _ = env.reset()
        assert obs.shape == (OBSERVATION_SIZE,), obs.shape
        assert env.observation_space.contains(obs), "reset obs outside declared space"

        # Standing still must be survivable and must score positively.
        # If a motionless robot falls over, the stand pose or the model is wrong.
        total = 0.0
        for _ in range(50):
            obs, reward, terminated, truncated, info = env.step(np.zeros(NUM_JOINTS))
            total += reward
            assert not terminated, "robot fell over while doing nothing"
        assert total > 0, f"standing still scored {total:.1f}, expected positive"

        # Sanity: it should be standing near the measured height, not sprawled.
        assert 0.06 < info["height"] < 0.11, f"stand height {info['height']:.3f} m"

        # The reward must actually be gated on height -- a belly-height robot
        # moving at the target speed must score far below a standing one.
        standing, _ = compute_reward(**_probe(height=STAND_HEIGHT_M))
        crawling, _ = compute_reward(**_probe(height=0.05))
        assert crawling < standing / 3, (
            f"height gate too weak: crawling {crawling:.2f} vs standing {standing:.2f}"
        )

        # Falling must never be profitable. The cheapest way to "cheat" is to
        # dive early and stop paying the running costs, so the fall penalty has
        # to outweigh everything a passive robot could bank by merely surviving.
        upright, _ = compute_reward(**_probe(height=STAND_HEIGHT_M))
        fell, _ = compute_reward(**_probe(height=STAND_HEIGHT_M, fallen=True))
        assert upright - fell == FALL_PENALTY
        assert FALL_PENALTY > W_ALIVE * EPISODE_STEPS, "diving beats standing still"

        print(f"env OK  |  50 still steps scored {total:+.1f}  "
              f"|  height {info['height']:.3f} m  "
              f"|  standing {standing:.2f} vs crawling {crawling:.2f}")
    finally:
        env.close()

