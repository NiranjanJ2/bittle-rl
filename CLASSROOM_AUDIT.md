# Classroom Audit — `bittle-rl` as a 5-week RL final project

Audited against: 35 juniors, post-AP-CSP, 4 weeks of hand-coded tabular Q-learning, **18 robots**, ~50-min periods, school laptops.

*Revised 2026-08-20: originally audited assuming one robot. 18 changes the logistics conclusion and introduces a new per-unit calibration risk — see Blocker 2.*

**Disclosure:** I wrote this codebase earlier in the same session, so treat this as a self-audit and weight the measured numbers over the judgments. Also, packaging changed *during* the audit at the user's instruction (Colab/local split, pinned deps, one shipped model). Sections below describe the repo **as it now stands**; where a finding was invalidated by that change I say so.

Every claim is tagged **[M]** measured by running it, or **[I]** inferred from reading code.

---

## Verdict

**Usable with specific modifications — not usable as-is.** The engineering is sound: the env is a clean Gymnasium implementation, the sim/hardware contract is genuinely shared rather than duplicated, runs are bit-for-bit reproducible **[M]**, and a hardware-verified policy ships so the robot walks on day one. The modification surface is also unusually clean — a student editing reward weights touches exactly one file, and nothing they'd break lives near the learning algorithm. But the flagship assignment does not work. I trained the "good" and "sabotaged" rewards to 1M steps and **the behavioral difference is not visible** (7.9 cm/5.6 cm·s⁻¹ vs 7.7 cm/6.3 cm·s⁻¹) **[M]**. The README's central exercise — break the reward, watch it crawl — needs a ~10M-step run students cannot afford, and the repo currently promises it will work at any budget. That's a full class period spent on an experiment that produces a shrug. With 18 robots the logistics are comfortable — ~2 students per unit, no servo-duty concern — but the constants in `contract.py` were measured on **one** Bittle and are now being asked to describe eighteen. Fix the exercise set (I've since replaced it with faster-acting ones, unverified), spot-check ROM across a few units, and this is a strong final project.

---

## 1. Environment construction

**Class and interface.** `BittleSimEnv`, `training/env.py:191`, subclasses `gymnasium.Env`. No custom base class. The hardware twin is `BittleRobotEnv`, `robot/env.py:85`, which also subclasses `gymnasium.Env` and implements the same three methods. **[M]** — `stable_baselines3.common.env_checker.check_env` passes (`python training/train.py --check` → `environment OK`).

**Signatures.**
```python
def reset(self, seed=None, options=None):   # env.py:235
    return build_observation(...), {}        # (ndarray(246,) float32, dict)

def step(self, action):                      # env.py:314
    return (observation,                     # ndarray(246,) float32
            reward,                          # python float
            fallen,                          # bool  — terminated
            self.step_count >= EPISODE_STEPS,# bool  — truncated
            {"reward_parts": parts, "speed": ..., "height": ...})
```
The `info` dict carrying `reward_parts` is the grading hook — see §4.

**Observation space.** `Box(low=-1.0, high=1.0, shape=(246,), dtype=float32)`, declared `training/env.py:216` and identically at `robot/env.py:100`. Size derives from `contract.py:188`: `30 × 8 + 6 = 246`. **[M]** the self-check asserts `observation_space.contains(obs)` at reset.

**Action space.** `Box(low=-1.0, high=1.0, shape=(8,), dtype=float32)`, `training/env.py:212`, mirrored `robot/env.py:99`.

**Termination vs truncation.** Correctly separated.
- **Terminated** = `_has_fallen()`, `training/env.py:392`: `|roll| > 0.9 rad` or `|pitch| > 0.9 rad` (≈52°) or `base_z < 0.045 m`. Suppressed for the first 10 steps while the robot settles from its spawn 5 mm above the floor.
- **Truncated** = `step_count >= EPISODE_STEPS` (200), i.e. 10 s.
- **Hardware returns `False, False` always** (`robot/env.py:186-188`). A fallen robot keeps receiving commands until `--steps` runs out or a human hits Ctrl-C. Deliberate (no reliable fall detector on hardware) but it is a safety item for the class. **[M]**

**Normalization / scaling / frame-skip.** All three exist and all live in `contract.py`:
- Action → degrees: `apply_action`, `contract.py:143`. Scales by `STEP_ANGLE_DEG=5`, clips to measured joint limits, clips to the signed-byte packet range, rounds to whole degrees.
- Joints → observation: `normalize_joint_deg`, `contract.py:197`, divides by each joint's own envelope.
- Accel → observation: `ACCEL_SCALE = 0.012`, `contract.py:194`.
- Frame skip: `SUBSTEPS = 12`, `training/env.py:63` — 12 physics ticks per decision.

**Three instantiation paths, two classes.**

| Path | Code | Wrapper |
|---|---|---|
| Training | `train.py:154` `make_vec_env(...)` | `SubprocVecEnv` (`DummyVecEnv` if `--envs 1`) |
| Evaluation | `train.py:165` | `Monitor(BittleSimEnv(...))`, single, non-parallel |
| Hardware | `robot/run.py:82` | none — `BittleRobotEnv` directly |

Training and eval share the same class; hardware is a separate class implementing the same interface. That's the right structure, and it's the thing that makes the sim→real story teachable.

---

## 2. The action loop

**Simulation trace**, one full step:

| # | Call | Location |
|---|---|---|
| 1 | `policy.predict(obs, deterministic=…)` → `ndarray(8,)` in [-1,1] | SB3 |
| 2 | `BittleSimEnv.step(action)` | `training/env.py:314` |
| 3 | `apply_action(self.joint_deg, action)` | `contract.py:143` |
| 4 | ↳ `× 5.0`, `np.clip` to limits, `np.clip` to ±127, `np.round` | `contract.py:157-160` |
| 5 | `_to_sim_frame(joint_deg)` → `deg2rad(deg × ROTATION_DIRECTION)` | `training/env.py:301` |
| 6 | `p.setJointMotorControlArray(..., POSITION_CONTROL, forces=0.14)` | `training/env.py:324` |
| 7 | `p.stepSimulation()` **× 12** | `training/env.py:331` |
| 8 | history roll, `_read_imu()`, `_contacts()`, `compute_reward()` | `env.py:336-377` |

**Transformation chain:** `[-1,1] → ×5° → +current → clip(measured ROM) → clip(±127) → round → ×(±1 sign flip) → radians → PyBullet target`.

**Control frequency.** 20 Hz, `contract.py:116-117` (`CONTROL_HZ = 20.0`). PyBullet's internal tick is 1/240 s (`training/env.py:62`). **12 physics steps per policy step.** Consistent: 12/240 = 50 ms. **[M]**

**Hardware trace** — diverges at step 5:

| # | Call | Location |
|---|---|---|
| 1-4 | identical, same `apply_action` | `contract.py:143` |
| 5 | **`self._clip(...)`** — extra safety margin, no sim equivalent | `robot/env.py:138,149` |
| 6 | loop of 8 interpolated micro-targets over 40 ms | `robot/env.py:162-166` |
| 7 | `link.send_joints()` → `b"Y" + struct.pack("8b",…) + b"~"` | `robot/link.py:131,144` |
| 8 | firmware applies `rotationDirection` itself, drives PWM | `motion.h:17` |

Two real divergences:
- **No sign flip in Python on hardware** — the firmware does it. Sim does it in `_to_sim_frame`. Net physical result is identical; the asymmetry is intentional but it's the kind of thing that confuses a reader.
- **The safety margin silently changes the shipped policy.** `SAFETY_MARGIN_DEG = 15.0` (`robot/env.py:69`) narrows every limit. **[M] I measured the shipped `walk_v9` policy against it: 27 of 200 steps (14%) are altered, by up to 4°, on both shoulders.** The default deployment path does not reproduce what the simulator showed, and nothing reports it. This is exactly the class of silent sim2real divergence the README warns about, present in the repo's own default. See Blockers.

---

## 3. The observation

| Index | Component | Source | Real sensor? |
|---|---|---|---|
| 0–3 | quaternion x,y,z,w — **pitch and roll only, yaw zeroed** | `build_body_state`, `contract.py:203` | **Yes**, IMU |
| 4 | `ax × 0.012` forward accel | IMU | **Yes** |
| 5 | `ay × 0.012` lateral accel | IMU | **Yes** |
| 6–245 | 30 past joint poses × 8, normalized; **newest at 238–245** | command log | **No — synthesized** |

**Sensors vs software.** Only indices 0–5 come from hardware. Indices 6–245 are a log of what was **commanded**, not measured — the robot has no joint encoders (`firmware/README.md` documents that the firmware's `j` command echoes the target table, not true position). No foot contact, no odometry, no camera. **[I]** from code + firmware docs; consistent with the project's own notes.

Yaw is deliberately discarded (`contract.py:220-235`) because the hardware DMP drifts ~10°/s with no compass, while sim has perfect yaw — training on it would be pure distribution shift.

**History buffer.** Maintained in `step()` at `training/env.py:337-340` (and `robot/env.py:169-171`). Stores `normalize_joint_deg(joint_deg)`, updated **every other step** (`if self.step_count % 2 == 0`), so 30 slots span 60 control steps = 3 s. A separate 3-slot `recent_poses` buffer updates every step and feeds only the smoothness penalty.

**Sim vs hardware identical?** Structurally yes — both call the same `build_body_state` and `build_observation`. Three real differences:
1. **Staleness.** Hardware holds the previous body state if no IMU sample newer than 100 ms arrived (`robot/env.py:205`); sim always has a fresh one.
2. **Yaw zeroing offset.** Hardware subtracts a per-episode `yaw_zero` (`robot/env.py:214`); sim doesn't need to. No observable effect since yaw is discarded anyway.
3. **Cold-start hazard.** `robot/env.py:92` initializes `body_state = np.zeros(6)`. A zero quaternion `(0,0,0,0)` is not a valid orientation and sim can never produce it. `RobotLink.__init__` waits for a first IMU packet before returning, so this should be unreachable — but nothing asserts it. **[I]**

---

## 4. The reward

One function: `compute_reward`, `training/env.py:134-182`. **18 terms.** Weights are a labeled block at `training/env.py:108-126`.

| Term | Weight | Measures |
|---|---|---|
| `alive` | `W_ALIVE` 0.15 | flat payment per surviving step |
| `forward` | `W_FORWARD` 2.0 | speed match **× height gate** |
| `height` | `W_HEIGHT` 1.0 | height gate alone |
| `reverse` | `W_REVERSE` 1.0 | backward motion |
| `overspeed` | `W_OVERSPEED` 1.5 | speed above 0.12 m/s |
| `tilt` | `W_TILT` 2.0 | roll² + pitch² |
| `spin` | `W_SPIN` 0.2 | angular velocity x,y |
| `bounce` | `W_BOUNCE` 0.1 | vertical velocity² |
| `sideways` | `W_SIDEWAYS` 0.5 | lateral velocity² |
| `yaw` | `W_YAW` 0.1 | heading error² |
| `foot_slip` | `W_FOOT_SLIP` 0.5 | paw horizontal speed while in contact |
| `thigh_ground` | `W_THIGH_GROUND` 0.5 | upper-leg ground contact |
| `shin_ground` | `W_SHIN_GROUND` 2.0 | lower-leg ground contact |
| `belly_ground` | `W_BELLY_GROUND` 2.0 | torso/battery contact |
| `effort` | `W_EFFORT` 0.03 | Σaction² |
| `jerk` | `W_JERK` 0.2 | Σ(action − prev_action)² |
| `smoothness` | `W_SMOOTH_1` 0.2, `W_SMOOTH_2` 0.05 | joint velocity + acceleration |
| `fall` | `FALL_PENALTY` 100.0 | one-time, on termination |

**Added vs gated.** 17 terms are additive. **One is multiplicative and it's the important one:**

```python
height_gate = np.exp(-((height - STAND_HEIGHT_M)**2) / HEIGHT_SIGMA)   # env.py:161
"forward": W_FORWARD * height_gate * speed_match,                       # env.py:167
```

`height_gate` acts as a **precondition on** `forward`, not as a separate addend — at belly height, forward progress is worth ~0 however fast. Note it *also* appears additively as the `height` term (`env.py:168`), so height is paid for twice, once as a gate and once as an addend. That dual role is the reason the sabotage exercise fails (see Feasibility).

**Easy to miss on a first read:**
- `FALL_PENALTY = 100` dwarfs everything; the self-check asserts it exceeds a full episode's alive bonus.
- `RESET_GRACE`: falls are ignored for the first 10 steps.
- `alive` is unconditional — standing still scores **+55.2 over 50 steps** **[M]**, a comfortable local optimum a policy can sit in for a long time.
- The 10-step grace also substitutes `TARGET_BASE_HEIGHT` for the real height in the reward during those steps.
- **`smoothness` is the one term whose weights are applied outside `compute_reward`** — `W_SMOOTH_1/2` multiply at the call site, `training/env.py:364-366`, and arrive pre-weighted as `"smoothness": -smoothness`. Editing the constant still works, but a student tracing the reward will not find the multiplication where the other 17 are.

**Spread?** The weighted sum is in one function. Its *ingredients* are computed in three places: `step()` (`env.py:359-377`), `_contacts()` (`env.py:437-470`), and the smoothness expression above.

**The grading artifact.** `training/watch.py` prints a per-term breakdown for one episode. Invocation and real output **[M]**:

```
$ python training/watch.py robot/policies/walk_v9.zip

200 steps   total reward +538.5   survived
final speed +8.2 cm/s   height 8.1 cm

reward term          total   per step
-------------------------------------
forward             +381.3     +1.907
height              +198.0     +0.990
alive                +30.0     +0.150
jerk                 -19.3     -0.097
effort               -19.2     -0.096
foot_slip            -12.1     -0.061
spin                 -11.9     -0.059
tilt                  -7.8     -0.039
smoothness            -0.2     -0.001
yaw                   -0.2     -0.001
sideways              -0.1     -0.001
bounce                -0.0     -0.000
```

Sorted by absolute contribution, zero terms suppressed. This is a genuinely good grading artifact — a student can point at it and say what their policy optimized. Add `--gif walk.gif` to also get a video (requires `imageio`).

---

## 5. The policy and training

**Library.** Stable-Baselines3, **2.8.0**, now pinned `==` in `requirements-train.txt` / `requirements-robot.txt`. **[M]** (It was `>=2.0` when the audit started; pinning happened mid-session.) PyTorch 2.x underneath — **note it is not listed in either requirements file**; it arrives as an SB3 transitive dependency, which means a ~900 MB download nobody declared.

**Architecture.** `MlpPolicy` with `net_arch=[256, 256]`, `training/train.py:174`. **[M]** from the saved weights:

```
mlp_extractor.policy_net.0  (256, 246)   mlp_extractor.value_net.0  (256, 246)
mlp_extractor.policy_net.2  (256, 256)   mlp_extractor.value_net.2  (256, 256)
action_net                  (8, 256)     value_net                  (1, 256)
log_std                     (8,)
```
Two hidden layers of 256, **separate actor and critic trunks** (not shared). Activation is SB3's default `Tanh` — not specified in the repo. **[I]**

**Explicitly-set hyperparameters** (everything else is SB3 default):

| Parameter | Value | Location |
|---|---|---|
| `seed` | 42 | `train.py:173` (`--seed`) |
| `policy_kwargs.net_arch` | `[256, 256]` | `train.py:174` |
| `n_steps` | `round(16384/envs/64)*64` | `train.py:180` |
| `learning_rate` | 1e-4 | `train.py:181` (`--lr`) |
| `target_kl` | 0.05 | `train.py:182` |
| `ent_coef` | 0.005 | `train.py:183` |
| `verbose` | 1 | `train.py:184` |
| `tensorboard_log` | `<out>/tensorboard` | `train.py:185` |

Left at default: `batch_size` 64, `n_epochs` 10, `gamma` 0.99, `gae_lambda` 0.95, `clip_range` 0.2, `vf_coef` 0.5, `max_grad_norm` 0.5.

**Checkpoints.** Two callbacks, `train.py:190-203`:
- `CheckpointCallback` → `<out>/checkpoints/ppo_<n>_steps.zip`, every 1M env-steps.
- `EvalCallback` → `<out>/best_model.zip`, rewritten whenever deterministic eval (3 episodes) improves, checked every 100k env-steps.
- `<out>/final_model.zip` at the end.

Format is SB3's zip. **[M]** contents: `data` (33 KB JSON config), `policy.pth` (1.0 MB), `policy.optimizer.pth` (2.1 MB), `pytorch_variables.pth`, `_stable_baselines3_version`, `system_info.txt`.

**Is the critic saved?** Yes — `value_net` weights live inside `policy.pth` alongside the actor, plus 2.1 MB of Adam optimizer state. **None of it is needed at inference**; `PPO.load()` restores it all anyway. A deployed policy is ~3 MB where ~1 MB would do.

**Loading.** Identical in both paths: `PPO.load(str(path))` then `policy.predict(obs, deterministic=True)` — `training/watch.py:39` and `robot/run.py:77-80`. `robot/run.py` additionally resolves a policy path relative to the repo root or `robot/`.

---

## 6. The simulation-to-hardware boundary

**What's shared, and how.** A single module, `contract.py`, imported by both `training/env.py:41` and `robot/env.py:51`. It carries joint order, `ROTATION_DIRECTION`, joint limits, packet bounds, `apply_action`, `normalize_joint_deg`, `build_body_state`, `build_observation`, and the reset poses. No duplicated constants, no config file, no codegen. **[M]** — the only import-time divergence is that `robot/` never loads PyBullet (verified: `pybullet not in sys.modules` after importing all three robot modules).

**What could still drift.** Three things:
1. **`SAFETY_MARGIN_DEG`** (`robot/env.py:69`) exists only on the hardware side and has no simulator counterpart. Measured to alter 14% of the shipped policy's steps. This is real drift, in the repo, today.
2. **`MICRO_STEPS`/`MICRO_WINDOW`** intra-step interpolation is hardware-only. Defensible (it *reduces* the gap) but it is behavior sim doesn't model.
3. **Episode length.** Sim truncates at 200 steps; hardware never truncates and `run.py` defaults to 400. A policy is being run twice as long as any episode it trained on. Probably harmless for a periodic gait; unverified.

**Self-tests.** No test framework — each file has an `assert`-based `__main__` block. **[M] All pass:**

| Command | Verifies | Result |
|---|---|---|
| `python contract.py` | joint count/uniqueness, obs size 246, limits ordered, stand pose within limits, zero action holds, full action moves exactly one step and stays in bounds, packet bounds hold, level robot → identity quaternion, pitch appears in obs | `contract OK … 5 deg/step -> 100 deg/s max slew` |
| `python training/env.py` | reset obs shape and space membership, robot doesn't fall doing nothing for 50 steps, positive score, stand height plausible, **height gate makes crawling < ⅓ of standing**, fall penalty exceeds full-episode alive bonus | `env OK \| 50 still steps scored +55.2 \| height 0.092 m \| standing 3.15 vs crawling 0.82` |
| `python training/train.py --check` | SB3 `check_env` | `environment OK` |
| `python robot/link.py` | needs hardware; prints IMU rows + worst sample age | not run — no robot attached |

Coverage is decent for the contract and thin for the env internals (nothing tests the history buffer indexing, `_contacts` classification, or the IMU gravity math).

**Hardware-specific conventions.**

| Convention | Where applied |
|---|---|
| L/R sign flip `[1,-1,-1,1,-1,1,1,-1]` | `contract.py:90`; sim applies it in `_to_sim_frame` (`env.py:301`), hardware lets **firmware** apply it (`motion.h:17`) |
| Joint order = firmware servo IDs 8–15, **not** URDF tree order | `contract.py:54`; sim rebuilds by name lookup, `env.py:288-299` |
| Measured ROM (tighter than firmware's software limits) | `contract.py:134-135` |
| Signed-byte packet bound ±127 | `contract.py:140`, enforced `link.py:142` |
| IMU +x mounted backwards → `accel_body[0] = -accel_body[0]` | `training/env.py:428` |
| Gravity added back into sim accel (real IMUs read +9.81 at rest) | `training/env.py:424` |
| Yaw discarded (DMP drift) | `contract.py:236` |
| Per-episode yaw zeroing | `robot/env.py:212-214` |
| Servo torque *under*-modeled at 0.14 N·m to mimic sag | `training/env.py:79` |

No calibration offsets in Python — those live in firmware EEPROM (`calibratedZeroPosition`).

---

## Feasibility

### Runtime **[M] measured, extrapolated to Colab [I]**

Measured on **Apple M4, 10 cores**, uncontended, wall clock including startup and eval:

| Config | Throughput | 1M steps |
|---|---|---|
| `--envs 2` | 1,289 steps/s | ~13 min |
| `--envs 10` | 1,908 steps/s | ~9 min |
| `--envs 5` ×2 concurrent | ~1,750 steps/s each | 579 s **[M]** actual |

Scaling is poor — 5× the envs buys 1.5× throughput, so this is dominated by per-step Python/PyBullet overhead, not parallelism. **Practical consequence: `--envs 2` is nearly as good as `--envs 10`, which is convenient because Colab free tier gives ~2 cores.**

Colab free-tier cores are roughly 2–3× slower per-core than an M4 **[I]**, so estimate **~450–650 steps/s**:

| Steps | Colab estimate | Behavior |
|---|---|---|
| 300k | 8–11 min | stands, shuffles |
| 1M | 26–37 min | walks slowly **[M]** at 5.6 cm/s |
| 10M | 4.3–6.2 h | shipped-policy quality (7.4 cm/s) |

10M in one Colab free session is not realistic (idle disconnects). A school laptop is likely comparable to or slower than Colab.

### Minimum viable step budget **[M] — and the answer is bad news**

I ran the experiment the README proposes. Two 1M-step runs, identical seed and settings, differing only in whether the height gate multiplies the forward term. Both evaluated in the **same** standard environment (the reward numbers aren't comparable across variants, so I compared *behavior*):

| Policy | Mean height | Mean speed | Outcome |
|---|---|---|---|
| 500k, standard reward | 8.1 cm | 2.9 cm/s | survived |
| 500k, gate **removed** | 8.4 cm | 1.6 cm/s | survived |
| 1M, standard reward | 7.9 cm | 5.6 cm/s | survived |
| 1M, gate **removed** | 7.7 cm | 6.3 cm/s | survived |
| shipped v9 (10M) | 8.0 cm | 7.4 cm/s | survived |

**At neither 500k nor 1M is the sabotaged reward distinguishable from the good one.** Both stand at ~8 cm and walk. At 1M the sabotaged one is marginally *faster*. A student would correctly report "I changed it and nothing happened."

**Why**, and it's a genuine lesson: removing the gate decouples speed from posture, but `W_HEIGHT = 1.0` still pays for standing as a separate additive term. Crawling only wins once the policy has learned to crawl *fast*, and that exploration is expensive. The historical v7 sprawl this exercise is modeled on happened under a **different reward shape** over 10M steps — the current shape is more robust than the story it's used to tell.

**What I'd try first** (untested, ~15 min each to check): set `W_HEIGHT = 0.0` *and* remove the gate, so nothing at all pays for posture. Failing that, use reward edits whose effect is a *removal* of behavior rather than a discovery of new behavior — those need far less exploration:
- `TARGET_SPEED_MS = 0.0` → asks the robot to stand still; should show inside 300k.
- `FALL_PENALTY = 0.0` → falling becomes free.
- `W_EFFORT` 0.03 → 1.0 → moving becomes expensive; policy should go rigid.

I've replaced the README's exercise list with these and documented the negative result, **but I have not verified the three replacements** — do that before the class runs them.

### Installation **[M] for wheel availability, [I] for Colab specifics**

Dependency wheel matrix on **Linux x86_64** (queried from PyPI):

| Package | cp310 | cp311 | cp312 |
|---|---|---|---|
| stable-baselines3 2.8.0 | ✓ pure | ✓ | ✓ |
| gymnasium 1.2.3 | ✓ pure | ✓ | ✓ |
| pyserial 3.5 | ✓ pure | ✓ | ✓ |
| imageio 2.37.4 | ✓ pure | ✓ | ✓ |
| torch 2.x | ✓ | ✓ | ✓ |
| numpy 1.26.4 | ✓ | ✓ | ✓ |
| scipy 1.14.1 | ✓ | ✓ | ✓ |
| **pybullet 3.2.7** | ✓ | ✓ | **✗ compiles** |

**The critical finding: pybullet publishes wheels for Linux x86_64 only, and only up to Python 3.11.** There are **no macOS wheels and no Windows wheels at all** — every Mac and every Windows machine compiles from source, which on Windows needs MSVC Build Tools. On school-managed laptops without admin rights that is a hard stop.

This was the single largest blocker, and **the Colab/local split resolves it** — because `robot/` never imports pybullet, the local install is pure wheels on any Python 3.9–3.13, and the simulator only ever runs on Colab's Linux. **[M]** verified.

Remaining install notes:
- **Colab Python version is the open risk.** If Colab is on 3.12+, `pip install pybullet` compiles from source — several minutes per session, every session, and it will look like a hang. It normally succeeds (build tools present). I could not verify Colab's current Python from here.
- **torch is undeclared** in both requirements files; it comes in via SB3 as a ~900 MB download.
- No GPU needed anywhere — PPO on a 246→256→256→8 MLP is CPU-bound on the simulator, not the network.
- Linux serial access needs `sudo usermod -aG dialout $USER` **and a re-login**. Documented in `QUICKSTART.md`. This is the classic Linux robotics failure and it needs admin rights once.

### Determinism **[M] verified**

**Yes.** Two runs, `--seed 7 --envs 2 --steps 25000`, produced **bit-identical network weights** across every layer. I compared the full `state_dict`, not just the reward curve.

Seed is set at `train.py:173` via `--seed` (default 42) and propagates through SB3's `set_random_seed` to the vectorized envs. Notably, **the environment itself has no stochasticity at all** — fixed reset pose, no domain randomization, no observation noise. Eval reward reports `± 0.00` across 3 episodes **[M]**, confirming it.

That's excellent for grading (two students with the same config get the same answer) and **poor for robustness** — nothing teaches that a policy must survive variation, and policies trained this way are brittle in exactly the way the sim2real narrative warns about. Threading is not a defeater here; `SubprocVecEnv` was used in the test.

### Modification surface

If students may edit reward weights, observation composition, and termination conditions:

| What | File | Lines |
|---|---|---|
| Reward weights, all 18 | `training/env.py` | 97–131 (labeled block) |
| Reward structure (gating) | `training/env.py` | 134–182 |
| Termination thresholds | `training/env.py` | 104–105 |
| Observation composition | `contract.py` | 186–194 |

**Cleanly separable — this is the codebase's strongest feature for teaching.** Weights are a contiguous labeled block of named constants. `compute_reward` is a pure function of keyword arguments with no simulator objects in scope, so a student cannot reach PyBullet from inside it even by accident. Nothing they're allowed to touch sits near PPO.

Two caveats: editing `JOINT_HISTORY_LEN` in `contract.py` changes `OBSERVATION_SIZE` and **invalidates every existing policy** including the shipped one, with a shape-mismatch traceback on load. And the `smoothness` weights, though in the labeled block, multiply elsewhere (§4).

### Failure modes

**Loud** (traceback — no pre-announcement needed):
- Changing `JOINT_HISTORY_LEN`, then loading an old policy → shape mismatch.
- Syntax/name error in the reward block → immediate crash at `python training/env.py`.
- `python training/watch.py` on Colab **without** `--gif` → PyBullet tries `p.connect(p.GUI)` on a headless machine and fails **[I]**. The notebook always passes `--gif`; a student who types the README command by hand will hit it.
- Serial permission denied on Linux without `dialout`.
- Deleting a reward key from the `parts` dict → `KeyError` in `watch.py`.

**Silent — these need pre-announcing to the class:**
1. **A reward edit that changes nothing.** The measured headline case. The student sees a plausible number, a plausible GIF, and no difference. Nothing distinguishes "my change had no effect" from "I didn't train long enough."
2. **`watch.py` recomputes the reward with *current* code against an *old* policy.** Edit weights, re-run `watch.py` on yesterday's model, and the breakdown numbers change while the gait doesn't. Very easy to misread as "my change worked."
3. **The 15° hardware safety margin** silently altering 14% of commanded steps **[M]**. Sim and robot disagree and nothing says so.
4. **Setting a weight absurdly high** (e.g. `W_EFFORT = 100`) → policy freezes, reward curve goes flat, no error. Looks like a broken install.
5. **`best_model.zip` never updating** if eval never improves — the student trains for 40 minutes and silently loads a near-random policy. `final_model.zip` exists but the README points at `best_model.zip`.
6. **Truncation at 200 sim steps vs 400 hardware steps** — no warning.

### Hardware deployment

**One-time instructor setup:**
1. Flash firmware with the RL patch (`firmware/rl_patch.diff`, ~60 lines, `git apply` onto stock OpenCat, build with `RL_SIM2REAL_MINIMAL`, board `esp32:esp32:esp32`). Needs arduino-cli or the Arduino IDE.
2. `sudo usermod -aG dialout <user>` + re-login, per machine.
3. Verify with `python robot/link.py`.

**Per session:** plug in USB, switch on the servo battery (USB powers the board but not the servos — a documented and very common confusion), `python robot/run.py <policy>`, flat surface, hand nearby.

**Can a policy be deployed without reflashing between students? Yes** — this is the good news. The firmware is a generic executor of `Y` posture packets; it holds no policy and no student-specific state. Swapping students is swapping a `.zip` file path on the laptop. Provided nobody changes `JOINT_HISTORY_LEN` or the action semantics in `contract.py`, every student's policy runs on the same flash. **[I]** from the firmware handler; not exercised with 35 policies.

**With 18 robots the throughput problem disappears.** 35 students to 18 robots is ~2 per robot. A `run.py` session is 20 s of walking plus handling — call it 2 minutes — so a pair needs ~4 minutes of robot time. That fits a single period with room for repeat attempts, and it removes the servo-duty-cycle worry entirely (2 runs per robot, not 35).

**What 18 robots costs instead is setup and variance:**
- **18 firmware flashes.** One-time, but `git apply` + build + flash per unit. Budget an afternoon, and flash one first to confirm the toolchain before doing 17 more.
- **18 sets of serial permissions** if students use their own laptops — though `dialout` is per-*laptop*, not per-robot, so this scales with machines, not units.
- **Per-unit variation, which is new and unhandled.** See Blocker 2.

### Shipped assets **[M]**

| Asset | Status |
|---|---|
| `robot/policies/walk_v9.zip` | Trained 10M steps, **hardware-verified**. Scores **+538.5**, survives 200 steps, 8.2 cm/s, 8.1 cm height. Zero contact penalties. |
| `docs/walk.gif` | 1.0 MB rendered episode of v9 walking |
| `training/model/bittle.urdf` + 10 meshes | 3.1 MB, from CAD reconstruction |
| `firmware/rl_patch.diff` | The 60-line firmware delta |
| `colab_train.ipynb` | 17-cell training notebook |

**Yes — demonstration works with zero training.** `python training/watch.py robot/policies/walk_v9.zip` produces the full reward breakdown immediately, and `robot/run.py` walks the physical robot. That's a strong day-one hook. A second policy (`walk_v10`) was removed at the user's request; it had never been hardware-tested anyway.

---

## Blockers — ranked

1. **The flagship exercise doesn't work at any feasible budget. [M]** Measured at both 500k and 1M: no visible difference. The README made an unqualified promise it cannot keep. I've rewritten it with a documented negative result and three faster replacements — **the replacements are unverified**. Verify one before committing a class period.
2. **The measured constants come from one specific robot, and you have 18. [I]** This replaces the original "one robot" throughput blocker, which 18 units resolve. `contract.py` hard-codes numbers a human measured on a single Bittle: joint ROM (`LIMIT_LOW_DEG`/`LIMIT_HIGH_DEG`, lines 134-135), `STAND_HEIGHT_M = 0.08`, and the `STAND_POSE_DEG = [30]*8` reset. Petoi's firmware absorbs per-unit servo-horn offsets in its EEPROM calibration, so identical units *should* land close — but "should" is doing real work there. Assembly tolerance, servo wear, and a horn one spline off all shift where a leg actually stops relative to the shell. A robot whose true ROM is tighter than the table will drive a leg into its own body and stall a servo, and **nothing in the code detects this** — the policy just performs worse. Before the class: run the firmware `balance` posture on all 18, confirm each stands, and spot-check the shoulder extremes on 3-4 units. If they vary meaningfully, the 15° safety margin (Blocker 3) stops being a nuisance and becomes the thing protecting your hardware.
3. **The 15° safety margin silently alters the shipped policy. [M]** 14% of steps, up to 4°. Either report it in `run.py`, apply the same clip in sim, or default it to 0 with a documented "raise it if the robot fights its shell."
4. **Colab's Python version is unverified.** If 3.12+, every session pays a multi-minute pybullet source build that looks like a hang. Verify once, then tell students exactly what to expect.
5. **`best_model.zip` can silently be near-random** if eval never improves. A student can burn 40 minutes and deploy noise with no error. Print a warning when best-eval never beat the initial value.
6. **No domain randomization at all. [M]** Every episode is identical (`± 0.00` eval variance). Great for grading, but the repo teaches sim2real while omitting the main technique for surviving it, and it makes student policies brittle on the one robot.
7. **torch undeclared.** ~900 MB implicit dependency in both install paths.

## Cheap wins — under an hour each

1. **Print expected wall-clock** when `train.py` starts (`steps / measured_throughput`), so a student knows at minute 2 whether they've asked for 6 hours.
2. **Warn when `best_model` never improved** — three lines in a callback, kills silent failure #5.
3. **Count and report clipped steps in `run.py`** — kills silent failure #3, and is itself a teachable moment about sim2real gaps.
4. **Stamp the policy's reward config into the run directory** and have `watch.py` warn if the current `env.py` weights differ from those the policy trained under. Kills silent failure #2, the nastiest one.
5. **Add `--compare a.zip b.zip` to `watch.py`** producing one side-by-side GIF and a two-column reward table. This is the grading artifact the assignment actually wants.
6. **Move the `smoothness` weighting inside `compute_reward`** so all 18 terms are weighted in one place.
7. **Add `torch` to both requirements files** with the download size in a comment.
8. **A `--quick` preset** (300k, `--envs 2`, eval every 25k) so the first experiment fits in one period.

## What I could not determine

| Question | Why | What would settle it |
|---|---|---|
| Colab's current Python version, and whether pybullet compiles or installs from wheel | No Colab access from here | Run cell 1 of `colab_train.ipynb` once |
| Real Colab wall-clock | Extrapolated ×2–3 from M4 measurements | Run 300k in Colab, read the reported fps |
| Whether the three replacement exercises produce visible change | Ran out of budget after the two 1M runs | Three 300k runs, ~15 min each |
| Whether `W_HEIGHT = 0` + no gate reproduces the crawl | Same | One 1M run |
| Per-unit ROM variation across the 18 | Needs the robots | Jog the shoulders on 3-4 units, compare against `contract.py` limits |
| Whether school laptops can install even the robot-side wheels | Depends on local IT policy | Try it on one managed laptop |
| Whether `walk_v10` works on hardware | Never tested; now removed from the repo | n/a |
| Whether `walk_v9` transfers to all 18 units, or only the one it was tuned on | Only ever run on a single robot | Deploy it to 3-4 units and compare gaits |
| Real classroom pacing | Not inferable from code | A pilot with 4–5 students |
| Whether the hardware path still works end-to-end after this session's rewrite | No robot attached; `robot/link.py` self-check never run | 5 minutes with the robot plugged in |

The last row deserves emphasis: **the entire hardware path is unexercised since the rewrite.** It imports cleanly and the protocol matches the firmware, but no byte has gone down a wire. Run `python robot/link.py` with the robot attached before promising a class it works.
