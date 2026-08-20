# The firmware side

The Bittle runs Petoi's stock OpenCat firmware on an ESP32. It is a big
codebase built for a robot that walks pre-written gaits, responds to voice and
IR remotes, and talks over Bluetooth.

None of that suits reinforcement learning. **Three changes make it suitable.**
They are in `rl_patch.diff` — about 60 lines, against stock OpenCat.

You do not need to read the firmware to understand the RL. But you do need to
know *why* these three changes exist, because each one is a place where the
real world refuses to behave like a tidy `env.step()`.

---

## Change 1 — a command that does not block

**`reaction.h`, new `Y` token.** The heart of the patch:

```c
case T_RL_RAW_POSTURE_BIN:            // 'Y'
  {
    gyroBalanceQ = false;             // turn off the built-in balancer
    resetAdjust();
    if (cmdLen == WALKING_DOF) {      // exactly 8 joints
      for (byte i = 0; i < WALKING_DOF; i++)
        calibratedPWM(DOF - WALKING_DOF + i, (int8_t)newCmd[i]);
    }
    break;
  }
```

Petoi's existing posture commands *interpolate*. You give them a target and
they ease into it over some hundreds of milliseconds, blocking until they
arrive. That is exactly right for choreographed motion and exactly wrong here:
the policy issues a new target every 50 ms, and a command still easing toward
the previous target when the next one arrives means the robot never executes
what the policy asked for.

`Y` writes the servo targets and returns. No interpolation, no blocking.

Two details worth noticing:

- **`gyroBalanceQ = false`.** The stock firmware runs its own balance
  controller that quietly corrects the joint angles using the IMU. Leaving that
  on would mean two controllers fighting over the same servos, and the policy
  would be learning to compensate for a system it cannot see. The RL policy
  must be the *only* controller. This is a general rule: if something else is
  also steering, your policy is not learning what you think it is.

- **`(int8_t)`.** The angles arrive as raw signed bytes, which is why
  `contract.py` clips every command to [-128, 127]. The wire format is a real
  constraint on the action space, not an arbitrary choice.

## Change 2 — sensor readings 8x more often

**`imu.h`, print interval 200 ms → 25 ms.**

```c
- const unsigned long PRINT6AXIS_MIN_INTERVAL = 200;   // 5 Hz
+ #define PRINT6AXIS_MIN_INTERVAL_MS 25                // 40 Hz
```

Stock firmware prints IMU data at 5 Hz, which is plenty for a status display.
The policy runs at 20 Hz, so at the stock rate three out of every four control
steps would act on a repeat of an old reading — the robot could tip measurably
before the policy ever saw it.

At 40 Hz there is always a reading newer than the last control step.

The 200 ms limit was not arbitrary, though. It was protecting the ESP32's
stack from `snprintf` with float formatting, which is expensive. 25 ms holds up
in practice, but it is a real budget, not free.

## Change 3 — turn off everything else

**`OpenCat.h`, the `RL_SIM2REAL_MINIMAL` build flag.** Compiles out Bluetooth,
BLE, the web server, and most extension modules.

Those features share the ESP32's radio, CPU and UARTs with the serial link. Any
of them can stall the main loop for tens of milliseconds at an unpredictable
moment — and a 30 ms stall inside a 50 ms control period is a control step that
silently did not happen.

The robot becomes less capable and much more *predictable*. That trade is
almost always right for a control loop.

(A fourth hunk in `motion.h` is a one-line build fix: some calibration code
references a Bluetooth buffer that no longer exists when BLE is compiled out.)

---

## Applying it

```bash
git clone https://github.com/PetoiCamp/OpenCatEsp32.git
cd OpenCatEsp32
git apply /path/to/rl_patch.diff
```

Then flash with `RL_SIM2REAL_MINIMAL` defined, board `esp32:esp32:esp32`.
Set your model and board in `OpenCatEsp32.ino` as usual (`#define BITTLE`,
`#define BiBoard_V1_0`) before building.

Verify it worked without moving the robot:

```bash
python robot/link.py
```

You should see IMU rows with sample ages around 25 ms. If nothing arrives, the
`Y` token or the IMU rate did not make it onto the chip.

---

## What the firmware still cannot tell you

Worth being explicit about, because it shapes the whole observation design in
`contract.py`:

- **No joint position feedback.** The firmware's `j` command looks like it
  reports joint angles, but it echoes the *target table* — the angles it was
  told to hold, not where the servos actually are. The servos have no encoder
  to ask. If a leg is jammed against the floor, nothing reports it.
- **No foot contact sensors.** The robot cannot feel the ground.
- **No odometry.** No way to know how far it has travelled, or how fast.

So the policy navigates on orientation and acceleration alone, plus a memory of
what it *commanded*. Every richer signal the reward uses during training —
height, velocity, foot slip — exists only in the simulator.
