# Quickstart

Two machines, two jobs. You never need the simulator and the robot on the same
computer.

| Where | What it does | Install |
|---|---|---|
| **Google Colab** | Trains the policy (needs a simulator + lots of CPU) | one cell |
| **Your Pop!\_OS laptop** | Runs the trained policy on the real robot | one command |

---

## A. Train in Colab

Click this, then `Runtime -> Run all`:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NiranjanJ2/bittle-rl/blob/main/colab_train.ipynb)

Nothing to upload and nothing to install — the link opens the notebook in
Colab straight from GitHub.

It clones the repo, installs the dependencies, trains, and gives you a
`walk.zip` to download. Everything is in the notebook — nothing to type.

**Budget your time.** Training is CPU-bound; a Colab free-tier instance gives
about 2 cores, so expect roughly:

| Steps | Colab free tier | What you get |
|---|---|---|
| 200k | ~8 min | stands up, barely shuffles |
| 1M | ~40 min | walks, slowly and imperfectly |
| 10M | ~7 hours | the quality of the shipped policy |

Colab free disconnects after a while, so 10M in one sitting is not realistic.
Train 1M, or resume from a checkpoint across sessions.

## B. Run on the robot (Pop!\_OS)

```bash
git clone https://github.com/NiranjanJ2/bittle-rl.git
cd bittle-rl
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-robot.txt
```

Nothing compiles — every package is a prebuilt wheel, so this works without
build tools or admin rights, on any Python from 3.9 to 3.13.

**One-time: give yourself permission to use the USB port.** On Linux, serial
devices belong to the `dialout` group and you are not in it by default:

```bash
sudo usermod -aG dialout $USER
```

Then **log out and back in** — group changes do not apply to an open session.
Skipping this is the single most common Linux setup failure, and it appears as
`PermissionError: [Errno 13] Permission denied: '/dev/ttyUSB0'`.

Check the robot is talking (this moves nothing):

```bash
python robot/link.py
```

You should see IMU rows with sample ages around 25 ms. Then walk it:

```bash
python robot/run.py robot/policies/walk_v9.zip
```

`walk_v9.zip` is included in this repo and already works on hardware, so you
can do this before training anything of your own. To run a policy you trained
in Colab, download it from the notebook and pass its path instead.

**Before you press enter:** flat open surface, battery switched on (USB powers
the board but *not* the servos), and a hand nearby. Ctrl-C rests the servos.

---

## Which Python?

- **Robot side:** anything modern. All wheels, no constraints.
- **Colab / training side:** PyBullet publishes Linux wheels only for Python
  3.10 and 3.11. On 3.12+ it compiles from source — several minutes, but it
  does work on Colab. The notebook handles this; just be patient on the
  install cell.

## If you want to train locally instead of Colab

Works, but check your Python version first:

```bash
python3 --version
```

If it is 3.10 or 3.11, `pip install -r requirements-train.txt` is instant.
Pop!\_OS 22.04 ships 3.10 and 24.04 ships 3.12; on 3.12 the PyBullet install
compiles, which needs `sudo apt install build-essential python3-dev`.
