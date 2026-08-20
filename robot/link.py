"""THE CONNECTOR — the serial cable between Python and the robot.

This is the transport layer and nothing else. It knows how to find the robot,
send joint angles to it, and read its sensors. It knows nothing about
reinforcement learning.

--------------------------------------------------------------------------
THE PROTOCOL
--------------------------------------------------------------------------
The robot listens on USB serial at 115200 baud for single-letter commands.
Only three matter here:

    Y <8 signed bytes> ~     set the 8 leg joints, right now, no smoothing
    gP                       start streaming IMU readings continuously
    d                        relax into the built-in rest crouch

`Y` is a custom command added for this project -- see `firmware/`. Petoi's
normal posture commands BLOCK: they interpolate smoothly to the target and do
not return until they arrive, which is lovely for choreography and useless for
RL, where a new target arrives every 50 ms and must take effect immediately.

Note that the packet carries SIGNED BYTES. That is the real, physical reason
`contract.py` clips commands to [-128, 127]: a larger angle cannot be
expressed on the wire, and would silently wrap to a wildly wrong value.

--------------------------------------------------------------------------
WHY THERE IS A BACKGROUND THREAD
--------------------------------------------------------------------------
The IMU streams at about 40 Hz; the policy runs at 20 Hz. Those clocks are
unrelated, so a sample is always partway through arriving when we want one.

The naive approach -- "read from the port until a reading shows up" -- makes
the control loop wait on the robot, so serial jitter turns directly into
control-rate jitter, and the policy stops running at the frequency it was
trained for.

Instead a thread reads continuously and keeps only the newest reading. The
control loop takes whatever is there and never blocks.

That buys something more valuable than smoothness: HONESTY ABOUT STALENESS.
`read_imu()` returns the reading AND its age. If the cable is jostled or the
robot browns out, the age climbs and `run.py` can stop instead of confidently
steering on a sensor reading from two seconds ago. Silent staleness is one of
the nastier failure modes in real robotics -- everything looks fine, the
numbers are plausible, and they are simply old.
"""

from __future__ import annotations

import glob
import re
import struct
import threading
import time
from typing import Optional

import serial  # pyserial

BAUD = 115200

# Serial ports that look like a robot. Excludes Bluetooth and debug consoles,
# which show up as serial devices on macOS and will happily accept a
# connection while doing nothing at all.
PORT_PATTERNS = ["/dev/cu.usbserial*", "/dev/cu.wchusbserial*", "/dev/cu.usbmodem*",
                 "/dev/cu.SLAB_USBtoUART*", "/dev/ttyUSB*", "/dev/ttyACM*"]

# The firmware prints IMU readings as a line like:
#     ICM: -0.31  0.14  9.79   12.4   -2.1    0.7
#          ax     ay    az     -yaw   pitch   roll
# Accelerations are m/s^2, angles are degrees. (The minus on yaw is the
# firmware's own convention, not a mistake.)
_IMU_LINE = re.compile(r"(?:ICM|MCU):")
_NUMBER = re.compile(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?")


class RobotLink:
    """A serial connection to a Bittle, with a non-blocking IMU reader."""

    def __init__(self, port: Optional[str] = None, timeout: float = 8.0):
        self.port = port or self._find_port()
        self.serial = serial.Serial(self.port, BAUD, timeout=0.05)

        # The ESP32 reboots when the serial port opens (DTR toggles the reset
        # line), so the first couple of seconds are boot chatter, not data.
        # Waiting is not optional: commands sent during boot are simply lost.
        time.sleep(2.0)
        self.serial.reset_input_buffer()

        self._latest: Optional[tuple[float, ...]] = None
        self._latest_at = 0.0
        self._lock = threading.Lock()
        self._running = True
        self._reader = threading.Thread(target=self._read_forever, daemon=True)
        self._reader.start()

        self.send_raw(b"gP\n")  # ask the firmware to stream IMU readings

        # Confirm the robot is actually talking before anything else runs.
        # Failing here, loudly, beats failing later as a policy stepping on
        # a frozen all-zero observation.
        if not self.wait_for_imu(timeout):
            raise RuntimeError(
                f"Connected to {self.port} but no IMU data arrived in {timeout:.0f}s.\n"
                f"  - Is the robot powered on? (USB alone powers the board, not the servos)\n"
                f"  - Is a Serial Monitor holding the port? (lsof {self.port})\n"
                f"  - Is this firmware built with the RL patch? (see firmware/)"
            )

    # ------------------------------------------------------------------
    # Finding the robot
    # ------------------------------------------------------------------
    @staticmethod
    def _find_port() -> str:
        ports = [p for pattern in PORT_PATTERNS for p in glob.glob(pattern)]
        if not ports:
            raise RuntimeError(
                "No robot found. Plug it in over USB and check the cable is a "
                "data cable, not charge-only.\nLooked for: " + ", ".join(PORT_PATTERNS)
            )
        if len(ports) > 1:
            print(f"[link] multiple ports found, using {ports[0]}  (others: {ports[1:]})")
        return ports[0]

    # ------------------------------------------------------------------
    # Sending
    # ------------------------------------------------------------------
    def send_raw(self, data: bytes) -> None:
        self.serial.write(data)
        self.serial.flush()

    def send_joints(self, angles_deg) -> None:
        """Send all 8 leg joint angles as one `Y` packet.

        Angles must already be clipped to the transmittable range -- that is
        `contract.apply_action`'s job, and doing it there rather than here is
        what keeps the simulator honest about the same limits.
        """
        values = [int(round(float(a))) for a in angles_deg]
        if len(values) != 8:
            raise ValueError(f"expected 8 joint angles, got {len(values)}")
        if any(v < -128 or v > 127 for v in values):
            # struct.pack would raise anyway, but this says why.
            raise ValueError(f"angle outside the signed-byte packet range: {values}")
        self.send_raw(b"Y" + struct.pack("8b", *values) + b"~")

    def rest(self) -> None:
        """Relax into the firmware's built-in crouch. Always end a session here.

        Leaving servos holding a pose keeps them drawing current, heating up,
        and straining the gears against whatever they are pressed into.
        """
        self.send_raw(b"d\n")

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------
    def read_imu(self) -> tuple[Optional[tuple[float, ...]], float]:
        """Return (newest IMU reading, how many seconds old it is).

        Returns (None, inf) if nothing has ever arrived. Never blocks, and
        never waits for a fresh sample -- it hands back whatever the reader
        thread has, and tells you honestly how stale that is.

        The reading is (ax, ay, az, -yaw, pitch, roll); feed it straight into
        `contract.build_body_state`.
        """
        with self._lock:
            if self._latest is None:
                return None, float("inf")
            return self._latest, time.time() - self._latest_at

    def wait_for_imu(self, timeout: float = 5.0) -> bool:
        """Block until at least one IMU reading has arrived. Startup only."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.read_imu()[0] is not None:
                return True
            time.sleep(0.02)
        return False

    def _read_forever(self) -> None:
        """Reader thread: keep only the newest IMU reading, discard the rest.

        Deliberately keeps no queue. If parsing falls behind, old readings are
        worthless to a controller -- it wants the present, not a backlog.
        """
        while self._running:
            try:
                raw = self.serial.readline()
            except Exception:
                break  # port closed underneath us; shutting down
            if not raw:
                continue
            line = raw.decode("utf-8", errors="ignore")
            if not _IMU_LINE.search(line):
                continue  # boot messages, command echoes, other firmware chatter
            values = _NUMBER.findall(line)
            if len(values) < 6:
                continue  # a line torn in half by a buffer boundary
            try:
                reading = tuple(float(v) for v in values[:6])
            except ValueError:
                continue
            with self._lock:
                self._latest = reading
                self._latest_at = time.time()

    # ------------------------------------------------------------------
    def close(self) -> None:
        self._running = False
        try:
            self.rest()
            time.sleep(0.5)
        finally:
            self._reader.join(timeout=1.0)
            self.serial.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# ============================================================================
# Self-check: run `python robot/link.py` with a robot plugged in.
# Moves nothing -- it only listens, so it is safe to run any time.
# ============================================================================
if __name__ == "__main__":
    print("looking for a robot...")
    with RobotLink() as link:
        print(f"connected on {link.port}\n")
        print(f"{'ax':>8}{'ay':>8}{'az':>8}{'-yaw':>8}{'pitch':>8}{'roll':>8}{'age ms':>9}")
        rates = []
        for _ in range(40):
            time.sleep(0.05)
            reading, age = link.read_imu()
            if reading is None:
                print("  (no data)")
                continue
            rates.append(age)
            print("".join(f"{v:>8.2f}" for v in reading) + f"{age * 1000:>9.0f}")

        worst = max(rates) * 1000
        print(f"\nworst sample age {worst:.0f} ms")
        # At ~40 Hz the newest sample should never be much older than 25 ms.
        # Consistently higher means dropped packets or a busy port, and the
        # policy would be steering on stale data.
        print("OK — IMU stream is fresh." if worst < 200 else
              "WARNING — stale samples. Check the cable and close other serial monitors.")
