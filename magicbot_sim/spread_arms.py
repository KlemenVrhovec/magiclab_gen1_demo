"""
Spread: the pinned MagicBot Gen1 lifts both arms out to the sides, and holds them there.

    python spread_arms.py              # viewer
    python spread_arms.py --headless   # print where the hands go, no window

This is step 1 of pick_box.py on its own. Exactly ONE joint per arm does the work --
shoulder_roll -- so the pose is written as plain degrees instead of being solved by
inverse kinematics. Every other joint is commanded to stay at zero.
"""

import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

MJCF = Path(__file__).resolve().parent / "magicbot-gen1_description" / "mjcf" / "MAGICBOT.xml"

KP, KD, TAU_LIMIT = 300.0, 8.0, 100.0   # same controller as arm_demo.py
PELVIS_H = 1.25                          # where the pelvis is bolted to the world
SPREAD_DEG = 75.0                        # how far out the arms swing
MOVE_TIME = 3.0                          # seconds to get there


def load_model():
    """Official MJCF with the pelvis pinned, so the robot hangs instead of falling."""
    spec = mujoco.MjSpec.from_file(str(MJCF))
    pelvis = spec.body("pelvis")
    spec.delete(pelvis.joints[0])        # drop the freejoint -> pelvis fixed to world
    pelvis.pos = [0.0, 0.0, PELVIS_H]
    return spec.compile()


model = load_model()
data = mujoco.MjData(model)

# Each of the 30 actuators drives one joint. These arrays say where that joint's angle
# and speed live inside data.qpos / data.qvel.
JOINT = model.actuator_trnid[:, 0]
QADR = model.jnt_qposadr[JOINT]
VADR = model.jnt_dofadr[JOINT]
NAMES = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in JOINT]

# The two poses we move between. Everything not named here stays at 0.
# Left opens positive and right opens negative: their joint ranges are mirrored.
HOME = np.zeros(model.nu)
SPREAD = np.zeros(model.nu)
SPREAD[NAMES.index("left_shoulder_roll_joint")] = np.deg2rad(+SPREAD_DEG)
SPREAD[NAMES.index("right_shoulder_roll_joint")] = np.deg2rad(-SPREAD_DEG)


def target(t):
    """Move HOME -> SPREAD over MOVE_TIME seconds, then stay there."""
    s = min(t / MOVE_TIME, 1.0)
    s = 0.5 - 0.5 * np.cos(np.pi * s)    # ease in and out, no sharp start or stop
    return HOME * (1 - s) + SPREAD * s


def step():
    """One 2 ms tick: work out the torques, then advance the physics."""
    angle = data.qpos[QADR]
    speed = data.qvel[VADR]
    twist = data.qfrc_bias[VADR] + KP * (target(data.time) - angle) - KD * speed
    data.ctrl[:] = np.clip(twist, -TAU_LIMIT, TAU_LIMIT)
    mujoco.mj_step(model, data)


data.qpos[QADR] = HOME
mujoco.mj_forward(model, data)

if "--headless" in sys.argv:
    hand = model.body("left_wrist_pitch_link").id
    roll = NAMES.index("left_shoulder_roll_joint")
    print(f"{'t':>5} {'roll(deg)':>10} {'hand x':>8} {'hand z':>8}")
    nxt = 0.0
    while data.time < MOVE_TIME + 2.0:
        step()
        if data.time >= nxt:
            nxt += 0.5
            print(f"{data.time:5.1f} {np.rad2deg(data.qpos[QADR[roll]]):10.1f} "
                  f"{data.xpos[hand][0]:8.3f} {data.xpos[hand][2]:8.3f}")
    raise SystemExit

print(f"arms lift out to {SPREAD_DEG:.0f} deg over {MOVE_TIME:.0f} s, then hold")
print("left-drag orbit | scroll zoom | Esc quit")

with mujoco.viewer.launch_passive(model, data) as viewer:
    wall = time.perf_counter()
    while viewer.is_running():
        step()
        viewer.sync()
        lag = data.time - (time.perf_counter() - wall)   # keep it at real-time speed
        if lag > 0:
            time.sleep(lag)
