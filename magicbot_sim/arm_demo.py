"""
PD-control demo for the MagicBot Gen1 — makes the robot hold a pose and wave an arm.

The cloned magicbot-gen1_description repo is read-only and is never modified. When
the base is pinned, the freejoint is removed from an in-memory MjSpec copy of the
model at load time; nothing is written back to disk.

    python arm_demo.py              # pelvis pinned, left arm waves (clearest view)
    python arm_demo.py --still      # pelvis pinned, PD holds the home pose, no motion
    python arm_demo.py --floating   # free-floating base: PD holds the joints, but the
                                    #   robot still topples — there is no balance controller

Why this script exists at all: every actuator in MAGICBOT.xml is a <motor>, i.e. a raw
torque source with gear=1. Writing a constant value into data.ctrl applies a constant
torque, which accelerates a joint into its limit rather than moving it to an angle. To
command a *position* you have to recompute the torque every step from the current
tracking error -- that is the loop below.
"""

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

HERE = Path(__file__).resolve().parent
MJCF = HERE / "magicbot-gen1_description" / "mjcf" / "MAGICBOT.xml"

# The model already carries 20 N*m*s/rad of joint damping, so the plant is heavily
# damped on its own and KD only needs to take the edge off.

#These three numbers are the "personality" of the controller 
# — how hard it pulls, how much it brakes, and how much force it's allowed to use. 

# Every 2 milliseconds, the script asks each joint one question: "where are you, versus where I want you?" 
# Then it applies a twist (torque, measured in newton-metres, N·m) to fix the gap.

# The formula in words:
# KP is spring like force, KD is damper (dušilec, viskoznost) like force
# twist  =  gravity help  +  KP × (how far off)  −  KD × (how fast it's moving)

KP = 300.0
KD = 8.0
# TAU LIMIT is max torque that can be applied to a joint. If the controller wants to apply more than this, 
# it will be clamped to this value. This is important because if the controller tries to apply 
# too much torque, it can cause instability or damage to the robot.
TAU_LIMIT = 100.0  # N*m clamp; the MJCF declares no forcerange, so impose a sane one

PINNED_PELVIS_HEIGHT = 1.25  # m, high enough that the legs dangle clear of the ground


def load_model(pin_base: bool) -> mujoco.MjModel:
    """Compile the official MJCF, optionally pinning the pelvis to the world."""
    spec = mujoco.MjSpec.from_file(str(MJCF))
    if pin_base:
        pelvis = spec.body("pelvis")
        spec.delete(pelvis.joints[0])  # drop the <freejoint/> -> pelvis fixed to world
        pelvis.pos = [0.0, 0.0, PINNED_PELVIS_HEIGHT]

     # ---- new object goes here ----
    cube = spec.worldbody.add_body()
    cube.name = "cube"
    cube.pos = [0.45, 0.0, 1.6]
    cube.add_freejoint()                 # 6 DOF, so it can fall and tumble

    g = cube.add_geom()
    g.name = "cube_geom"
    g.type = mujoco.mjtGeom.mjGEOM_BOX
    g.size = [0.05, 0.05, 0.05]          # HALF-extents -> a 10 cm cube
    g.rgba = [0.9, 0.3, 0.2, 1.0]        # red
    g.mass = 0.5                         # kg
    # -----------------------------
    return spec.compile()


class JointMap:
    """Maps each actuator to the joint it drives, and to that joint's state indices."""

    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self.qadr = np.zeros(model.nu, dtype=int)  # index into data.qpos
        self.vadr = np.zeros(model.nu, dtype=int)  # index into data.qvel / qfrc_bias
        self.name = []
        for a in range(model.nu):
            joint = model.actuator_trnid[a, 0]
            self.qadr[a] = model.jnt_qposadr[joint]
            self.vadr[a] = model.jnt_dofadr[joint]
            self.name.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint))
        self.lo = model.jnt_range[model.actuator_trnid[:, 0], 0].copy()
        self.hi = model.jnt_range[model.actuator_trnid[:, 0], 1].copy()

    def index(self, joint_name: str) -> int:
        return self.name.index(joint_name)


def home_pose(jm: JointMap) -> np.ndarray:
    """Neutral standing pose: all zeros, with the arms held slightly out from the torso
    so the wave does not drive the upper arm straight through the ribcage."""
    q = np.zeros(jm.model.nu)
    q[jm.index("left_shoulder_roll_joint")] = 0.30
    q[jm.index("right_shoulder_roll_joint")] = -0.30
    return q


def wave_targets(jm: JointMap, target: np.ndarray, t: float) -> np.ndarray:
    """Overlay a slow left-arm wave on the home pose. Stays inside the joint limits."""
    target = target.copy()
    w = 2.0 * np.pi * 0.35  # 0.35 Hz
    target[jm.index("left_shoulder_pitch_joint")] = np.deg2rad(-35.0 + 35.0 * np.sin(w * t))
    target[jm.index("left_shoulder_roll_joint")] = np.deg2rad(45.0)
    target[jm.index("left_elbow_roll_joint")] = np.deg2rad(-55.0 + 45.0 * np.sin(w * t + 1.2))
    return np.clip(target, jm.lo, jm.hi)


def control(model, data, jm: JointMap, target: np.ndarray) -> None:
    """Gravity-compensated PD. qfrc_bias is the gravity + Coriolis torque needed just to
    hold still, so feeding it forward lets modest gains track accurately."""
    q = data.qpos[jm.qadr]
    v = data.qvel[jm.vadr]
    tau = data.qfrc_bias[jm.vadr] + KP * (target - q) - KD * v
    data.ctrl[:] = np.clip(tau, -TAU_LIMIT, TAU_LIMIT)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--floating", action="store_true",
                    help="keep the freejoint; robot topples since there is no balance controller")
    ap.add_argument("--still", action="store_true", help="hold the home pose, no arm wave")
    ap.add_argument("--headless", type=float, metavar="SECONDS", default=None,
                    help="run without a window for N seconds and print tracking error")
    args = ap.parse_args()

    model = load_model(pin_base=not args.floating)
    data = mujoco.MjData(model)
    jm = JointMap(model)
    home = home_pose(jm)

    # Start from the home pose rather than the XML default, so the PD loop is not
    # fighting a large initial error on the first step.
    data.qpos[jm.qadr] = home

    
    mujoco.mj_forward(model, data)

    def target_at(t: float) -> np.ndarray:
        return home if args.still else wave_targets(jm, home, t)

    if args.headless is not None:
        while data.time < args.headless:
            control(model, data, jm, target_at(data.time))
            mujoco.mj_step(model, data)
        err = np.rad2deg(np.abs(target_at(data.time) - data.qpos[jm.qadr]))
        print(f"t={data.time:.2f}s  nq={model.nq} nv={model.nv}  finite={np.isfinite(data.qpos).all()}")
        print(f"tracking error (deg): max={err.max():.2f}  mean={err.mean():.2f}")
        sp = jm.index("left_shoulder_pitch_joint")
        print(f"left_shoulder_pitch: target={np.rad2deg(target_at(data.time)[sp]):7.2f}  "
              f"actual={np.rad2deg(data.qpos[jm.qadr[sp]]):7.2f}")
        return 0

    print(f"model: {MJCF.name}   base: {'floating' if args.floating else 'pinned'}   "
          f"nq={model.nq} nv={model.nv} nu={model.nu}")
    print("left-drag orbit | scroll zoom | ctrl+right-drag push a body | Space pause | Esc quit")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        wall_start = time.perf_counter()
        while viewer.is_running():
            control(model, data, jm, target_at(data.time))
            mujoco.mj_step(model, data)
            viewer.sync()
            # pace the sim to wall-clock so the wave runs at a natural speed
            lag = data.time - (time.perf_counter() - wall_start)
            if lag > 0:
                time.sleep(lag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
