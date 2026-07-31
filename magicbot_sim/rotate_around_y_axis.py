"""
Bimanual rotate: pick the box up, then tip its top towards the robot's torso.

    python rotate_around_y_axis.py                # interactive viewer
    python rotate_around_y_axis.py --headless     # no window, reports the angle achieved
    python rotate_around_y_axis.py --angle 30     # tip it part way
    python rotate_around_y_axis.py --angle -45    # tip the top away from the torso

WHICH AXIS THIS IS. The top of the box moves towards the torso and the bottom moves
away, so the box turns about the left-right horizontal axis. In this model's world frame
that is X, not Y: the robot faces world -Y, so -Y is "forwards", +Y is "towards the
body", and X runs from the robot's right hand to its left. The file is named for the
axis as it is usually pictured; the code says X because that is what the frame calls it.

WHY THIS ONE IS HARD, NOT SIMPLE. It looks like the easiest of the three rotations and
it is by far the worst. The reason is that X is the PINCH axis -- the line joining the
two palms. Both hand centres sit on it, so turning the box about it does not move the
hands anywhere at all: hand_targets() below returns the same two points at every angle.
The entire rotation has to come from the palms spinning about their own normals, which
is the one thing this arm cannot do:

  * the joint that spins a palm about its own normal is wrist_roll, and its axis runs
    along the forearm. In the grasp pose the forearms point forwards, along -Y, so
    wrist_roll turns the palm about Y -- useless here -- and its -55..+41 deg of travel
    buys nothing;
  * shoulder_roll DOES turn about world X directly, but it swings the whole arm through
    an arc about the shoulder, which drags the hand off the box. It is also already
    pinned at its limit just reaching the box near the body midline;
  * wrist_pitch, the remaining candidate, spans +/-18 deg and is likewise pinned.

Compare rotate_around_z_axis.py, which works: turning about the vertical moves each hand
a quarter circle round the box, and shoulder_yaw (+/-142 deg) and wrist_yaw (+/-157 deg)
both drive it directly with room to spare.

WHAT THIS SCRIPT ACTUALLY ACHIEVES is printed at the end of every --headless run, as the
box's measured rotation, not the commanded one. Do not trust the --angle you asked for.

Everything else -- the model, the controller, the grasp geometry, the approach path, the
squeeze that turns leftover joint error into grip force -- is imported unchanged from
pick_box.py.
"""

import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np

from pick_box import (
    BOX_Y,
    GRASP_Z,
    GRIP_X,
    LIFT_H,
    OPEN_X,
    SPREAD_X,
    SPREAD_Y,
    SPREAD_Z,
    SQUEEZE,
    TABLE_TOP,
    Arms,
    JointMap,
    build_model,
    control,
    grip_force,
)

# --- the tip itself ---
TIP_DEG = 45.0       # positive brings the TOP of the box towards the torso
TIP_STEPS = 6        # IK waypoints across the sweep
TIP_SECS = 6.0       # seconds spent tipping
HOLD_SECS = 3.0      # seconds held at the end, to show the grip survived

# Waypoints before the tip starts. Same route as pick_box: out to the sides and up,
# clear of the table, then forward and inward onto the box.
LEAD_IN = [("home", 0.0), ("spread", 2.5), ("approach", 5.0), ("close", 6.5), ("lift", 9.0)]

GRIPPING = ("close", "lift")


def timeline(total_deg: float, steps: int, secs: float):
    """(phases, tip angle per phase). Phases are (name, start time) in order."""
    phases = list(LEAD_IN)
    tip = {name: 0.0 for name, _ in phases}
    t0 = phases[-1][1]
    for i in range(1, steps + 1):
        name = f"tip{i}"
        phases.append((name, t0 + secs * i / steps))
        tip[name] = np.deg2rad(total_deg * i / steps)
    last = phases[-1][0]
    phases.append(("hold", phases[-1][1] + HOLD_SECS))
    tip["hold"] = tip[last]
    return phases, tip


def rot_x(angle: float) -> np.ndarray:
    """Rotation about the left-right axis. A NEGATIVE angle carries +Z towards +Y,
    i.e. tips the top of the box towards the torso -- see tip_of() for the sign."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1.0, 0.0, 0.0],
                     [0.0, c, -s],
                     [0.0, s, c]])


def tip_of(deg: float) -> float:
    """--angle in the user's sense (+ = top towards the torso) -> a rotation about X."""
    return np.deg2rad(-deg)


def hand_targets(phase, tip=0.0):
    """World targets for the two hand geom centres (left, right).

    Note what the tip argument does to the POSITIONS: nothing. Both hands lie on the
    rotation axis, so rot_x(tip) @ [gap, 0, 0] == [gap, 0, 0] for every angle. That is
    the whole difficulty of this rotation, made explicit -- see the module docstring.
    """
    if phase == "spread":
        centre, gap, tip = np.array([0.0, SPREAD_Y, SPREAD_Z]), SPREAD_X, 0.0
    elif phase == "approach":
        centre, gap = np.array([0.0, BOX_Y, GRASP_Z]), OPEN_X
    elif phase == "close":
        centre, gap = np.array([0.0, BOX_Y, GRASP_Z]), GRIP_X
    else:                                     # lift, tip1..N, hold
        centre, gap = np.array([0.0, BOX_Y, GRASP_Z + LIFT_H]), GRIP_X
    offset = rot_x(tip) @ np.array([gap, 0.0, 0.0])
    return centre + offset, centre - offset


class TipArms(Arms):
    """pick_box.Arms, but the IK target attitude can be rolled about the pinch axis.

    Arms.solve() hardwires the palms to the identity attitude; here that becomes
    Rx(tip). Everything else -- damped least squares, the joint-limit clamp, aiming at
    the geom centre via mj_jac rather than the body origin -- is unchanged.

    The palm normal is unchanged by Rx, so there is no point offering the normal-only
    relaxation that rotate_around_z_axis.py has: the entire commanded change here IS the
    spin about the normal, and relaxing it would ask for nothing at all.
    """

    def solve(self, data, tl, tr, tip=0.0, iters=3000, lam=0.10, step=0.5, w_rot=0.30):
        """Returns (angles, pos err L, pos err R, palm err L deg, palm err R deg)."""
        q_tgt = np.zeros(4)
        mujoco.mju_mat2Quat(q_tgt, rot_x(tip).flatten())

        for _ in range(iters):
            mujoco.mj_forward(self.model, data)
            rows, errs = [], []
            for body, geom, tgt in zip(self.bodies, self.geoms, (tl, tr)):
                point = data.geom_xpos[geom].copy()
                jp = np.zeros((3, self.model.nv))
                jr = np.zeros((3, self.model.nv))
                mujoco.mj_jac(self.model, data, jp, jr, point, body)
                rows.append(jp[:, self.dofs])
                errs.append(tgt - point)

                rows.append(jr[:, self.dofs])
                errs.append(w_rot * self.attitude_err(data, geom, q_tgt))

            J = np.vstack(rows)
            e = np.concatenate(errs)
            if np.linalg.norm(e) < 1e-4:
                break
            dq = J.T @ np.linalg.solve(J @ J.T + lam**2 * np.eye(J.shape[0]), e)
            data.qpos[self.qids] = np.clip(data.qpos[self.qids] + step * dq, self.lo, self.hi)

        mujoco.mj_forward(self.model, data)
        return (data.qpos[self.qids].copy(),
                np.linalg.norm(tl - data.geom_xpos[self.geoms[0]]),
                np.linalg.norm(tr - data.geom_xpos[self.geoms[1]]),
                np.degrees(np.linalg.norm(self.attitude_err(data, self.geoms[0], q_tgt))),
                np.degrees(np.linalg.norm(self.attitude_err(data, self.geoms[1], q_tgt))))

    def attitude_err(self, data, geom, q_tgt):
        """Rotation vector taking this palm to the target attitude (rad)."""
        q_cur, q_inv, q_err, w = np.zeros(4), np.zeros(4), np.zeros(4), np.zeros(3)
        mujoco.mju_mat2Quat(q_cur, data.geom_xmat[geom])
        mujoco.mju_negQuat(q_inv, q_cur)
        mujoco.mju_mulQuat(q_err, q_tgt, q_inv)
        mujoco.mju_quat2Vel(w, q_err, 1.0)
        return w


def add_squeeze(model, jm, arms, pose, amount):
    """Bias the commanded joint angles inward, past where the hands can actually go,
    WITHOUT letting the palms roll while doing it.

    pick_box's version of this constrains only the hand positions -- it calls mj_jac
    with no rotation Jacobian -- so the least-squares step is free to rotate the palms
    as a side effect, and nothing notices because pick_box never measures the box's
    orientation. Here it matters enormously, because a palm roll about X is exactly the
    motion this script is trying to command. Measured with the box held and a tip of
    ZERO commanded, pick_box's squeeze alone turns the box 17 deg at 30 mm of
    over-travel, 48 deg at 45 mm and 102 deg at 60 mm: the box was being rolled by the
    grip, not by the plan.

    Adding the rotation rows with a zero target pins the palm attitude through the
    squeeze, so the over-travel becomes pure translation and the only thing that turns
    the box is the tip that was actually asked for.
    """
    scratch = mujoco.MjData(model)
    scratch.qpos[jm.qadr] = pose
    mujoco.mj_forward(model, scratch)

    rows, errs = [], []
    for body, geom, inward in zip(arms.bodies, arms.geoms, (-1.0, +1.0)):
        jp = np.zeros((3, model.nv))
        jr = np.zeros((3, model.nv))
        mujoco.mj_jac(model, scratch, jp, jr, scratch.geom_xpos[geom].copy(), body)
        rows.append(jp[:, arms.dofs])
        errs.append(np.array([inward * amount, 0.0, 0.0]))
        rows.append(jr[:, arms.dofs])
        errs.append(np.zeros(3))               # hold this palm's attitude still

    J = np.vstack(rows)
    e = np.concatenate(errs)
    dq = J.T @ np.linalg.solve(J @ J.T + 0.05**2 * np.eye(J.shape[0]), e)

    out = pose.copy()
    biased = np.clip(scratch.qpos[arms.qids] + dq, arms.lo, arms.hi)
    for n, v in zip(arms.names, biased):
        out[jm.index(n)] = v
    return out


def plan(model, jm, phases, tip, squeeze=0.0, verbose=True) -> dict:
    """Solve IK once per waypoint; return full 30-vector joint targets."""
    scratch = mujoco.MjData(model)
    arms = TipArms(model)

    home = np.zeros(model.nu)
    home[jm.index("left_shoulder_roll_joint")] = 0.25
    home[jm.index("right_shoulder_roll_joint")] = -0.25

    poses = {"home": home}
    # seed with the arms already out sideways -- keeps the IK off the elbow-backwards
    # solutions it otherwise falls into on the very first waypoint
    seed = np.deg2rad([-20, 70, -20, -40, 0, 0, 0, -20, -70, 20, 40, 0, 0, 0])
    if verbose:
        print(f"  {'waypoint':<10} {'tip':>5}  {'pos err L/R (mm)':>17}  "
              f"{'palm err L/R (deg)':>19}  {'step':>6}")
    for name, _ in phases[1:]:
        if name == "hold":
            continue
        tl, tr = hand_targets(name, tip[name])
        mujoco.mj_resetData(model, scratch)
        scratch.qpos[jm.qadr] = home
        scratch.qpos[arms.qids] = np.clip(seed, arms.lo, arms.hi)
        q, el, er, rl, rr = arms.solve(scratch, tl, tr, tip[name])
        full = home.copy()
        for n, v in zip(arms.names, q):
            full[jm.index(n)] = v
        poses[name] = full
        jump = np.degrees(np.abs(q - seed).max())
        seed = q                                   # warm-start the next waypoint
        if verbose:
            print(f"  {name:<10} {np.degrees(tip[name]):4.0f}d  "
                  f"{el*1000:8.1f} {er*1000:8.1f}  {rl:9.1f} {rr:9.1f}  {jump:5.0f}d")

    if squeeze > 0:
        # the palm normals stay on world X at every tip angle, so the squeeze direction
        # never has to change -- but it does have to leave the palm attitude alone
        for name in poses:
            if name in GRIPPING or name.startswith("tip"):
                poses[name] = add_squeeze(model, jm, arms, poses[name], squeeze)
        if verbose:
            print(f"  squeeze bias: {squeeze*1000:.0f} mm of inward over-travel")

    poses["hold"] = poses[phases[-2][0]]
    return poses


def target_at(t, poses, phases):
    """Cosine-eased interpolation between waypoint joint vectors."""
    if t <= phases[0][1]:
        return poses[phases[0][0]], phases[0][0]
    for (n0, t0), (n1, t1) in zip(phases, phases[1:]):
        if t < t1:
            s = (t - t0) / (t1 - t0)
            s = 0.5 - 0.5 * np.cos(np.pi * s)
            return poses[n0] * (1 - s) + poses[n1] * s, f"{n0}->{n1}"
    return poses[phases[-1][0]], phases[-1][0]


def box_tip(model, data, q0) -> tuple:
    """(total rotation of the box since q0, and the tip in the user's sense) in deg.

    Positive tip = the top of the box has moved towards the torso.
    """
    q_inv, q_rel, w = np.zeros(4), np.zeros(4), np.zeros(3)
    mujoco.mju_negQuat(q_inv, q0)
    mujoco.mju_mulQuat(q_rel, data.xquat[model.body("box").id], q_inv)
    mujoco.mju_quat2Vel(w, q_rel, 1.0)
    return np.degrees(np.linalg.norm(w)), np.degrees(-w[0])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--headless", action="store_true", help="no window; report the outcome")
    ap.add_argument("--angle", type=float, default=TIP_DEG,
                    help="degrees to tip the box; positive brings the TOP towards the torso")
    ap.add_argument("--steps", type=int, default=TIP_STEPS, help="IK waypoints in the tip")
    ap.add_argument("--tip-secs", type=float, default=TIP_SECS)
    ap.add_argument("--squeeze", type=float, default=SQUEEZE,
                    help="metres of inward over-travel commanded into the box (grip force)")
    ap.add_argument("--duration", type=float, default=None, help="default: the full timeline")
    args = ap.parse_args()

    phases, tip = timeline(args.angle, args.steps, args.tip_secs)
    tip = {k: tip_of(np.degrees(v)) for k, v in tip.items()}   # user sense -> about X
    duration = args.duration if args.duration else phases[-1][1] + 1.0

    model = build_model()
    jm = JointMap(model)
    print(f"planning grasp + {args.angle:.0f} deg tip (top towards the torso):")
    poses = plan(model, jm, phases, tip, squeeze=args.squeeze)

    data = mujoco.MjData(model)
    data.qpos[jm.qadr] = poses["home"]
    mujoco.mj_forward(model, data)

    box = model.body("box").id
    z0 = data.xpos[box][2]
    q0 = data.xquat[box].copy()
    print(f"\ntable top {TABLE_TOP:.3f} m, box centre starts at z={z0:.3f} m")

    def step():
        tgt, phase = target_at(data.time, poses, phases)
        control(model, data, jm, tgt)
        mujoco.mj_step(model, data)
        return phase

    if args.headless:
        rows, nxt, peak = [], 0.0, z0
        lowest = np.inf                         # lowest the box gets once it is airborne
        lift_t = phases[len(LEAD_IN) - 1][1]
        while data.time < duration:
            phase = step()
            peak = max(peak, data.xpos[box][2])
            if data.time > lift_t:
                lowest = min(lowest, data.xpos[box][2])
            if data.time >= nxt:
                nxt += 1.0
                _, tipped = box_tip(model, data, q0)
                rows.append((data.time, phase, data.xpos[box][2], tipped,
                             grip_force(model, data)))
        print(f"\n{'t':>5} {'phase':<16} {'box z':>7} {'box tip':>9} {'grip N':>8}")
        for t, p, z, r, f in rows:
            print(f"{t:5.1f} {p:<16} {z:7.3f} {r:8.1f}d {f:8.1f}")

        total, tipped = box_tip(model, data, q0)
        print(f"\nbox rose {(peak - z0)*1000:.0f} mm above its start height")
        print(f"box rotated {total:.1f} deg total, {tipped:.1f} deg of it as a tip "
              f"towards the torso (commanded {args.angle:.0f})")
        print(f"lowest the box got after the lift: z={lowest:.3f} m (table top {TABLE_TOP:.3f})")
        held = lowest > TABLE_TOP               # never dropped back onto the table
        turned = abs(tipped) > abs(args.angle) * 0.6
        print("RESULT:", "TIPPED" if (held and turned) else
              ("DROPPED" if not held else "held but under-rotated"))
        return 0 if (held and turned) else 1

    print("left-drag orbit | scroll zoom | ctrl+right-drag push | Space pause | Esc quit\n")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        wall = time.perf_counter()
        while viewer.is_running():
            step()
            viewer.sync()
            lag = data.time - (time.perf_counter() - wall)
            if lag > 0:
                time.sleep(lag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
