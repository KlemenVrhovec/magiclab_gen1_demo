"""
Bimanual rotate, version 2: tip the top of the box towards the torso, on V2's grasp.

    python rotate_around_y_axis_v2.py                # interactive viewer
    python rotate_around_y_axis_v2.py --headless     # no window, reports the angle achieved
    python rotate_around_y_axis_v2.py --angle 30     # tip it part way

WHICH AXIS THIS IS. Top of the box towards the torso, bottom away: a rotation about the
left-right horizontal axis. In this model's world frame that is X (the robot faces -Y);
the file is named for the axis as usually pictured.

WHY V1 OF THIS ROTATION ONLY MANAGED ~24 DEG. Both palms sit ON the rotation axis, so
tipping the box moves the hands nowhere -- the whole rotation must come from the palms
rolling about their own normal, and rotate_around_y_axis.py measured the V1 grasp
delivering roughly 24 deg of that before saturating, with 28 deg of palm error at 60
commanded. The bottleneck was not the wrist, it was the BRANCH: V1's full-spread lead-in
lands the IK on a solution with three joints jammed on their limits, and from there the
palm cannot roll.

WHAT V2 CHANGES. pick_box_v2.py's lead-in (half a spread, continuity between waypoints,
elbows pulled down through the null space) lands on a branch where the same IK solves
the grasp EXACTLY -- and, the key measurement, solves it exactly with the palms
pre-rolled anywhere in -15..-30 deg about X, and to 4.9 mm at -50. That pre-roll
direction is precisely "top towards the torso". So this script grasps the box with
square palms on V2's branch, then sweeps the commanded palm attitude from 0 down to
-ANGLE in small warm-started steps, and the box, pinched flat between the palms, comes
with them.

THE SQUEEZE MUST NOT ROLL. pick_box's add_squeeze constrains only hand positions, and
its least-squares step rolls the palms as a side effect -- measured at up to 102 deg of
uncommanded box rotation at 60 mm of over-travel. Here the squeeze pins the palm
attitude (rotation rows with a zero target), and conveniently the pinch axis IS world X
at every tip angle -- rotating about X never moves the palm normals off X -- so the
over-travel direction never needs to change.

What the box actually does is measured and printed at the end of every --headless run.
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
    SQUEEZE,
    TABLE_TOP,
    JointMap,
    build_model,
    control,
    grip_force,
)
from pick_box_v2 import (
    CONTINUITY,
    ELBOW_PULL,
    GRASP_RAISE,
    SPREAD_FRAC,
    ReachArms,
    home_hand,
    rot_x,
)
from pick_box import SPREAD_X, SPREAD_Y, SPREAD_Z, hand_geom

# --- making the grip able to carry a twist at all ---
# Measured mid-sweep, the box is held by exactly TWO contact points, one per palm, at
# x = +0.037 and -0.036 with the same y and z. They lie on a straight line, and that
# line is the axis being rotated about: the box is skewered on it and spins freely. It
# is not like that at the grasp -- there are 5 contacts spread over 5 cm at t=9 -- but
# as the palms roll, flat-on-flat alignment is lost and each side collapses to one
# dug-in point (penetration 8 mm -> 18 mm).
#
# With one point per palm, ordinary sliding friction has no lever arm about that axis,
# so the only thing that could carry the twist is TORSIONAL friction -- and both geoms
# ship with condim=3, which has no torsional component at all. Raising condim to 4 adds
# it. MuJoCo takes the element-wise MAXIMUM of the two geoms' friction, so the torsional
# coefficient has to be set on both to take effect (the palms ship with 1.0, which would
# otherwise win).
CONDIM = 4
TORSION = 0.10       # torsional friction coefficient; has units of length


def build_model_gripping(torsion=TORSION):
    """pick_box's scene, with contacts that can transmit a twist.

    The other way to attack this would be to get MORE contact points across the palm, so
    ordinary sliding friction gets a lever arm about the roll axis -- MuJoCo's multiccd
    flag does exactly that for convex meshes. It does not exist in this build (3.11), and
    the palms are meshes, so torsional friction is the lever available here.
    """
    model = build_model()
    for gid in (model.geom("box_geom").id,
                hand_geom(model, "left"), hand_geom(model, "right")):
        model.geom_condim[gid] = CONDIM
        model.geom_friction[gid][1] = torsion
    return model


# --- gripping with the flat of the hand, not its bottom edge ---
# The palms only need pick_box_v2's raised height while they are coming IN over the
# table; once they are beside the box they can drop, and they have to. At the approach
# height the palm centre is 1.33 and the palm spans 1.195..1.465 while the box spans
# 1.100..1.300, so they overlap only over 1.195..1.300 -- the box hangs off the palm's
# bottom 10 cm, which is why the contact collapses to one dug-in point per side.
#
# How far down is set by the table: the palm hangs 13.5 cm below its centre, so a palm
# centre of 1.24 puts its bottom edge at 1.105, just clear of the 1.100 tabletop. That
# overlaps the box over 1.105..1.300 -- 19.5 cm, essentially the whole box height
# against the flat of the hand. Dropping further would dig the palm into the table.
GRASP_LOWER = 0.090      # metres the hands drop from the approach height before closing

# Metres to pull the grasp back towards the body before closing. Measured contacts on
# the way in land at y = -0.418 while the box's front face is only at -0.405: the palms
# bite near the box's leading edge, and that off-centre bite yaws the box as the grip
# closes. Shifting the pair back centres the bite on the face.
GRASP_Y_SHIFT = 0.020

# Where the tip pivots. 0 puts the axis through both palm centres, which is what the
# first version did -- and because both hands sit ON that axis they never move, so the
# whole rotation has to come from the wrists twisting against the box, which is what
# stalled at 19 deg. Dropping the pivot to the BOX's centre instead makes the palms
# swing on an arc around it: shoulder and elbow carry the hands round while the wrists
# roll to match, and the grip stays square to the box the whole way. That is the same
# mechanism that makes rotate_around_z_axis.py work, where the hands orbit the box.
# The palms sit ~40 mm above the box centre at the lowered grasp.
PIVOT_DROP = 0.040

# --- the tip itself ---
TIP_DEG = 45.0       # positive brings the TOP of the box towards the torso
TIP_STEPS = 6        # IK waypoints across the sweep
TIP_SECS = 6.0       # seconds spent tipping
HOLD_SECS = 3.0      # seconds held at the end, to show the grip survived

LEAD_IN = [("home", 0.0), ("spread", 2.5), ("approach", 5.0), ("close", 6.5),
           ("lift", 9.0)]


def timeline(total_deg: float, steps: int, secs: float):
    """(phases, palm roll about X per phase). User sense: +deg tips towards the torso,
    which is a NEGATIVE rotation about world X."""
    phases = list(LEAD_IN)
    roll = {name: 0.0 for name, _ in phases}
    t0 = phases[-1][1]
    for i in range(1, steps + 1):
        name = f"tip{i}"
        phases.append((name, t0 + secs * i / steps))
        roll[name] = np.deg2rad(-total_deg * i / steps)
    phases.append(("hold", phases[-1][1] + HOLD_SECS))
    roll["hold"] = roll[phases[-2][0]]
    return phases, roll


def hand_targets(phase, spread_pt, roll=0.0, lower=GRASP_LOWER,
                 y_shift=GRASP_Y_SHIFT, pivot_drop=PIVOT_DROP):
    """World targets for the two hand geom centres (left, right).

    "approach" stays at pick_box_v2's raised height, which is what gets the palms in
    over the table without catching on it. Everything from "close" onwards drops by
    `lower` and pulls back by `y_shift`, so the box is held against the flat of the
    hand and bitten centrally.

    During the tip the pair is rotated about a pivot `pivot_drop` BELOW the palms --
    the box's centre -- so the hands swing on an arc instead of spinning in place. The
    +/-X offset between the two hands is unchanged by a rotation about X, so only the
    centre moves.
    """
    if phase == "spread":
        return (np.array([spread_pt[0], spread_pt[1], spread_pt[2]]),
                np.array([-spread_pt[0], spread_pt[1], spread_pt[2]]))
    if phase == "approach":
        centre = np.array([0.0, BOX_Y, GRASP_Z + GRASP_RAISE])
        gap = OPEN_X
    else:
        rise = LIFT_H if phase != "close" else 0.0
        centre = np.array([0.0, BOX_Y + y_shift, GRASP_Z + GRASP_RAISE - lower + rise])
        gap = GRIP_X
        pivot = centre - np.array([0.0, 0.0, pivot_drop])
        centre = pivot + rot_x(roll) @ (centre - pivot)
    return centre + [gap, 0, 0], centre - [gap, 0, 0]


def add_squeeze(model, jm, arms, pose, amount):
    """Inward over-travel along the pinch axis with the palm attitude pinned.

    One Jacobian step commanding both hands towards each other along X, with rotation
    rows targeting zero so the step cannot roll the palms. The box blocks the motion,
    so the PD turns the leftover joint error into grip force.
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


def plan(model, jm, phases, roll, squeeze=0.0, lower=GRASP_LOWER,
         y_shift=GRASP_Y_SHIFT, pivot_drop=PIVOT_DROP, verbose=True) -> dict:
    """Solve IK once per waypoint on V2's branch; return full 30-vector joint targets."""
    scratch = mujoco.MjData(model)
    arms = ReachArms(model)

    home = np.zeros(model.nu)
    home[jm.index("left_shoulder_roll_joint")] = 0.25
    home[jm.index("right_shoulder_roll_joint")] = -0.25

    full = np.array([SPREAD_X, SPREAD_Y, SPREAD_Z])
    h0 = home_hand(model, jm, home)
    spread_pt = h0 + SPREAD_FRAC * (full - h0)

    poses = {"home": home}
    seed = np.deg2rad([-20, 70, -20, -40, 0, 0, 0, -20, -70, 20, 40, 0, 0, 0])
    if verbose:
        print(f"  {'waypoint':<10} {'roll':>6}  {'pos err L/R (mm)':>17}")
    for name, _ in phases[1:]:
        if name == "hold":
            continue
        tl, tr = hand_targets(name, spread_pt, roll[name], lower,
                              y_shift, pivot_drop)
        mujoco.mj_resetData(model, scratch)
        scratch.qpos[jm.qadr] = home
        scratch.qpos[arms.qids] = np.clip(seed, arms.lo, arms.hi)
        q, el, er = arms.solve(scratch, tl, tr, roll[name],
                               elbow_pull=ELBOW_PULL, q_ref=seed, cont=CONTINUITY)
        full_pose = home.copy()
        for n, v in zip(arms.names, q):
            full_pose[jm.index(n)] = v
        poses[name] = full_pose
        seed = q                                   # warm-start the next waypoint
        if verbose:
            print(f"  {name:<10} {np.degrees(roll[name]):5.0f}d  "
                  f"{el*1000:8.1f} {er*1000:8.1f}")

    if squeeze > 0:
        for name in poses:
            if name in ("close", "lift") or name.startswith("tip"):
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


def box_contacts(model, data):
    """How many contact points hold the box, ignoring the table. One per palm means it
    is skewered on the pinch axis and free to spin; a handful means a real patch."""
    bg = model.geom("box_geom").id
    tg = model.geom("table_geom").id
    n = 0
    for i in range(data.ncon):
        c = data.contact[i]
        if bg in (c.geom1, c.geom2):
            other = c.geom2 if c.geom1 == bg else c.geom1
            if other != tg:
                n += 1
    return n


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
    ap.add_argument("--y-shift", type=float, default=GRASP_Y_SHIFT,
                    help="metres to pull the grasp back towards the body")
    ap.add_argument("--pivot-drop", type=float, default=PIVOT_DROP,
                    help="metres below the palms that the tip pivots about")
    ap.add_argument("--lower", type=float, default=GRASP_LOWER,
                    help="metres the hands drop from the approach height before closing")
    ap.add_argument("--torsion", type=float, default=TORSION,
                    help="torsional friction coefficient at the palm/box contacts")
    ap.add_argument("--duration", type=float, default=None, help="default: the full timeline")
    args = ap.parse_args()

    phases, roll = timeline(args.angle, args.steps, args.tip_secs)
    duration = args.duration if args.duration else phases[-1][1] + 1.0

    model = build_model_gripping(args.torsion)
    jm = JointMap(model)
    print(f"planning V2 grasp + {args.angle:.0f} deg tip (top towards the torso):")
    poses = plan(model, jm, phases, roll, squeeze=args.squeeze, lower=args.lower,
                 y_shift=args.y_shift, pivot_drop=args.pivot_drop)

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
        rows, nxt = [], 0.0
        lowest = np.inf                         # lowest the box gets once it is airborne
        peak = z0
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
                             grip_force(model, data), box_contacts(model, data)))
        print(f"\n{'t':>5} {'phase':<16} {'box z':>7} {'box tip':>9} {'grip N':>8}"
              f" {'contacts':>9}")
        for t, p, z, r, f, n in rows:
            print(f"{t:5.1f} {p:<16} {z:7.3f} {r:8.1f}d {f:8.1f} {n:9d}")

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
