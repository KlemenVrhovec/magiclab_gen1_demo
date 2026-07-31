"""
Bimanual rotate: pick the box up, then turn it towards the body without letting go.

    python rotate_around_z_axis.py                # interactive viewer
    python rotate_around_z_axis.py --headless     # no window, reports the angle achieved
    python rotate_around_z_axis.py --angle 45     # turn it part way
    python rotate_around_z_axis.py --angle -90    # turn it the other way

This is pick_box.py plus one extra idea, and it imports that file rather than repeating
it: the model, the controller, the grasp geometry and the IK all come from there.

WHAT IT DOES. The box is picked up exactly as in pick_box, then the whole grasp -- both
hands together -- is rotated about the vertical axis through the box. The palms start
facing each other across X and end up facing fore and aft along Y, so the face of the
box that was on the left is turned towards the body. The hands stay pressed on the same
two faces the whole way, so the grip is never broken and never regripped: each hand
simply walks a quarter circle round the box while pivoting to match.

WHY ABOUT THE VERTICAL AXIS. Two earlier attempts failed on kinematics, both measured:

  * rolling the box about the pinch axis X (tumbling it forwards) needs the palms to
    roll about their own normal. The only joints that can are wrist_roll (-55..+41 deg)
    and wrist_pitch (+/-18 deg), and both are already spent holding the box near the
    body midline. The IK's orientation error grew one-for-one with the commanded angle
    -- 29 deg of error at 15 deg commanded, 103 deg at 90 -- i.e. no roll at all, with
    shoulder_roll and wrist_pitch pinned at their limits in every solution.

  * tipping it about Y (drop one hand, raise the other) worked for the left arm, which
    tracked to 9 mm, but not the right, which missed by 131 mm. Both palms have to
    reach the SAME attitude in world coordinates, and the arms are mirror images with
    mirrored joint ranges, so one of them is always pitching the way its wrist cannot.

  Vertical is different: shoulder_yaw (+/-142 deg) and wrist_yaw (+/-157 deg) both turn
  about it directly and neither is anywhere near its limit in the grasp pose. Gravity
  also stays out of the way -- the palms remain vertical throughout, so the box is held
  by exactly the friction pinch that pick_box already proved, at every angle.

WHY IT GOES IN SMALL STEPS. The turn is split into TURN_STEPS waypoints, each solved by
IK warm-started from the previous one. That keeps the arms on one continuous branch of
the solution -- a single 90 degree jump lets the IK find a mirrored pose, and the
joint-space interpolation then swings the box through the table.

THE SQUEEZE HAS TO TURN TOO. pick_box turns leftover position error into grip force by
commanding the hands past each other along world X. Once the grasp has turned, "towards
the other hand" is no longer world X, so the over-travel is applied along the current
palm normal instead. See add_squeeze().
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

# --- the turn itself ---
TURN_DEG = 90.0      # palms end facing fore and aft instead of side to side
TURN_STEPS = 6       # IK waypoints across the sweep (15 deg apart at the default angle)
TURN_SECS = 6.0      # seconds spent turning
HOLD_SECS = 3.0      # seconds held at the end, to show the grip survived

# Waypoints before the turn starts. Same route as pick_box: out to the sides and up,
# clear of the table, then forward and inward onto the box.
LEAD_IN = [("home", 0.0), ("spread", 2.5), ("approach", 5.0), ("close", 6.5), ("lift", 9.0)]

# Phases that are already gripping the box, so they get the squeeze bias.
GRIPPING = ("close", "lift")


def timeline(total_deg: float, steps: int, secs: float):
    """(phases, turn angle per phase). Phases are (name, start time) in order."""
    phases = list(LEAD_IN)
    turn = {name: 0.0 for name, _ in phases}
    t0 = phases[-1][1]
    for i in range(1, steps + 1):
        name = f"turn{i}"
        phases.append((name, t0 + secs * i / steps))
        turn[name] = np.deg2rad(total_deg * i / steps)
    last = phases[-1][0]
    phases.append(("hold", phases[-1][1] + HOLD_SECS))
    turn["hold"] = turn[last]
    return phases, turn


def rot_z(angle: float) -> np.ndarray:
    """Rotation about the vertical axis -- the axis the whole grasp turns around."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0],
                     [s, c, 0.0],
                     [0.0, 0.0, 1.0]])


def hand_targets(phase, turn=0.0):
    """World targets for the two hand geom centres (left, right).

    Every phase from the lift onwards shares one centre point; the turn only walks the
    two hands around it, which is what rotates the box in place instead of carrying it.
    At turn=0 the offset is +/-X, exactly pick_box. At +90 deg the left hand has swung
    round to +Y -- the near side, towards the body -- and the right hand to -Y.
    """
    if phase == "spread":
        centre, gap, turn = np.array([0.0, SPREAD_Y, SPREAD_Z]), SPREAD_X, 0.0
    elif phase == "approach":
        centre, gap = np.array([0.0, BOX_Y, GRASP_Z]), OPEN_X
    elif phase == "close":
        centre, gap = np.array([0.0, BOX_Y, GRASP_Z]), GRIP_X
    else:                                     # lift, turn1..N, hold
        centre, gap = np.array([0.0, BOX_Y, GRASP_Z + LIFT_H]), GRIP_X
    offset = rot_z(turn) @ np.array([gap, 0.0, 0.0])
    return centre + offset, centre - offset


class TurnArms(Arms):
    """pick_box.Arms, but the IK target attitude can be turned about the vertical.

    Arms.solve() hardwires the palms' thin axis to world X with no turn about vertical.
    Here that identity target becomes Rz(turn), so the palms pivot to stay flush against
    the same two box faces as they walk round it. Everything else -- damped least
    squares, the joint-limit clamp, aiming at the geom centre via mj_jac rather than the
    body origin -- is unchanged.

    free_spin drops the constraint on how the palm is rotated ABOUT its own normal. A
    flat palm on a flat face does not care, so this is the honest 2-DOF version of the
    orientation task and it gives each arm a spare DOF. It is off by default because the
    full attitude is reachable for a vertical turn and keeps the 27 cm palm slab hanging
    downwards where pick_box's clearances assume it is.
    """

    def solve(self, data, tl, tr, turn=0.0, free_spin=False,
              iters=3000, lam=0.10, step=0.5, w_rot=0.30):
        """Returns (angles, pos err L, pos err R, palm err L deg, palm err R deg)."""
        R_tgt = rot_z(turn)
        q_tgt = np.zeros(4)
        mujoco.mju_mat2Quat(q_tgt, R_tgt.flatten())
        err = ((lambda d, g: self.normal_err(d, g, R_tgt[:, 0])) if free_spin else
               (lambda d, g: self.attitude_err(d, g, q_tgt)))

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
                errs.append(w_rot * err(data, geom))

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
                np.degrees(np.linalg.norm(err(data, self.geoms[0]))),
                np.degrees(np.linalg.norm(err(data, self.geoms[1]))))

    def attitude_err(self, data, geom, q_tgt):
        """Rotation vector taking this palm to the target attitude (rad)."""
        q_cur, q_inv, q_err, w = np.zeros(4), np.zeros(4), np.zeros(4), np.zeros(3)
        mujoco.mju_mat2Quat(q_cur, data.geom_xmat[geom])
        mujoco.mju_negQuat(q_inv, q_cur)
        mujoco.mju_mulQuat(q_err, q_tgt, q_inv)
        mujoco.mju_quat2Vel(w, q_err, 1.0)
        return w

    def normal_err(self, data, geom, n_des):
        """Smallest rotation vector bringing this palm's normal onto +/-n_des (rad)."""
        n = data.geom_xmat[geom].reshape(3, 3)[:, 0]      # palm thin axis, in world
        if np.dot(n, n_des) < 0.0:                        # the far face is the near one
            n_des = -n_des
        axis = np.cross(n, n_des)
        s = np.linalg.norm(axis)
        if s < 1e-9:
            return np.zeros(3)
        return axis / s * np.arctan2(s, float(np.dot(n, n_des)))


def add_squeeze(model, jm, arms, pose, amount, turn=0.0):
    """Bias the commanded joint angles inward, past where the hands can actually go.

    pick_box.add_squeeze does this along world X; once the grasp has turned, the two
    palms face each other along Rz(turn) @ X instead, so the over-travel has to be
    applied along that. Everything else is the same trick: one Jacobian step in the
    'both hands move towards each other' direction, added to the commanded pose. The box
    blocks the motion, so the PD turns the permanent joint error into grip force --
    amount is metres of virtual over-travel, not a force.
    """
    scratch = mujoco.MjData(model)
    scratch.qpos[jm.qadr] = pose
    mujoco.mj_forward(model, scratch)

    normal = rot_z(turn) @ np.array([1.0, 0.0, 0.0])
    rows, errs = [], []
    for body, geom, inward in zip(arms.bodies, arms.geoms, (-1.0, +1.0)):
        jp = np.zeros((3, model.nv))
        mujoco.mj_jac(model, scratch, jp, None, scratch.geom_xpos[geom].copy(), body)
        rows.append(jp[:, arms.dofs])
        errs.append(inward * amount * normal)

    J = np.vstack(rows)
    e = np.concatenate(errs)
    dq = J.T @ np.linalg.solve(J @ J.T + 0.05**2 * np.eye(6), e)

    out = pose.copy()
    biased = np.clip(scratch.qpos[arms.qids] + dq, arms.lo, arms.hi)
    for n, v in zip(arms.names, biased):
        out[jm.index(n)] = v
    return out


def plan(model, jm, phases, turn, squeeze=0.0, free_spin=False, verbose=True) -> dict:
    """Solve IK once per waypoint; return full 30-vector joint targets."""
    scratch = mujoco.MjData(model)
    arms = TurnArms(model)

    home = np.zeros(model.nu)
    home[jm.index("left_shoulder_roll_joint")] = 0.25
    home[jm.index("right_shoulder_roll_joint")] = -0.25

    poses = {"home": home}
    # seed with the arms already out sideways -- keeps the IK off the elbow-backwards
    # solutions it otherwise falls into on the very first waypoint
    seed = np.deg2rad([-20, 70, -20, -40, 0, 0, 0, -20, -70, 20, 40, 0, 0, 0])
    if verbose:
        print(f"  {'waypoint':<10} {'turn':>5}  {'pos err L/R (mm)':>17}  "
              f"{'palm err L/R (deg)':>19}  {'step':>6}")
    for name, _ in phases[1:]:
        if name == "hold":
            continue
        tl, tr = hand_targets(name, turn[name])
        mujoco.mj_resetData(model, scratch)
        scratch.qpos[jm.qadr] = home
        scratch.qpos[arms.qids] = np.clip(seed, arms.lo, arms.hi)
        q, el, er, rl, rr = arms.solve(scratch, tl, tr, turn[name], free_spin)
        full = home.copy()
        for n, v in zip(arms.names, q):
            full[jm.index(n)] = v
        poses[name] = full
        # biggest single-joint move since the last waypoint. The interpolation between
        # waypoints is in joint space, so if the IK hops to a mirrored branch here the
        # arms swing through whatever is in between. Past ~60 deg, use more steps.
        jump = np.degrees(np.abs(q - seed).max())
        seed = q                                   # warm-start the next waypoint
        if verbose:
            print(f"  {name:<10} {np.degrees(turn[name]):4.0f}d  "
                  f"{el*1000:8.1f} {er*1000:8.1f}  {rl:9.1f} {rr:9.1f}  {jump:5.0f}d")

    if squeeze > 0:
        for name in poses:
            if name in GRIPPING or name.startswith("turn"):
                poses[name] = add_squeeze(model, jm, arms, poses[name], squeeze, turn[name])
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


def box_turn(model, data, q0) -> tuple:
    """(total rotation of the box since q0 in deg, the part of it about vertical)."""
    q_inv, q_rel, w = np.zeros(4), np.zeros(4), np.zeros(3)
    mujoco.mju_negQuat(q_inv, q0)
    mujoco.mju_mulQuat(q_rel, data.xquat[model.body("box").id], q_inv)
    mujoco.mju_quat2Vel(w, q_rel, 1.0)
    return np.degrees(np.linalg.norm(w)), np.degrees(w[2])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--headless", action="store_true", help="no window; report the outcome")
    ap.add_argument("--angle", type=float, default=TURN_DEG,
                    help="degrees to turn the box about vertical (positive brings the "
                         "left face round towards the body)")
    ap.add_argument("--steps", type=int, default=TURN_STEPS, help="IK waypoints in the turn")
    ap.add_argument("--turn-secs", type=float, default=TURN_SECS)
    ap.add_argument("--squeeze", type=float, default=SQUEEZE,
                    help="metres of inward over-travel commanded into the box (grip force)")
    ap.add_argument("--free-spin", action="store_true",
                    help="constrain only the palm normal, not its spin about the normal")
    ap.add_argument("--duration", type=float, default=None, help="default: the full timeline")
    args = ap.parse_args()

    phases, turn = timeline(args.angle, args.steps, args.turn_secs)
    duration = args.duration if args.duration else phases[-1][1] + 1.0

    model = build_model()
    jm = JointMap(model)
    print(f"planning grasp + {args.angle:.0f} deg turn about vertical:")
    poses = plan(model, jm, phases, turn, squeeze=args.squeeze, free_spin=args.free_spin)

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
                _, about_z = box_turn(model, data, q0)
                rows.append((data.time, phase, data.xpos[box][2], about_z,
                             grip_force(model, data)))
        print(f"\n{'t':>5} {'phase':<16} {'box z':>7} {'box turn':>9} {'grip N':>8}")
        for t, p, z, r, f in rows:
            print(f"{t:5.1f} {p:<16} {z:7.3f} {r:8.1f}d {f:8.1f}")

        total, about_z = box_turn(model, data, q0)
        print(f"\nbox rose {(peak - z0)*1000:.0f} mm above its start height")
        print(f"box turned {total:.1f} deg total, {about_z:.1f} deg of it about vertical "
              f"(commanded {args.angle:.0f})")
        print(f"lowest the box got after the lift: z={lowest:.3f} m (table top {TABLE_TOP:.3f})")
        held = lowest > TABLE_TOP               # never dropped back onto the table
        turned = abs(about_z) > abs(args.angle) * 0.6
        print("RESULT:", "ROTATED" if (held and turned) else
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
