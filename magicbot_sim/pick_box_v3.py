"""
Bimanual grasp, version 3: fingers pointing forward, palms squeezing the box. No lift.

    python pick_box_v3.py               # interactive viewer
    python pick_box_v3.py --headless    # no window, reports the grasp and body clearance

Same robot, table, box and squeeze as pick_box_v2.py. Three things differ.

1. FINGERS FORWARD. V1/V2 hang the hands with the fingers pointing down. Here each hand
   ends up rolled -90 deg about world X -- about its own palm normal -- so the palms still
   face each other and squeeze the box, but the fingers point away from the robot
   (world -Y). That is V2's PREROLL taken all the way: V2 stops at -20 deg.

   With the fingers horizontal the slab's long axis no longer hangs down. Its vertical
   extent is now the hand's WIDTH: 54 mm below the aimed point, 86 mm above it. So the
   grasp sits lower than V2's (GRASP_Z_FWD) and the palm covers ~74 mm of the box side.
   How close to horizontal the fingers get depends on height. Measured at the grip:
   z = 1.25 -> 14 deg short (shoulder_roll and wrist_pitch on their limits), 1.28 ->
   6-8 deg, 1.35 -> exact. 1.35 would leave the palm 4 mm on the box, so 1.28.

2. A HAND PATH, NOT A JOINT BLEND. V1/V2 solve IK at three far-apart waypoints and blend
   the JOINT angles between them. The joints then take whatever route the blend gives,
   and with the fingers-forward attitude that route is ugly: measured, the left hand
   swings inward through the left hip (98 mm deep at t = 0.4 s) while the shoulder twists
   138 deg and the elbow goes behind the back before coming forward again.

   Here the HAND is given the path a person would use -- up and forward beside the body,
   then in towards the box -- and IK tracks it at SAMPLES closely spaced points, each
   warm-started from the last so the arm stays on one configuration branch. The palm
   rolls gradually along the way (see track() for why V2's elbow-down pull is gone). The
   motion is timed with a minimum-jerk profile: it starts and stops
   gently and is fastest in the middle, which is how human reaches are shaped.

   The robot's own geoms do not collide with each other in this model, so nothing in the
   simulation stops an arm going through the body. The headless run therefore checks it
   explicitly (body_clearance) on a probe copy of the model with those contacts enabled.

3. FASTER, AND IT STOPS AT THE GRIP. Reach in 2.0 s, close in 0.5 s, against V2's 6.5 s;
   the hands stay squeezing the box on the table -- no lift for now.
"""

import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np

from pick_box import (
    BOX_Y,
    GRIP_X,
    OPEN_X,
    SQUEEZE,
    TABLE_TOP,
    Arms,
    JointMap,
    add_squeeze,
    build_model,
    control,
    grip_force,
)
from pick_box_v2 import palm_clearance, rot_x

GRASP_Z_FWD = 1.28      # hand geom centre height at the grip; see 1. in the docstring
FINGER_ROLL_DEG = -90.0 # roll about world X at the grip. 0 is V1's fingers-down

# The point the LEFT hand's path passes through on its way to the box (the right hand
# mirrors it in x). Out beside the body, forward of it, short of the table edge at
# y = -0.29 and high enough that the hand comes in over the table rather than through
# its edge. Measured lowest-hand-corner clearance over the table: VIA z 1.20 -> -100 mm
# (the hand catches the edge and stays there), 1.30 -> +2, 1.35 -> +76.
VIA = np.array([0.30, -0.15, 1.35])

# The palm finishes turning at this fraction of the path, not at its end, so the hand is
# already fingers-forward -- and so at its shortest vertically -- as it crosses the table.
ROLL_LEAD = 0.6

# Elbows start slightly bent, as a relaxed human arm hangs. The elbow joint's range ends
# 2 deg past straight, so from the straight arm of V1/V2's home the IK pushes it the wrong
# way, pins it on that limit and loses the path by up to 190 mm. The ranges are mirrored
# (left -119..+2, right -2..+119), hence the sign per side in home_pose.
HOME_ELBOW = 0.35

REACH_T = 2.0           # home -> just outside the box
CLOSE_T = 0.5           # -> squeezing it
SAMPLES = 80            # IK points along the reach path

# Links checked against the body. The upper arm (shoulder_yaw_link) is left out on
# purpose: the torso's collision shape is a convex hull that bridges the hollow under the
# shoulder, so the upper arm "touches" it by a few mm in poses where the real meshes are
# clear -- including V2's. It is the forearm and hand that visibly went through the body.
ARM_LINKS = ("elbow_roll_link", "wrist_yaw_link", "wrist_roll_link", "wrist_pitch_link")
BODY_LINKS = ("pelvis", "waist_roll_link", "waist_yaw_link", "logo_link",
              "head_yaw_link", "head_pitch_link") + tuple(
    f"{s}_{n}" for s in ("left", "right")
    for n in ("hip_roll_link", "hip_yaw_link", "hip_pitch_link", "knee_pitch_link"))


def home_pose(model, jm):
    home = np.zeros(model.nu)
    home[jm.index("left_shoulder_roll_joint")] = 0.25
    home[jm.index("right_shoulder_roll_joint")] = -0.25
    home[jm.index("left_elbow_roll_joint")] = -HOME_ELBOW
    home[jm.index("right_elbow_roll_joint")] = HOME_ELBOW
    return home


def mat2quat(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, dtype=float).flatten())
    return q


def slerp(q0, q1, s):
    """Unit-quaternion interpolation, shortest way round."""
    d = float(np.dot(q0, q1))
    if d < 0.0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + s * (q1 - q0)
        return q / np.linalg.norm(q)
    th = np.arccos(d)
    return (np.sin((1 - s) * th) * q0 + np.sin(s * th) * q1) / np.sin(th)


def through(p0, p1, p2, s):
    """Quadratic Bezier from p0 to p2 that passes through p1 at s = 0.5."""
    c = 2.0 * p1 - 0.5 * (p0 + p2)
    return (1 - s) ** 2 * p0 + 2 * (1 - s) * s * c + s**2 * p2


def min_jerk(x):
    x = min(max(x, 0.0), 1.0)
    return x**3 * (10 - 15 * x + 6 * x**2)


def finger_angle(data, arms):
    """Degrees between each hand's finger direction (geom local -z) and world -Y."""
    out = []
    for geom in arms.geoms:
        fingers = -data.geom_xmat[geom].reshape(3, 3)[:, 2]
        out.append(np.degrees(np.arccos(np.clip(-fingers[1], -1.0, 1.0))))
    return out


def track(model, data, arms, targets, quats, iters=120, lam=0.05, step=0.5, w_rot=0.3):
    """IK for both hand geoms from whatever data.qpos holds now. Returns position errors.

    Same damped least squares as pick_box_v2.ReachArms.solve, but with an arbitrary
    attitude per hand, and meant to be called many times along a path: each call starts
    from the previous answer and only has a few mm to go.

    There is deliberately NO null-space elbow-down pull here, unlike V2. Along a dense
    path it does not just lower the elbow, it drives the upper arm round: measured, the
    shoulder yaw went 5 -> 127 deg in a tenth of the path while the wrist yaw unwound by
    the same amount, the hands barely moving. Commanded at reach speed that is a whip the
    PD cannot follow -- the arm lagged by 71 deg and swung through the hip. Plain damped
    least squares takes the minimum-norm step each time, which is what keeps it smooth.
    """
    for _ in range(iters):
        mujoco.mj_forward(model, data)
        rows, errs = [], []
        for body, geom, tgt, q_tgt in zip(arms.bodies, arms.geoms, targets, quats):
            point = data.geom_xpos[geom].copy()
            jp = np.zeros((3, model.nv))
            jr = np.zeros((3, model.nv))
            mujoco.mj_jac(model, data, jp, jr, point, body)
            rows += [jp[:, arms.dofs], jr[:, arms.dofs]]
            q_cur, q_inv, q_err, w = np.zeros(4), np.zeros(4), np.zeros(4), np.zeros(3)
            mujoco.mju_mat2Quat(q_cur, data.geom_xmat[geom])
            mujoco.mju_negQuat(q_inv, q_cur)
            mujoco.mju_mulQuat(q_err, q_tgt, q_inv)
            mujoco.mju_quat2Vel(w, q_err, 1.0)
            errs += [tgt - point, w_rot * w]
        J = np.vstack(rows)
        e = np.concatenate(errs)
        dq = J.T @ np.linalg.solve(J @ J.T + lam**2 * np.eye(J.shape[0]), e)
        data.qpos[arms.qids] = np.clip(data.qpos[arms.qids] + step * dq, arms.lo, arms.hi)
    mujoco.mj_forward(model, data)
    return [np.linalg.norm(t - data.geom_xpos[g]) for t, g in zip(targets, arms.geoms)]


def plan(model, jm, verbose=True, squeeze=SQUEEZE, roll_deg=FINGER_ROLL_DEG,
         grasp_z=GRASP_Z_FWD, via=VIA, box_y=BOX_Y, open_x=OPEN_X, grip_x=GRIP_X,
         grasp_att=None) -> dict:
    """Dense joint path for the reach, plus the squeezing pose at the end of it.

    box_y/open_x/grip_x describe the box being grasped (its centre's y, and the hand
    centres' |x| just outside it and squeezed into it); the defaults are pick_box's block.
    grasp_att, if given, is the world attitude (3x3) of both hand geoms at the grip and
    replaces the roll_deg about world X -- the P5 rubber hand's axes differ from Gen1's.
    """
    arms = Arms(model)
    data = mujoco.MjData(model)
    home = home_pose(model, jm)
    data.qpos[jm.qadr] = home
    mujoco.mj_forward(model, data)

    start = [data.geom_xpos[g].copy() for g in arms.geoms]
    att0 = [mat2quat(data.geom_xmat[g]) for g in arms.geoms]
    att1 = mat2quat(rot_x(np.deg2rad(roll_deg)) if grasp_att is None else grasp_att)
    mirror = np.array([-1.0, 1.0, 1.0])
    vias = [via, via * mirror]
    ends = [np.array([open_x, box_y, grasp_z]), np.array([-open_x, box_y, grasp_z])]

    path, worst = [], 0.0
    for s in np.linspace(0.0, 1.0, SAMPLES):
        targets = [through(start[i], vias[i], ends[i], s) for i in range(2)]
        quats = [slerp(att0[i], att1, min(1.0, s / ROLL_LEAD)) for i in range(2)]
        errs = track(model, data, arms, targets, quats)
        worst = max(worst, *errs)
        pose = home.copy()
        pose[[jm.index(n) for n in arms.names]] = data.qpos[arms.qids]
        path.append(pose)
    beside_err = errs
    fingers_beside = finger_angle(data, arms)

    # close in: the same attitude, hands moved in onto the box faces
    grip = [np.array([grip_x, box_y, grasp_z]), np.array([-grip_x, box_y, grasp_z])]
    close_err = track(model, data, arms, grip, [att1, att1], iters=600)
    fingers_close = finger_angle(data, arms)
    close = home.copy()
    close[[jm.index(n) for n in arms.names]] = data.qpos[arms.qids]
    if squeeze > 0:
        close = add_squeeze(model, jm, arms, close, squeeze)

    if verbose:
        print(f"  reach path: {SAMPLES} IK points, worst tracking error {worst*1000:.1f} mm")
        print(f"  beside box: {beside_err[0]*1000:.1f} / {beside_err[1]*1000:.1f} mm, "
              f"fingers {fingers_beside[0]:.1f} / {fingers_beside[1]:.1f} deg off forward")
        print(f"  gripping:   {close_err[0]*1000:.1f} / {close_err[1]*1000:.1f} mm, "
              f"fingers {fingers_close[0]:.1f} / {fingers_close[1]:.1f} deg off forward")
        if squeeze > 0:
            print(f"  squeeze bias: {squeeze*1000:.0f} mm of inward over-travel")
    return {"home": home, "path": np.array(path), "close": close}


def target_at(t, poses):
    """Joint target at time t: min-jerk along the reach path, then ease into the grip."""
    path = poses["path"]
    if t < REACH_T:
        x = min_jerk(t / REACH_T) * (len(path) - 1)
        i = min(int(x), len(path) - 2)
        f = x - i
        return path[i] * (1 - f) + path[i + 1] * f, "reach"
    if t < REACH_T + CLOSE_T:
        s = min_jerk((t - REACH_T) / CLOSE_T)
        return path[-1] * (1 - s) + poses["close"] * s, "close"
    return poses["close"], "hold"


def box_contacts(model, data):
    """{robot part name: summed normal force on the box}, the table left out.

    The grip number alone hides WHICH part does the squeezing -- in the first P5 run it
    was the forearms, with the hands not touching at all.
    """
    box = model.geom("box_geom").id
    table = model.geom("table_geom").id
    out = {}
    for i in range(data.ncon):
        c = data.contact[i]
        if box not in (c.geom1, c.geom2):
            continue
        g = c.geom2 if c.geom1 == box else c.geom1
        if g == table:
            continue
        f = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, f)
        name = model.geom(g).name or model.body(model.geom_bodyid[g]).name
        out[name] = out.get(name, 0.0) + f[0]
    return out


def body_clearance(builder=build_model):
    """A probe copy of the model where only forearm/hand-vs-body contacts exist.

    Returns check(qpos) -> (smallest signed distance in m, which pair). Distances are
    reported out to 20 mm; anything clearer than that comes back as +inf. Negative means
    the arm is inside the body. Both geom AND body contype/conaffinity have to be set --
    MuJoCo's broadphase filters on the per-body masks before it ever looks at geoms.
    """
    m = builder()
    d = mujoco.MjData(m)
    arm = {m.body(f"{s}_{n}").id: f"{s}_{n}"
           for s in ("left", "right") for n in ARM_LINKS}
    body = {m.body(n).id: n for n in BODY_LINKS}
    m.geom_contype[:] = 0
    m.geom_conaffinity[:] = 0
    m.body_contype[:] = 0
    m.body_conaffinity[:] = 0
    for g in range(m.ngeom):
        b = m.geom_bodyid[g]
        # the P5 model's rubber-hand slabs are boxes; every other robot geom is a mesh
        rubber = (m.geom(g).name or "").endswith("rubber_hand")
        if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH and not rubber:
            continue
        # the collision hulls are group 0; logo_link only has a visual mesh, use that
        if m.geom_group[g] != 0 and m.body(b).name != "logo_link" and not rubber:
            continue
        # a collision hull someone switched off (p5_model hides Gen1's hand this way)
        if (not rubber and m.body(b).name != "logo_link"
                and not m.geom_contype[g] and not m.geom_conaffinity[g]):
            continue
        if b in arm:
            m.geom_contype[g] = m.body_contype[b] = 1
        if b in body:
            m.geom_conaffinity[g] = m.body_conaffinity[b] = 1
        m.geom_margin[g] = 0.02

    def check(qpos):
        d.qpos[:] = qpos
        mujoco.mj_forward(m, d)
        best, who = np.inf, ""
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            a, b = (b1, b2) if b1 in arm else (b2, b1)
            if a in arm and b in body and c.dist < best:
                best, who = c.dist, f"{arm[a]} / {body[b]}"
        return best, who

    return check


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--headless", action="store_true", help="no window; report the grasp")
    ap.add_argument("--duration", type=float, default=4.0)
    ap.add_argument("--roll", type=float, default=FINGER_ROLL_DEG,
                    help="degrees to roll the hands about world X (-90 = fingers forward)")
    ap.add_argument("--grasp-z", type=float, default=GRASP_Z_FWD,
                    help="height of the hand centres at the grip")
    ap.add_argument("--squeeze", type=float, default=SQUEEZE,
                    help="metres of inward over-travel commanded into the box (grip force)")
    args = ap.parse_args()

    model = build_model()
    jm = JointMap(model)
    print("planning grasp:")
    poses = plan(model, jm, squeeze=args.squeeze, roll_deg=args.roll,
                 grasp_z=args.grasp_z)
    return simulate(model, jm, poses, args.headless, args.duration)


def simulate(model, jm, poses, headless=False, duration=4.0, builder=build_model) -> int:
    """Play the planned motion: in the viewer, or headless with a printed report."""
    data = mujoco.MjData(model)
    data.qpos[jm.qadr] = poses["home"]
    mujoco.mj_forward(model, data)

    box = model.body("box").id
    p0 = data.xpos[box].copy()
    print(f"\ntable top {TABLE_TOP:.3f} m, box centre starts at z={p0[2]:.3f} m")

    def step():
        tgt, phase = target_at(data.time, poses)
        control(model, data, jm, tgt)
        mujoco.mj_step(model, data)
        return phase

    arms_ref = Arms(model)
    if headless:
        check = body_clearance(builder)
        rows, nxt, clearance = [], 0.0, np.inf
        body_gap, body_who, body_t = np.inf, "", 0.0
        n = 0
        while data.time < duration:
            phase = step()
            clearance = min(clearance, palm_clearance(model, data, arms_ref))
            n += 1
            if n % 10 == 0:
                gap, who = check(data.qpos)
                if gap < body_gap:
                    body_gap, body_who, body_t = gap, who, data.time
            if data.time >= nxt:
                nxt += 0.5
                moved = np.linalg.norm(data.xpos[box] - p0)
                rows.append((data.time, phase, moved, grip_force(model, data)))
        print(f"\n{'t':>5} {'phase':<8} {'box moved mm':>13} {'grip N':>8}")
        for t, p, mv, f in rows:
            print(f"{t:5.1f} {p:<8} {mv*1000:13.1f} {f:8.1f}")

        fl, fr = finger_angle(data, arms_ref)
        moved = np.linalg.norm(data.xpos[box] - p0)
        print(f"\n  lowest hand corner cleared the tabletop by {clearance*1000:.0f} mm")
        if np.isfinite(body_gap):
            print(f"  closest forearm/hand to body: {body_gap*1000:.1f} mm "
                  f"({body_who}, t={body_t:.2f} s)")
        else:
            print("  forearms and hands stayed >20 mm clear of the body throughout")
        print(f"  fingers at the end: {fl:.1f} / {fr:.1f} deg off forward")
        print(f"  box ended {moved*1000:.1f} mm from where it started")
        print("  what is pressing on the box (normal force):")
        for part, force in box_contacts(model, data).items():
            print(f"    {part:<28} {force:6.1f} N")
        ok = grip_force(model, data) > 5.0 and moved < 0.01 and body_gap > 0
        print("RESULT:", "GRIPPED, BODY CLEAR" if ok else "check the numbers above")
        return 0 if ok else 1

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
