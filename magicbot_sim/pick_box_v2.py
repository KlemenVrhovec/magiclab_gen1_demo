"""
Bimanual pick, version 2: half a spread, palms pre-rolled, elbows pulled down.

    python pick_box_v2.py               # interactive viewer
    python pick_box_v2.py --headless    # no window, reports the lift and the end pose

Same robot, table, box and squeeze-to-grip trick as pick_box.py, which this file imports
rather than repeats. Three things about the ARM differ, all of them measured.

1. HALF A SPREAD. V1 swings the arms right out to the sides and up (hands to
   x = +/-0.42, z = 1.50) before coming forward onto the box. Going only halfway there
   lands the IK on a far better branch. Solving the same grasp from each seed:

       lead-in        pos err   palm err
       V1 full        13-20 mm  10-16 deg
       half spread     0.0 mm    0.0 deg

   Half a spread still clears the tabletop, which is the only reason the waypoint exists
   -- interpolating straight from arms-down to the grasp drags both hands through it.

2. PALMS PRE-ROLLED. The grasp attitude is rolled PREROLL_DEG about X instead of being
   left square. That is what makes the solution exact: at 0 deg the same IK misses by
   5.6 mm with 12.7 deg of palm error, at -20 deg it is 0.0 and 0.0. It also leaves the
   palms turned so that the roll they have left runs the other way, which is the point
   of doing it -- see rotate_around_y_axis.py for what that roll is worth.

3. ELBOWS PULLED DOWN. Each arm has 7 joints for a 6-DOF task, so once the hand pose is
   satisfied there is one direction of joint motion left over that does not disturb it.
   The IK spends it lowering the elbows, by projecting an "elbow down" gradient through
   the null space of the task Jacobian. Without it the IK parks the elbows at z = 1.756,
   above the shoulder at 1.596 -- a chicken-wing pose. What it gets to is printed at the
   end of every run.

WHAT V2 DOES NOT FIX. It does not restore the arm's ability to roll the box about world
X. Reaching a 9 cm box at the body midline forces shoulder_pitch well away from zero,
and shoulder_pitch sits BEFORE shoulder_roll in the chain, so it swings the shoulder-roll
axis off X and onto the vertical no matter which branch the IK takes. That is a property
of where the hands have to be, not of how they get there: a grip wide enough to keep each
hand in front of its own shoulder (box ~35 cm across) fixes it and nothing else does.
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
    add_squeeze,
    build_model,
    control,
    grip_force,
)

# Metres to raise the whole grasp above pick_box's GRASP_Z. The palm slab is 27 cm long
# and hangs ~13.5 cm below the point the IK aims at, so V1's 1.27 leaves its bottom edge
# at 1.135 with the tabletop at 1.100 -- 35 mm, and the palms catch on the table on the
# way in. The box top is at 1.30, so there is room to go up. Measured lowest-palm-corner
# clearance on the way in: -9 mm at +30 mm of raise, -3 at +45, +3 at +60 -- so 60 it
# is, and the palm still overlaps the box side face by ~105 mm, plenty for the grip.
GRASP_RAISE = 0.060

SPREAD_FRAC = 0.5      # how far along V1's spread to go. 1.0 is V1, 0.0 is not moving
PREROLL_DEG = -20.0    # palm roll about X at the grasp; 0 is V1's square palms
ELBOW_PULL = 0.6       # null-space gain pulling the elbows down. 0 disables it
CONTINUITY = 1.5       # null-space gain holding each waypoint near the previous one

PHASES = [("home", 0.0), ("spread", 2.5), ("approach", 5.0), ("close", 6.5),
          ("lift", 9.0), ("hold", 11.5)]
IK_PHASES = ("spread", "approach", "close", "lift")
ROLL_CHAIN = ("shoulder_roll", "elbow_roll", "wrist_roll")


def rot_x(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


class ReachArms(Arms):
    """pick_box.Arms plus a rolled grasp attitude and a null-space elbow preference.

    Arms.solve() targets the identity attitude and takes whatever elbow pose the damped
    least squares happens to land on. Here the attitude target is Rx(preroll), and the
    redundant DOF each arm has left over is spent lowering the elbow: the elbow-height
    gradient is projected through (I - J+ J), which is the subspace of joint motions
    that leave the commanded hand pose untouched. So the elbows come down for free,
    without trading away any of the grasp accuracy.
    """

    def __init__(self, model):
        super().__init__(model)
        self.elbows = [model.joint("left_elbow_roll_joint").id,
                       model.joint("right_elbow_roll_joint").id]
        self.elbow_bodies = [model.body("left_elbow_roll_link").id
                             if self._has("left_elbow_roll_link") else
                             model.jnt_bodyid[self.elbows[0]],
                             model.body("right_elbow_roll_link").id
                             if self._has("right_elbow_roll_link") else
                             model.jnt_bodyid[self.elbows[1]]]

    def _has(self, name):
        try:
            self.model.body(name)
            return True
        except KeyError:
            return False

    def elbow_gradient(self, data):
        """d(sum of elbow heights)/dq, restricted to the arm joints."""
        g = np.zeros(len(self.dofs))
        for jid, body in zip(self.elbows, self.elbow_bodies):
            jp = np.zeros((3, self.model.nv))
            mujoco.mj_jac(self.model, data, jp, None, data.xanchor[jid].copy(), body)
            g += jp[2, self.dofs]                 # world-z row
        return g

    def solve(self, data, tl, tr, preroll=0.0, elbow_pull=0.0, q_ref=None, cont=0.0,
              iters=3000, lam=0.10, step=0.5, w_rot=0.30):
        """Returns (angles, pos err L, pos err R).

        q_ref/cont add a second null-space objective: stay near the previous waypoint's
        joint vector. Without it each waypoint's IK picks its own place on the redundant
        circle -- the hands still follow a smooth path, but the elbows and wrists tour
        around between waypoints (measured: 51 deg of wrist_yaw reversal between two
        waypoints that share the same hand position). Continuity is what makes the
        commanded joint interpolation look like one motion instead of several.
        """
        q_tgt = np.zeros(4)
        mujoco.mju_mat2Quat(q_tgt, rot_x(preroll).flatten())

        for it in range(iters):
            # Secondary objectives are held at full strength for most of the solve and
            # ramped off over the last fifth, so the tail converges on the task alone.
            # Built with a DAMPED pseudo-inverse the projector does not annihilate task
            # motion exactly and they leak badly -- the grasp went from 12.6 mm of error
            # to 34.1 mm and the hands stopped touching the box. The projector below
            # therefore uses the TRUE pseudo-inverse, which is an exact projector, while
            # the task step keeps the damped one for stability near singularities.
            fade = min(1.0, max(0.0, (1.0 - it / iters) / 0.2))
            mujoco.mj_forward(self.model, data)
            rows, errs = [], []
            for body, geom, tgt in zip(self.bodies, self.geoms, (tl, tr)):
                point = data.geom_xpos[geom].copy()
                jp = np.zeros((3, self.model.nv))
                jr = np.zeros((3, self.model.nv))
                mujoco.mj_jac(self.model, data, jp, jr, point, body)
                rows.append(jp[:, self.dofs])
                errs.append(tgt - point)

                q_cur, q_inv, q_err, w = (np.zeros(4), np.zeros(4), np.zeros(4),
                                          np.zeros(3))
                mujoco.mju_mat2Quat(q_cur, data.geom_xmat[geom])
                mujoco.mju_negQuat(q_inv, q_cur)
                mujoco.mju_mulQuat(q_err, q_tgt, q_inv)
                mujoco.mju_quat2Vel(w, q_err, 1.0)
                rows.append(jr[:, self.dofs])
                errs.append(w_rot * w)

            J = np.vstack(rows)
            e = np.concatenate(errs)
            if np.linalg.norm(e) < 1e-4 and not (fade and (elbow_pull or cont)):
                break
            Jt_inv = J.T @ np.linalg.inv(J @ J.T + lam**2 * np.eye(J.shape[0]))
            dq = Jt_inv @ e
            if fade and (elbow_pull or cont):
                # both of these move through the null space of the task, so neither can
                # disturb the commanded hand pose -- they only choose WHICH of the many
                # arm configurations that achieve it gets used
                z = np.zeros(len(self.dofs))
                if elbow_pull:
                    z -= elbow_pull * self.elbow_gradient(data)
                if cont and q_ref is not None:
                    z += cont * (q_ref - data.qpos[self.qids])
                null = np.eye(J.shape[1]) - np.linalg.pinv(J) @ J
                dq = dq + fade * (null @ z)
            data.qpos[self.qids] = np.clip(data.qpos[self.qids] + step * dq,
                                           self.lo, self.hi)

        mujoco.mj_forward(self.model, data)
        return (data.qpos[self.qids].copy(),
                np.linalg.norm(tl - data.geom_xpos[self.geoms[0]]),
                np.linalg.norm(tr - data.geom_xpos[self.geoms[1]]))


def home_hand(model, jm, home):
    """Where the left hand geom sits with the arms down -- one end of the spread."""
    data = mujoco.MjData(model)
    data.qpos[jm.qadr] = home
    mujoco.mj_forward(model, data)
    return data.geom_xpos[Arms(model).geoms[0]].copy()


def hand_targets(phase, spread_pt, raise_by=GRASP_RAISE):
    """World targets for the two hand geom centres (left, right)."""
    if phase == "spread":
        centre = np.array([0.0, spread_pt[1], spread_pt[2]])
        gap = spread_pt[0]
    else:
        rise = LIFT_H if phase in ("lift", "hold") else 0.0
        centre = np.array([0.0, BOX_Y, GRASP_Z + raise_by + rise])
        gap = OPEN_X if phase == "approach" else GRIP_X
    return centre + [gap, 0, 0], centre - [gap, 0, 0]


def palm_clearance(model, data, arms):
    """Lowest corner of either palm above the tabletop, counted only while the palm is
    actually over the table. Returns inf when neither is.

    The palms are box geoms, so the lowest point is the centre minus the slab's extent
    projected onto world z -- which changes as the palms roll, and is the thing that
    catches on the table edge. Ignoring the footprint check makes this meaningless: the
    arms hang far below the tabletop at home, nowhere near the table.
    """
    tg = model.geom("table_geom").id
    tx, ty = model.geom_size[tg][0], model.geom_size[tg][1]
    cx, cy = data.geom_xpos[tg][0], data.geom_xpos[tg][1]
    worst = np.inf
    for g in arms.geoms:
        p = data.geom_xpos[g]
        if abs(p[0] - cx) > tx + 0.15 or abs(p[1] - cy) > ty + 0.15:
            continue                      # not over the table (+ a slab-length margin)
        R = data.geom_xmat[g].reshape(3, 3)
        drop = float(np.abs(R[2, :]) @ model.geom_size[g])
        worst = min(worst, p[2] - drop)
    return worst - TABLE_TOP if np.isfinite(worst) else np.inf


def plan(model, jm, verbose=True, squeeze=0.0, frac=SPREAD_FRAC,
         preroll_deg=PREROLL_DEG, elbow_pull=ELBOW_PULL, cont=CONTINUITY,
         raise_by=GRASP_RAISE) -> dict:
    """Solve IK once per waypoint; return full 30-vector joint targets."""
    scratch = mujoco.MjData(model)
    arms = ReachArms(model)

    home = np.zeros(model.nu)
    home[jm.index("left_shoulder_roll_joint")] = 0.25
    home[jm.index("right_shoulder_roll_joint")] = -0.25

    # the spread waypoint, frac of the way from where the hands hang to V1's spread
    full = np.array([SPREAD_X, SPREAD_Y, SPREAD_Z])
    spread_pt = home_hand(model, jm, home) + frac * (full - home_hand(model, jm, home))
    if verbose:
        print(f"  spread {frac:.0%} of V1: hands to "
              f"x={spread_pt[0]:.2f} y={spread_pt[1]:.2f} z={spread_pt[2]:.2f}")

    poses = {"home": home}
    seed = np.deg2rad([-20, 70, -20, -40, 0, 0, 0, -20, -70, 20, 40, 0, 0, 0])
    for name in IK_PHASES:
        tl, tr = hand_targets(name, spread_pt, raise_by)
        # the palms only take up the pre-roll once they are actually on the box
        roll = np.deg2rad(preroll_deg) if name in ("close", "lift") else 0.0
        mujoco.mj_resetData(model, scratch)
        scratch.qpos[jm.qadr] = home
        scratch.qpos[arms.qids] = np.clip(seed, arms.lo, arms.hi)
        q, el, er = arms.solve(scratch, tl, tr, roll, elbow_pull,
                               q_ref=seed, cont=cont)
        full_pose = home.copy()
        for n, v in zip(arms.names, q):
            full_pose[jm.index(n)] = v
        poses[name] = full_pose
        seed = q                                   # warm-start the next waypoint
        if verbose:
            print(f"  IK {name:<8} left {el*1000:6.1f} mm   right {er*1000:6.1f} mm"
                  f"   roll {np.degrees(roll):5.0f}d")

    if squeeze > 0:
        for name in ("close", "lift"):
            poses[name] = add_squeeze(model, jm, arms, poses[name], squeeze)
        if verbose:
            print(f"  squeeze bias: {squeeze*1000:.0f} mm of inward over-travel")

    poses["hold"] = poses["lift"]
    return poses


def pose_report(model, jm, pose):
    """Where the elbows ended up, and what roll the grasp has left in it."""
    data = mujoco.MjData(model)
    data.qpos[jm.qadr] = pose
    mujoco.mj_forward(model, data)

    shoulder_z = data.xanchor[model.joint("left_shoulder_roll_joint").id][2]
    elbow_z = [data.xanchor[model.joint(f"{s}_elbow_roll_joint").id][2]
               for s in ("left", "right")]
    print(f"\n  shoulder at z={shoulder_z:.3f}   elbows at "
          f"z={elbow_z[0]:.3f} / {elbow_z[1]:.3f}  "
          f"({'BELOW' if elbow_z[0] < shoulder_z else 'above'} the shoulder)")

    usable = 0.0
    print(f"\n  {'joint':<28} {'angle':>7} {'room':>7} {'|axis.X|':>9}")
    for side in ("left", "right"):
        for jn in ROLL_CHAIN:
            jid = model.joint(f"{side}_{jn}_joint").id
            v = data.qpos[model.jnt_qposadr[jid]]
            lo, hi = model.jnt_range[jid]
            align = abs(data.xaxis[jid][0])
            usable += align * min(v - lo, hi - v)
            print(f"  {side + '_' + jn:<28} {np.degrees(v):7.1f} "
                  f"{np.degrees(min(v - lo, hi - v)):7.1f} {align:9.2f}")
    for side in ("left", "right"):
        v = data.qpos[model.jnt_qposadr[model.joint(f"{side}_shoulder_pitch_joint").id]]
        print(f"  {side}_shoulder_pitch{'':<11} {np.degrees(v):7.1f}   "
              f"(0 would keep shoulder_roll on world X)")
    print(f"\n  usable world-X roll authority: {np.degrees(usable) / 6.0:.1f} deg "
          f"(V1 scores 8.7)")


def target_at(t, poses):
    """Cosine-eased interpolation between waypoint joint vectors."""
    if t <= PHASES[0][1]:
        return poses[PHASES[0][0]], PHASES[0][0]
    for (n0, t0), (n1, t1) in zip(PHASES, PHASES[1:]):
        if t < t1:
            s = (t - t0) / (t1 - t0)
            s = 0.5 - 0.5 * np.cos(np.pi * s)
            return poses[n0] * (1 - s) + poses[n1] * s, f"{n0}->{n1}"
    return poses[PHASES[-1][0]], PHASES[-1][0]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--headless", action="store_true", help="no window; report the outcome")
    ap.add_argument("--duration", type=float, default=14.0)
    ap.add_argument("--spread", type=float, default=SPREAD_FRAC,
                    help="fraction of V1's spread to use for the lead-in")
    ap.add_argument("--preroll", type=float, default=PREROLL_DEG,
                    help="degrees to pre-roll the palms about X at the grasp")
    ap.add_argument("--raise-grasp", type=float, default=GRASP_RAISE,
                    help="metres to raise the grasp, to keep the palms off the table")
    ap.add_argument("--continuity", type=float, default=CONTINUITY,
                    help="null-space gain holding waypoints near each other")
    ap.add_argument("--elbow-pull", type=float, default=ELBOW_PULL,
                    help="null-space gain pulling the elbows down (0 disables)")
    ap.add_argument("--squeeze", type=float, default=SQUEEZE,
                    help="metres of inward over-travel commanded into the box (grip force)")
    args = ap.parse_args()

    model = build_model()
    jm = JointMap(model)
    print("planning grasp:")
    poses = plan(model, jm, squeeze=args.squeeze, frac=args.spread,
                 preroll_deg=args.preroll, elbow_pull=args.elbow_pull,
                 cont=args.continuity, raise_by=args.raise_grasp)

    data = mujoco.MjData(model)
    data.qpos[jm.qadr] = poses["home"]
    mujoco.mj_forward(model, data)

    box = model.body("box").id
    z0 = data.xpos[box][2]
    print(f"\ntable top {TABLE_TOP:.3f} m, box centre starts at z={z0:.3f} m")

    def step():
        tgt, phase = target_at(data.time, poses)
        control(model, data, jm, tgt)
        mujoco.mj_step(model, data)
        return phase

    arms_ref = Arms(model)
    if args.headless:
        peak, rows, nxt = z0, [], 0.0
        clearance = np.inf
        while data.time < args.duration:
            phase = step()
            peak = max(peak, data.xpos[box][2])
            if data.time < PHASES[4][1]:          # while reaching in, before the lift
                clearance = min(clearance, palm_clearance(model, data, arms_ref))
            if data.time >= nxt:
                nxt += 1.0
                rows.append((data.time, phase, data.xpos[box][2], grip_force(model, data)))
        print(f"\n{'t':>5} {'phase':<16} {'box z':>7} {'grip N':>8}")
        for t, p, z, f in rows:
            print(f"{t:5.1f} {p:<16} {z:7.3f} {f:8.1f}")

        print(f"\n  lowest palm corner cleared the tabletop by "
              f"{clearance*1000:.0f} mm on the way in")
        pose_report(model, jm, poses["lift"])

        held = data.xpos[box][2]
        print(f"\nbox peaked {(peak - z0)*1000:.0f} mm above its start height, "
              f"settled {(held - z0)*1000:.0f} mm above it")
        ok = held - z0 > 0.05
        print("RESULT:", "LIFTED" if ok else "not lifted")
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
