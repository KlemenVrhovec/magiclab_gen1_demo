"""Bimanual roll: pick the box up, then roll it about the robot's fore-aft axis.

    python rotate_around_x_axis.py                # interactive viewer
    python rotate_around_x_axis.py --headless     # no window, reports what the box did
    python rotate_around_x_axis.py --angle 30     # roll it part way
    python rotate_around_x_axis.py --no-waist     # arms only, for comparison
    python rotate_around_x_axis.py --descend 0.14 # set it down from higher up

Picks the box up, rolls it, sets it on the table and lets go with one hand at a time,
leaving it resting on a different face than it started on.

WHICH AXIS THIS IS. The box's left face rises and its right face drops -- a barrel roll
about the axis running from the robot's chest straight out in front of it. The robot
faces world -Y, so in this model's frame that axis is Y. The file is named for the axis
as usually pictured, matching rotate_around_y_axis.py's convention.

WHY THIS ONE IS NOT THE HOPELESS ONE. Unlike the tip about world X, the palms are NOT on
this axis: they sit at +/-X from the box centre, so rolling the box swings one palm up
and the other down, and the palms stay flat on the same two box faces the whole way.
The grasp is preserved for all 90 degrees -- nothing has to slip, and no palm has to
spin about its own normal.

WHY IT STILL FAILS ON THE ARMS ALONE. The joint for it is wrist_roll: measured in the
grasp pose its axis is 0.79 aligned with world Y, and it has 96 deg of range. But the
grasp parks BOTH wrists on a stop, left at +41.0 and right at -41.0, and because both
wrist_roll axes carry the SAME world-Y sign a common-mode roll drives both wrists the
same way in joint space. So one of them is always already jammed. Measured, rolling +Y:

    roll    pos err L/R (mm)    wrist_roll L/R
      0d       9.7      9.7      41.0  -41.0
     45d       3.4     35.8       1.5  -41.0
     90d       4.3     75.3     -35.3  -41.0

The left arm does the whole 90 at 3-4 mm. The right never moves off -41. Three things
that do NOT fix it, all measured: a Ry pre-roll (it is itself a common-mode Y rotation,
so unpinning one wrist drives the other harder into its stop); a null-space term centring
wrist_roll (it is a hard limit, not a redundancy choice); and rolling lower to give the
rising arm headroom (right wrist_roll is -41.0 at every one of four lift heights).

WHAT THIS SCRIPT ADDS, ONE. The waist, which none of the other scripts touch. waist_roll's
axis is the robot's fore-aft axis -- world Y, exactly the axis being rolled about -- so
it rotates the torso, both shoulders, both arms and the box as one rigid piece, and asks
NOTHING of either wrist. It is worth +/-20 deg on its own, and it goes to its -20 stop
immediately and stays there. waist_yaw goes into the chain too, not because it is aligned
but because it re-poses both shoulders and gives the IK somewhere else to spend the roll.
Both are ordinary actuators the PD loop already drives; they were simply never in
Arms.dofs. Waist in the chain, palms tracking the roll flat-on: 54.1 deg of the 90.

WHAT THIS SCRIPT ADDS, TWO. Separating what the palms do in POSITION from what they do
in ATTITUDE. The palms orbit the full roll either way -- left palm down, right palm up,
5 cm each -- and that vertical shear is the couple that actually turns the box. It is
pure translation and it is cheap. Asking the palms to ALSO stay flat against the box
faces is what is expensive, and it is the demand that pins right wrist_roll. Letting the
palms follow only half the roll in attitude, so they drive the box edge-on rather than
flat-on, takes it from 54.1 to 65.1 deg. See ATTITUDE_FRAC for the sweep.

THE SQUEEZE HAS TO FOLLOW THE ROLL. pick_box's add_squeeze drives the hands together
along world X. That is right only while the palms face along X. Here the pinch axis
rotates with the box, so the over-travel direction is Ry(roll) * X at every waypoint --
squeezing along a fixed X would push the palms across the box faces instead of into them.

SETTING IT DOWN, AND WHY 65 DEG IS ENOUGH. The goal is that the box ends resting on a
DIFFERENT face than it started on, and the arms do not have to carry it all the way there.
The box is 9 cm square and 20 cm tall, so it goes over its bottom edge at
atan(0.045/0.10) = 24.2 deg. Carry it past that, set it on the table, let go, and gravity
finishes the turn. Measured: 62.8 deg carried, 89.2 deg once it settles.

The hands come off ONE AT A TIME, lower first. A positive roll swings the left palm down
and the right one up, so the left is the one trapped between the box and the table and it
is the one that has to leave first; pulling the upper hand first would leave the box
propped against the lower palm with nothing above it. Each palm retracts sideways along
world X, NOT along its own normal -- at this roll the normal points down as well as out,
and retracting along it would drive the lower palm into the tabletop.

MEASURED END TO END: box rose 173 mm, was rolled 62.8 deg while carried, came to rest
89.2 deg from where it started, flat on its +X face (started on -Z), 0.0 deg off flat, at
z=1.147 on a 1.100 tabletop, with both hands clear. What the box actually does is
measured and printed at the end of every --headless run -- never assumed.
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
    JointMap,
    build_model,
    control,
    grip_force,
)
from pick_box_v2 import (
    CONTINUITY,
    ELBOW_PULL,
    GRASP_RAISE,
    PREROLL_DEG,
    SPREAD_FRAC,
    ReachArms,
    home_hand,
    rot_x,
)

# The two waist joints, in chain order (pelvis -> waist_roll -> waist_yaw -> shoulders).
# waist_roll's axis is body X, which this scene maps to world -Y: the roll axis itself.
WAIST = ["waist_roll_joint", "waist_yaw_joint"]

# How much of the box's roll the palms follow in ATTITUDE. Positions always orbit the
# full roll -- see grasp_attitude(). 1.0 keeps the palms flat on the box faces the whole
# way, which is what pins right wrist_roll; below 1.0 the palms drive the box edge-on.
# Measured, commanding 90 deg:
#
#     frac    box roll achieved
#     1.0        54.1 deg      palms flat, right wrist_roll jammed the whole way
#     0.7        53.9 deg
#     0.5        65.1 deg      best
#     0.3       -50.0 deg      DROPPED -- too little contact area to hold it
#
# 0.5 is the trade: enough palm rotation to keep a real contact patch, little enough
# that the wrists are not the binding constraint.
ATTITUDE_FRAC = 0.5

# Commanded roll. The box reaches about 65 deg of this -- the palms give up the last
# stretch to the wrists -- so command the 90 and take the 65 rather than commanding 65
# and getting 48. What the box actually does is measured and printed, never assumed.
ROLL_DEG = 90.0      # positive lifts the box's left face and drops its right
ROLL_STEPS = 6       # IK waypoints across the sweep
ROLL_SECS = 6.0      # seconds spent rolling
HOLD_SECS = 2.0      # seconds held at the end of the roll, to show the grip survived

# The point of the exercise: the box must end up resting on a DIFFERENT face than it
# started on. That does not need the full 90 deg carried. The box is 9 cm square and
# 20 cm tall, so it goes over its bottom edge at atan(0.045/0.10) = 24.2 deg -- past
# that, setting it down and letting go is enough, and gravity finishes the turn. 50 deg
# carried is a comfortable margin over the 24.2, and is the bar this script is held to.
MIN_ROLL_DEG = 50.0

# --- putting it down ---
# Metres the pair descends after the roll, to set the box on the table. The box ends
# rolled ~65 deg, so its half-height is hx*sin(65) + hz*cos(65) = 0.045*0.906 +
# 0.10*0.423 = 0.083 m, not the 0.10 it stands at. With the box centre at ~1.365 after
# the roll its lowest corner is at ~1.282, so ~0.18 puts it on the 1.100 tabletop.
# Anything short of that is a drop rather than a placement, which is fine -- it is
# resting on a corner at this angle and will topple whatever the release height.
DESCEND = 0.18
DESCEND_SECS = 2.5

# Metres each palm retracts, straight out sideways along world X. NOT along the palm
# normal: at this roll the normal points down as well as out, so retracting along it
# would drive the lower palm into the tabletop.
RELEASE = 0.10
RELEASE_SECS = 1.5   # per hand; the two hands come off one after the other
CLEAR_SECS = 2.0

LEAD_IN = [("home", 0.0), ("spread", 2.5), ("approach", 5.0), ("close", 6.5),
           ("lift", 9.0)]

# Phases that keep hold of the box, and so keep the squeeze bias. Once a hand starts
# retracting it must not also be commanded into the box.
GRIPPING = ("close", "lift", "hold", "descend")


def rot_y(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


class WaistArms(ReachArms):
    """ReachArms with the waist joints spliced into the front of the chain.

    Everything in ReachArms is indexed off self.names / self.qids / self.dofs, so
    widening those is all it takes -- mj_jac already returns columns for every DOF in
    the model and the solver just slices more of them. The task is unchanged at 12 rows
    (two hands, position and orientation); the chain goes from 14 joints to 16, so there
    are now four redundant directions instead of two to spend on the elbows and on
    staying near the previous waypoint.
    """

    def __init__(self, model, use_waist=True):
        super().__init__(model)
        if not use_waist:
            self.waist = []
            return
        self.waist = list(WAIST)
        self.names = self.waist + self.names
        self.qids = np.concatenate([
            [model.jnt_qposadr[model.joint(n).id] for n in self.waist], self.qids])
        self.dofs = np.concatenate([
            [model.jnt_dofadr[model.joint(n).id] for n in self.waist], self.dofs])
        self.lo = np.concatenate([
            [model.jnt_range[model.joint(n).id, 0] for n in self.waist], self.lo])
        self.hi = np.concatenate([
            [model.jnt_range[model.joint(n).id, 1] for n in self.waist], self.hi])


def timeline(total_deg: float, steps: int, secs: float, descend=DESCEND,
             release=RELEASE):
    """(phases, per-phase (roll, drop, left retract, right retract)).

    After the roll the pair descends to set the box down, then the hands come off ONE AT
    A TIME. The left palm is the lower one -- a positive roll swings it down and the
    right one up -- so it is the one trapped between the box and the table, and it is
    taken off first. Pulling the upper hand first would leave the box propped on the
    lower palm with nothing above it, and it would roll straight off.
    """
    phases = list(LEAD_IN)
    sched = {name: (0.0, 0.0, 0.0, 0.0) for name, _ in phases}
    t0 = phases[-1][1]
    for i in range(1, steps + 1):
        name = f"roll{i}"
        phases.append((name, t0 + secs * i / steps))
        sched[name] = (np.deg2rad(total_deg * i / steps), 0.0, 0.0, 0.0)

    r = sched[phases[-1][0]][0]
    for name, dt, entry in (
            ("hold", HOLD_SECS, (r, 0.0, 0.0, 0.0)),
            ("descend", DESCEND_SECS, (r, descend, 0.0, 0.0)),
            ("open_left", RELEASE_SECS, (r, descend, release, 0.0)),
            ("open_right", RELEASE_SECS, (r, descend, release, release)),
            ("clear", CLEAR_SECS, (r, descend - 0.06, 2.0 * release, 2.0 * release))):
        phases.append((name, phases[-1][1] + dt))
        sched[name] = entry
    return phases, sched


def grasp_attitude(roll: float, preroll_deg: float, frac: float = 1.0) -> np.ndarray:
    """Palm attitude at a given box roll: V2's pre-rolled grasp, carried round by it.

    `frac` is how much of the box's roll the palms are asked to follow in ATTITUDE. The
    palms always orbit the full roll in POSITION -- that vertical shear is the couple
    that turns the box, and it is cheap, being pure translation. Tracking the attitude
    as well is what is expensive: it is the demand that pins right wrist_roll on -41.0.
    At frac < 1 the palms sit at an angle to the box faces and drive it edge-on rather
    than flat-on, trading contact area for a roll the wrists can actually reach.
    """
    return rot_y(frac * roll) @ rot_x(np.deg2rad(preroll_deg))


def hand_targets(phase, spread_pt, roll=0.0, drop=0.0, open_l=0.0, open_r=0.0):
    """World targets for the two hand geom centres (left, right).

    During the roll the +/-X offset between the palms is carried round by Ry(roll), so
    the pair orbits the box centre: left palm down and towards the midline, right palm
    up. That keeps each palm on the box face it started on. `drop` lowers the pair to
    set the box down; `open_l`/`open_r` retract each palm sideways to let go.
    """
    if phase == "spread":
        return (np.array([spread_pt[0], spread_pt[1], spread_pt[2]]),
                np.array([-spread_pt[0], spread_pt[1], spread_pt[2]]))
    if phase == "approach":
        centre = np.array([0.0, BOX_Y, GRASP_Z + GRASP_RAISE])
        return centre + [OPEN_X, 0, 0], centre - [OPEN_X, 0, 0]
    rise = LIFT_H if phase != "close" else 0.0
    centre = np.array([0.0, BOX_Y, GRASP_Z + GRASP_RAISE + rise - drop])
    off = rot_y(roll) @ np.array([GRIP_X, 0.0, 0.0])
    out = np.array([1.0, 0.0, 0.0])
    return centre + off + open_l * out, centre - off - open_r * out


def add_squeeze(model, jm, arms, pose, amount, roll, frac=1.0):
    """Inward over-travel along the CURRENT pinch axis, with the palm attitude pinned.

    The pinch axis is the palm normal, Ry(frac*roll) * X, not X -- see the module
    docstring. Rotation rows target zero so the squeeze step cannot disturb the roll it
    is applied on top of.
    """
    scratch = mujoco.MjData(model)
    scratch.qpos[jm.qadr] = pose
    mujoco.mj_forward(model, scratch)

    axis = rot_y(frac * roll) @ np.array([1.0, 0.0, 0.0])
    rows, errs = [], []
    for body, geom, inward in zip(arms.bodies, arms.geoms, (-1.0, +1.0)):
        jp = np.zeros((3, model.nv))
        jr = np.zeros((3, model.nv))
        mujoco.mj_jac(model, scratch, jp, jr, scratch.geom_xpos[geom].copy(), body)
        rows.append(jp[:, arms.dofs])
        errs.append(inward * amount * axis)
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


def plan(model, jm, phases, sched, squeeze=0.0, preroll_deg=PREROLL_DEG,
         use_waist=True, frac=ATTITUDE_FRAC, verbose=True) -> dict:
    """Solve IK once per waypoint, warm-started along the sweep."""
    scratch = mujoco.MjData(model)
    arms = WaistArms(model, use_waist)

    home = np.zeros(model.nu)
    home[jm.index("left_shoulder_roll_joint")] = 0.25
    home[jm.index("right_shoulder_roll_joint")] = -0.25

    full = np.array([SPREAD_X, SPREAD_Y, SPREAD_Z])
    h0 = home_hand(model, jm, home)
    spread_pt = h0 + SPREAD_FRAC * (full - h0)

    poses = {"home": home}
    seed = np.zeros(len(arms.names))
    arm_seed = np.deg2rad([-20, 70, -20, -40, 0, 0, 0, -20, -70, 20, 40, 0, 0, 0])
    seed[-14:] = arm_seed
    if verbose:
        print(f"  {'waypoint':<10} {'roll':>6} {'pos err L/R (mm)':>18} "
              f"{'waist roll/yaw':>16} {'wrist_roll L/R':>16}")
    wr = [arms.names.index(f"{s}_wrist_roll_joint") for s in ("left", "right")]
    for name, _ in phases[1:]:
        if name == "hold":
            continue
        r, drop, ol, orr = sched[name]
        tl, tr = hand_targets(name, spread_pt, r, drop, ol, orr)
        R = grasp_attitude(r, preroll_deg if name not in
                           ("spread", "approach") else 0.0, frac)

        if name == "close":
            # the branch is decided here, at the first waypoint that actually holds the
            # box; every later waypoint is warm-started and continuity keeps it there
            q, el, er, label = solve_branch(arms, model, jm, home, tl, tr, R, seed,
                                            verbose=verbose)
            if verbose:
                print(f"      -> took the '{label}' branch")
        else:
            mujoco.mj_resetData(model, scratch)
            scratch.qpos[jm.qadr] = home
            scratch.qpos[arms.qids] = np.clip(seed, arms.lo, arms.hi)
            q, el, er = solve_to(arms, scratch, tl, tr, R, q_ref=seed)

        full_pose = home.copy()
        for n, v in zip(arms.names, q):
            full_pose[jm.index(n)] = v
        poses[name] = full_pose
        seed = q                                   # warm-start the next waypoint
        if verbose:
            wz = ([np.degrees(q[arms.names.index(n)]) for n in arms.waist]
                  if arms.waist else [0.0, 0.0])
            print(f"  {name:<10} {np.degrees(r):5.0f}d "
                  f"{el*1000:8.1f} {er*1000:8.1f} "
                  f"{wz[0]:8.1f} {wz[1]:7.1f} "
                  f"{np.degrees(q[wr[0]]):8.1f} {np.degrees(q[wr[1]]):7.1f}")

    poses["hold"] = poses[f"roll{sum(1 for n in poses if n.startswith('roll'))}"]

    if squeeze > 0:
        for name in poses:
            if name in GRIPPING or name.startswith("roll"):
                poses[name] = add_squeeze(model, jm, arms, poses[name], squeeze,
                                          sched[name][0], frac)
        if verbose:
            print(f"  squeeze bias: {squeeze*1000:.0f} mm of inward over-travel, "
                  f"dropped from '{GRIPPING[-1]}' onwards")
    return poses


def solve_to(arms, data, tl, tr, R, q_ref=None):
    """ReachArms.solve, but taking a full attitude matrix instead of an Rx angle."""
    import pick_box_v2 as _v2
    saved = _v2.rot_x
    _v2.rot_x = lambda _a: R
    try:
        return arms.solve(data, tl, tr, 1.0, elbow_pull=ELBOW_PULL, q_ref=q_ref,
                          cont=CONTINUITY)
    finally:
        _v2.rot_x = saved


# --- picking the branch the roll can actually be spent from ---
# The grasp is mirror-symmetric, so the IK naturally lands on the mirror-symmetric
# solution: left wrist_roll +41.0, right -41.0, each on the stop at ITS end of a mirrored
# range. A common-mode roll then needs both wrists to move the same way in joint space,
# and the right one has nowhere to go -- measured, it never leaves -41.0 for the whole
# sweep, at any lift height, with or without a null-space term pulling on it.
#
# wrist_yaw, wrist_roll and wrist_pitch are a 3-DOF wrist, so most palm attitudes have a
# second solution roughly 180 deg away in wrist_yaw with wrist_roll flipped in sign. That
# is the branch worth having here: it puts right wrist_roll near +41 instead of -41, with
# the same palm pose, and hands the roll 80-odd degrees of travel it otherwise lacks.
# wrist_yaw has +/-157 deg to play with, so the flip fits.
#
# Rather than assume the flip exists, seed the grasp both ways and keep whichever scores
# better. Score is grasp accuracy first, then how much room the wrists have left in the
# direction the roll needs -- a solution 2 mm worse that unlocks 80 deg is a good trade.
#
# MEASURED RESULT: it never fires. All three flipped seeds converge back to zero wrist
# headroom, at 5.0, 40.8 and 60.7 mm of grasp error against the nominal branch's 4.7, so
# 'nominal' always wins. The flipped wrist solution does not exist at this hand pose --
# wrist_pitch only spans +/-18 deg, far too little to make up the attitude a flipped
# wrist_roll gives away. This is kept because the negative result is worth having: it
# rules out the branch explanation, which leaves the joint limit itself, which is what
# ATTITUDE_FRAC works around. Costs four IK solves at one waypoint, once, at plan time.
WRIST_FLIP = np.deg2rad(180.0)
HEADROOM_WEIGHT = 0.25       # mm of grasp error each degree of wrist headroom is worth


def branch_seeds(arms, seed):
    """The nominal seed, plus the same seed with each wrist flipped."""
    out = [("nominal", seed)]
    for side in ("right", "left", "both"):
        flipped = seed.copy()
        sides = ("left", "right") if side == "both" else (side,)
        for s in sides:
            iy = arms.names.index(f"{s}_wrist_yaw_joint")
            ir = arms.names.index(f"{s}_wrist_roll_joint")
            flipped[iy] = np.clip(seed[iy] + WRIST_FLIP, arms.lo[iy], arms.hi[iy])
            flipped[ir] = np.clip(-seed[ir], arms.lo[ir], arms.hi[ir])
        out.append((f"flip {side}", flipped))
    return out


def roll_headroom(arms, q):
    """Degrees of wrist_roll travel left in the direction a positive roll needs.

    Both wrist_roll axes carry the same world-Y sign, so a positive box roll drives both
    wrists towards their lower limits. The binding one is whichever has less left.
    """
    room = []
    for s in ("left", "right"):
        i = arms.names.index(f"{s}_wrist_roll_joint")
        room.append(np.degrees(q[i] - arms.lo[i]))
    return min(room)


def solve_branch(arms, model, jm, home, tl, tr, R, seed, verbose=False):
    """Solve the grasp from several wrist branches; return the best (q, errs, label)."""
    scratch = mujoco.MjData(model)
    best = None
    for label, s in branch_seeds(arms, seed):
        mujoco.mj_resetData(model, scratch)
        scratch.qpos[jm.qadr] = home
        scratch.qpos[arms.qids] = np.clip(s, arms.lo, arms.hi)
        q, el, er = solve_to(arms, scratch, tl, tr, R, q_ref=s)
        score = max(el, er) * 1000 - HEADROOM_WEIGHT * roll_headroom(arms, q)
        if verbose:
            print(f"      branch {label:<11} err {el*1000:5.1f}/{er*1000:5.1f} mm  "
                  f"wrist_roll headroom {roll_headroom(arms, q):5.1f} deg  "
                  f"score {score:6.1f}")
        if best is None or score < best[0]:
            best = (score, q.copy(), el, er, label)
    return best[1], best[2], best[3], best[4]


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
    """How many contact points hold the box, ignoring the table."""
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


def box_roll(model, data, q0) -> tuple:
    """(total rotation of the box since q0, and the roll component) in deg.

    Positive roll = the box's left face has risen, matching the commanded sense.
    """
    q_inv, q_rel, w = np.zeros(4), np.zeros(4), np.zeros(3)
    mujoco.mju_negQuat(q_inv, q0)
    mujoco.mju_mulQuat(q_rel, data.xquat[model.body("box").id], q_inv)
    mujoco.mju_quat2Vel(w, q_rel, 1.0)
    return np.degrees(np.linalg.norm(w)), np.degrees(w[1])


def down_face(model, data) -> tuple:
    """Which of the box's own faces is lying against the table, and how squarely.

    Column i of the body's rotation matrix is where its local axis i points in world, so
    the local axis with the largest downward component names the face it is resting on.
    Upright at the start that is -Z; anything else at the end means it turned over onto
    a new side, which is the whole point of the manoeuvre.
    """
    R = data.xmat[model.body("box").id].reshape(3, 3)
    dots = R.T @ np.array([0.0, 0.0, -1.0])
    i = int(np.argmax(np.abs(dots)))
    return f"{'+' if dots[i] > 0 else '-'}{'XYZ'[i]}", abs(dots[i])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--headless", action="store_true", help="no window; report the outcome")
    ap.add_argument("--angle", type=float, default=ROLL_DEG,
                    help="degrees to roll the box about the fore-aft axis")
    ap.add_argument("--steps", type=int, default=ROLL_STEPS, help="IK waypoints in the roll")
    ap.add_argument("--roll-secs", type=float, default=ROLL_SECS)
    ap.add_argument("--squeeze", type=float, default=SQUEEZE,
                    help="metres of inward over-travel commanded into the box (grip force)")
    ap.add_argument("--preroll", type=float, default=PREROLL_DEG,
                    help="degrees to pre-roll the palms about X at the grasp")
    ap.add_argument("--attitude-frac", type=float, default=ATTITUDE_FRAC,
                    help="fraction of the roll the palms follow in attitude; the palms "
                         "orbit the full roll in position regardless")
    ap.add_argument("--no-waist", action="store_true",
                    help="leave the waist out of the IK chain, for comparison")
    ap.add_argument("--descend", type=float, default=DESCEND,
                    help="metres the pair lowers after the roll, to set the box down")
    ap.add_argument("--release", type=float, default=RELEASE,
                    help="metres each palm retracts sideways to let go")
    ap.add_argument("--duration", type=float, default=None, help="default: the full timeline")
    args = ap.parse_args()

    phases, sched = timeline(args.angle, args.steps, args.roll_secs,
                             args.descend, args.release)
    at = dict(phases)
    duration = args.duration if args.duration else phases[-1][1] + 2.0

    model = build_model()
    jm = JointMap(model)
    print(f"planning grasp + {args.angle:.0f} deg roll about the fore-aft axis, "
          f"then set down and release "
          f"({'arms only' if args.no_waist else 'waist in the chain'}):")
    poses = plan(model, jm, phases, sched, squeeze=args.squeeze,
                 preroll_deg=args.preroll, use_waist=not args.no_waist,
                 frac=args.attitude_frac)

    data = mujoco.MjData(model)
    data.qpos[jm.qadr] = poses["home"]
    mujoco.mj_forward(model, data)

    box = model.body("box").id
    z0 = data.xpos[box][2]
    q0 = data.xquat[box].copy()
    face0, _ = down_face(model, data)
    print(f"\ntable top {TABLE_TOP:.3f} m, box centre starts at z={z0:.3f} m, "
          f"resting on its {face0} face")

    def step():
        tgt, phase = target_at(data.time, poses, phases)
        control(model, data, jm, tgt)
        mujoco.mj_step(model, data)
        return phase

    if args.headless:
        rows, nxt = [], 0.0
        peak = z0
        carried = None          # roll achieved while still held, before it is set down
        while data.time < duration:
            phase = step()
            peak = max(peak, data.xpos[box][2])
            if carried is None and data.time >= at["descend"]:
                carried = box_roll(model, data, q0)[1]
            if data.time >= nxt:
                nxt += 1.0
                _, rolled = box_roll(model, data, q0)
                rows.append((data.time, phase, data.xpos[box][2], rolled,
                             grip_force(model, data), box_contacts(model, data)))
        print(f"\n{'t':>5} {'phase':<18} {'box z':>7} {'box roll':>9} {'grip N':>8}"
              f" {'hand contacts':>14}")
        for t, p, z, r, f, n in rows:
            print(f"{t:5.1f} {p:<18} {z:7.3f} {r:8.1f}d {f:8.1f} {n:14d}")

        total, rolled = box_roll(model, data, q0)
        held_by = box_contacts(model, data)
        z_end = data.xpos[box][2]
        print(f"\nbox rose {(peak - z0)*1000:.0f} mm above its start height")
        print(f"box was rolled {carried:.1f} deg while carried (commanded {args.angle:.0f}), "
              f"and came to rest {rolled:.1f} deg from where it started "
              f"({total:.1f} deg total rotation)")
        print(f"box settled at z={z_end:.3f} m, table top {TABLE_TOP:.3f} "
              f"-- {'ON THE TABLE' if z_end > TABLE_TOP else 'OFF THE TABLE'}")
        print(f"hand/box contacts at the end: {held_by} "
              f"({'released' if held_by == 0 else 'STILL TOUCHING'})")

        face1, square = down_face(model, data)
        print(f"box now rests on its {face1} face (started on {face0}), "
              f"{np.degrees(np.arccos(min(square, 1.0))):.1f} deg off flat")

        turned = abs(carried) >= MIN_ROLL_DEG
        placed = z_end > TABLE_TOP and held_by == 0
        new_face = face1 != face0
        for label, ok in (("carried past 50 deg", turned),
                          ("left on the table, hands clear", placed),
                          ("landed on a new face", new_face)):
            print(f"  [{'x' if ok else ' '}] {label}")
        print("RESULT:", "TURNED OVER AND PLACED" if (turned and placed and new_face)
              else "FAILED")
        return 0 if (turned and placed and new_face) else 1

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
