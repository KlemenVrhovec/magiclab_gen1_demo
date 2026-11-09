"""Turn a block onto a new face WITHOUT picking it up: shove it over with one palm.

    python topple_box.py                  # interactive viewer
    python topple_box.py --headless       # no window, reports which face ended up down
    python topple_box.py --sweep-shove    # how wide the window of working push distances is
    python topple_box.py --sweep-shape    # the same push against five different blocks
    python topple_box.py --sweep-table    # the same push at five table heights

WHY THIS EXISTS. rotate_around_x_axis.py and its v2/v3 do the job the hard way: pinch the
box between two palms, lift it, roll it in mid-air, and release it tilted just above the
table. Three separate problems are stacked in that sentence, and the repo has already
measured all three:

  * the robot has no fingers, so the grasp is a bimanual pinch on a 611-vertex convex
    hull, which yields as little as ONE contact point per hand (grasp_contact_findings.md
    section 3). rotate_around_x_axis_v3.py needs a fabricated palm pad to fix that.
  * the box's carried attitude is therefore not commanded, it is emergent -- the pads
    alone move it by ~34 deg (v3 docstring).
  * and the release is bistable. A box let go tilted, in the air, above a table can fall
    onto the new face or rock back onto the old one, and v3's own sweep shows TEN MICRONS
    of descent flipping a clean 90 deg turn into complete failure:

        descend 0.11959010235910551 -> 86.1 deg, +X face   OK
        descend 0.11960000000000000 ->  0.0 deg, -Z face   FELL BACK

THE OBSERVATION. Every one of those problems comes from the box being airborne. None of
them is inherent to the task. A block standing on a base of half-width a and half-height h
topples about its bottom edge as soon as it is tilted past atan(a/h) -- and if it is still
standing ON the table when it passes that angle, resting on that bottom edge, there is
nothing to be bistable about. It cannot rock back, because the edge it would rock back
over is the one it is already turning about. Gravity finishes the move for free.

So: never grasp, never lift, never release. Put one palm against the upper part of the
block and push sideways until it goes past its tipping angle. That is the whole method.
The three hard problems above are not solved here, they are deleted.

The block does have to tip rather than slide, and the textbook condition for that is
generous here: pushing at height z on a base of half-width a with friction mu, it tips
first when z > a/mu, which for this scene's friction of 2.0 and a = 60 mm means anything
above 30 mm. Do not read too much into that margin. It is a statement about a quasi-static
push by a point, and --sweep-height measures what actually happens: below about 90% of the
way up the face the block does not reliably go over at all, because the palm stays in
contact through the fall and drives a block that gravity is already turning. PUSH_FRAC
carries that measurement and the reasoning behind it.

What the supported turn does buy is insensitivity to the push DISTANCE. At the right
height, every push from 120 mm to 220 mm produces the same 90 deg result, because the
block separates from the palm within a few degrees and everything after that is gravity.
That is the number to compare against v3's ten microns -- a hundred-millimetre window
against a ten-micron one -- and it comes from the block never being airborne, not from the
friction margin above.

WHAT WAS CHANGED IN THE SCENE, AND WHY (the two things the prompt asked about):

  TABLE HEIGHT, 1.10 -> TABLE_TOP below. pick_box.py put the tabletop at chest height
  because a bimanual pinch has to reach ACROSS the body midline, and the hand geom can
  only do that in the narrow band z = 1.20..1.30. A one-handed side push never crosses the
  midline, so that constraint is gone and the table can drop. It has to: the palm must
  reach the UPPER part of the block for leverage, and the block is now taller.

  BLOCK SHAPE, 90x90x200 -> BOX_HALF below. Bigger on every axis, and slimmer in
  proportion. Three reasons, in order of how much they matter:
    1. slimmer lowers the tipping angle (atan(a/h)), so less push does the job;
    2. once over, it lands on a face whose own tipping angle is atan(h/a) -- the reciprocal
       -- so the new face is much MORE stable than the old one. The move is one-way;
    3. the robot's only proud contact surface is a fingertip strip about 38 x 60 mm
       (grasp_contact_findings.md section 1). A wider face gives that strip somewhere
       to land that is not an edge.
  --sweep-shape runs the original block and a cube through the same push, so the effect of
  the shape is measured here rather than argued.

WHAT THIS DOES NOT DO. It turns the block 90 deg onto an adjacent face; it does not invert
it, and it does not put the block anywhere in particular -- the block ends up displaced by
about its own height in the push direction. If the requirement is a specific face down at a
specific spot, this is the first of two moves, not the whole job. The claim made here is
only the one the prompt asked for: a different face is against the table afterwards, by the
easiest means available.
"""

import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np

import pick_box
from pick_box import (
    ARM_L,
    ARM_R,
    HAND_HALF_THICK,
    IMPRATIO,
    PELVIS_H,
    Arms,
    JointMap,
    control,
)

# --- the block ---------------------------------------------------------------------
# Half-extents. 120 x 120 x 260 mm against pick_box's 90 x 90 x 200: bigger for the
# fingertip strip to press on, and slimmer so it goes over sooner. See the docstring.
BOX_HALF = np.array([0.060, 0.060, 0.130])
BOX_MASS = 0.50
BOX_FRICTION = 2.0

# --- the table ---------------------------------------------------------------------
# Low enough that the palm reaches the top of a 260 mm block, and long enough in X that
# the block still lands on it after falling its own height sideways. The push runs in -X,
# so the block is parked at +X to give itself room to land.
TABLE_TOP = 1.00
TABLE_Y = -0.45
TABLE_HALF = np.array([0.45, 0.20, TABLE_TOP / 2.0])
# The block sits well out to the LEFT, not on the midline. The push runs inward in -X and
# the hand geom can only cross the body midline in the narrow band z = 1.20..1.30
# (pick_box.py); parking the block at x = 0.18 keeps the whole stroke outboard of it, so
# the reach constraint that forced pick_box's chest-high table never applies here. Aimed at
# the midline instead, the shove waypoint misses by 163 mm and the arm simply stops short.
BOX_X = 0.18
BOX_Y = -0.36

# --- the push ----------------------------------------------------------------------
# Where on the block's face the fingertips land, as a fraction of its height. This is the
# parameter the manoeuvre actually turns on -- far more than the push distance below -- and
# it wants to be at the very TOP. Measured across the two together (--sweep-height):
#
#     frac   the same six push distances, 100..220 mm
#     0.30   never goes over at all: peaks at 8-13 deg and settles back
#     0.70   over, but peaks at 105-180 deg: the block is tumbling, not tipping
#     0.90   over, but three of six end up off the table
#     0.95   90, 90, 91, 92, 91 deg -- every one flat, on the table, on its -X face
#
# High is better because of when the hand LOSES the block. As the block turns about its
# far bottom edge, the contact point rides up and inboard; with the strip at the top edge
# the block's corner slips off it within a few degrees and the block falls free. Pushing
# lower keeps the palm on the face all the way down, and the hand then drives a block that
# gravity is already turning -- that is what the 180 deg peaks in the 0.70 row are.
PUSH_FRAC = 0.95
# Metres of horizontal travel commanded after the palm touches. Once the block goes over
# nothing stops the hand, so this is over-travel into empty space, not a squeeze -- which
# is why it does not have to be accurate. At PUSH_FRAC anything from 120 to 220 mm gives
# the same 90 deg result; 160 is the middle of that. See --sweep-shove.
SHOVE = 0.16
STANDOFF = 0.05           # air between palm and block at the start of the push
PARK = np.array([-0.42, -0.15, 1.50])       # where the unused arm waits

# The hand pulls back the instant it reaches the deepest point -- there is no dwell. That
# is not a detail. Traced at 10 Hz, an earlier version held the palm in place for two
# seconds after the shove and the block did not fall: it leant against the stationary hand
# and ROCKED, 100 deg to 177 deg and back, four times, and where it finally landed depended
# on which way it happened to be swinging when the hand eventually moved. Past its tipping
# angle a block on the table has only one way to go, but only if there is nothing holding
# it up. So the palm leaves.
PHASES = [("home", 0.0), ("spread", 2.5), ("set", 5.0), ("shove", 7.0),
          ("retreat", 8.5), ("done", 11.0)]
IK_PHASES = ("spread", "set", "shove", "retreat")

FACE_RGBA = [[0.85, 0.25, 0.20, 1.0], [0.95, 0.75, 0.15, 1.0],
             [0.20, 0.55, 0.85, 1.0], [0.30, 0.70, 0.35, 1.0],
             [0.60, 0.30, 0.70, 1.0], [0.95, 0.95, 0.92, 1.0]]


def tip_deg(half=BOX_HALF) -> float:
    """Tilt past which the block's centre of mass clears the bottom edge it turns about."""
    return float(np.degrees(np.arctan(half[0] / half[2])))


def build_scene(table_top=TABLE_TOP, box_half=BOX_HALF, pads=False) -> mujoco.MjModel:
    """pick_box.build_model's scene, with the table and the block made parameters.

    Same pinned pelvis, same elliptic cone and impratio. The differences are that the
    tabletop height and the block's half-extents are arguments rather than constants, and
    that each of the block's six faces gets its own colour -- with no collision and no
    density, so they are decoration and nothing else. Without them you cannot see from the
    viewer whether the block turned onto a new face or merely slid.
    """
    spec = mujoco.MjSpec.from_file(str(pick_box.MJCF))
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = IMPRATIO

    pelvis = spec.body("pelvis")
    spec.delete(pelvis.joints[0])
    pelvis.pos = [0.0, 0.0, PELVIS_H]

    table = spec.worldbody.add_body()
    table.name = "table"
    table.pos = [0.0, TABLE_Y, table_top / 2.0]
    tg = table.add_geom()
    tg.name = "table_geom"
    tg.type = mujoco.mjtGeom.mjGEOM_BOX
    tg.size = [TABLE_HALF[0], TABLE_HALF[1], table_top / 2.0]
    tg.rgba = [0.55, 0.42, 0.30, 1.0]
    tg.contype, tg.conaffinity = 1, 1

    box = spec.worldbody.add_body()
    box.name = "box"
    box.pos = [BOX_X, BOX_Y, table_top + box_half[2]]
    box.add_freejoint()
    bg = box.add_geom()
    bg.name = "box_geom"
    bg.type = mujoco.mjtGeom.mjGEOM_BOX
    bg.size = list(box_half)
    bg.rgba = FACE_RGBA[5]
    bg.mass = BOX_MASS
    bg.contype, bg.conaffinity = 1, 1
    bg.friction = [BOX_FRICTION, 0.02, 0.001]

    # one thin coloured plate per face, purely so the turn is visible
    for i, (axis, sign) in enumerate([(a, s) for a in range(3) for s in (-1, 1)]):
        size = box_half.copy() * 0.92
        size[axis] = 0.002
        pos = np.zeros(3)
        pos[axis] = sign * box_half[axis]
        d = box.add_geom()
        d.name = f"face_{'xyz'[axis]}{'-+'[sign > 0]}"
        d.type = mujoco.mjtGeom.mjGEOM_BOX
        d.size = list(size)
        d.pos = list(pos)
        d.rgba = FACE_RGBA[i]
        d.contype, d.conaffinity = 0, 0
        d.density = 0.0

    model = spec.compile()
    if pads:
        pick_box.add_palm_pads(spec, model)
        model = spec.compile()
    return model


def finger_band(model, side="left") -> tuple:
    """Measured z extent of the palm's PROUD contact strip, relative to the geom frame.

    The hand is the wrist_pitch_link STL's convex hull and it is not flat: the fingertips
    sit within a millimetre of the gripping plane while the palm behind them is recessed
    15-20 mm (rotate_around_x_axis_v3.py, section 1). So the strip that will actually touch
    the block is nowhere near the middle of the hand, and aiming the block's face at the
    geom CENTRE would put the contact a hundred millimetres too low. This reads the strip
    off the mesh instead of taking that on trust.
    """
    g = pick_box.hand_geom(model, side)
    mid = model.geom_dataid[g]
    adr, num = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
    v = model.mesh_vert[adr:adr + num].reshape(-1, 3)
    sign = -1.0 if side == "left" else 1.0        # the face that grips
    x = sign * v[:, 0]
    z = np.sort(v[x >= x.max() - 0.002, 2])
    # The proud vertices come in TWO clusters, not one: the fingertips low down and a
    # small patch at the wrist end ~200 mm above them, with the recessed palm between.
    # Averaging across both would aim the push at the empty middle of the hand, so split
    # at the largest gap and keep the lower, larger group -- the fingertips.
    # in this mesh -z is downward, so the fingertips are the LOWER group.
    tips = z[:int(np.argmax(np.diff(z))) + 1]
    return float(tips.min()), float(tips.max())


def hand_targets(phase, model, box_half, table_top, shove, standoff, push_frac=None):
    """World targets for the two hand geom centres (left, right).

    The left hand does everything; the right waits at PARK. Heights are expressed through
    the measured fingertip strip, so `push_z` is where the block gets touched and the geom
    centre target is derived from it rather than guessed.
    """
    lo, hi = finger_band(model, "left")
    drop = 0.5 * (lo + hi)                     # strip centre, below the geom frame origin
    push_z = table_top + (PUSH_FRAC if push_frac is None else push_frac) * 2.0 * box_half[2] - drop
    touch_x = BOX_X + box_half[0] + HAND_HALF_THICK

    if phase == "spread":
        return np.array([0.42, -0.15, 1.50]), PARK.copy()
    if phase == "set":
        return np.array([touch_x + standoff, BOX_Y, push_z]), PARK.copy()
    if phase == "shove":
        return np.array([touch_x - shove, BOX_Y, push_z]), PARK.copy()
    if phase == "retreat":
        return np.array([touch_x + 0.20, BOX_Y, push_z + 0.10]), PARK.copy()
    raise ValueError(phase)


def plan(model, jm, box_half, table_top, shove, standoff, push_frac=None, verbose=True) -> dict:
    """Solve IK once per waypoint; return full joint-target vectors. No squeeze anywhere.

    pick_box.plan needs an add_squeeze() pass because a pinch has to command the hands
    INTO the box to make grip force. A push needs no such thing: the over-travel is the
    trajectory itself, and the block moves out of the way rather than resisting.
    """
    scratch = mujoco.MjData(model)
    arms = Arms(model)

    home = np.zeros(model.nu)
    home[jm.index("left_shoulder_roll_joint")] = 0.25
    home[jm.index("right_shoulder_roll_joint")] = -0.25

    poses = {"home": home}
    seed = np.deg2rad([-20, 70, -20, -40, 0, 0, 0, -20, -70, 20, 40, 0, 0, 0])
    for name in IK_PHASES:
        tl, tr = hand_targets(name, model, box_half, table_top, shove, standoff, push_frac)
        mujoco.mj_resetData(model, scratch)
        scratch.qpos[jm.qadr] = home
        scratch.qpos[arms.qids] = np.clip(seed, arms.lo, arms.hi)
        q, el, er = arms.solve(scratch, tl, tr)
        full = home.copy()
        for n, v in zip(ARM_L + ARM_R, q):
            full[jm.index(n)] = v
        poses[name] = full
        seed = q
        if verbose:
            print(f"  IK {name:<8} left {el * 1000:6.1f} mm   right {er * 1000:6.1f} mm")

    poses["done"] = poses["retreat"]
    return poses


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


def down_face(model, data) -> tuple:
    """Which of the block's own faces lies against the table, and how squarely.

    Column i of the body's rotation matrix is where its local axis i points in world, so
    the local axis with the largest downward component names the resting face.
    """
    R = data.xmat[model.body("box").id].reshape(3, 3)
    dots = R.T @ np.array([0.0, 0.0, -1.0])
    i = int(np.argmax(np.abs(dots)))
    return f"{'+' if dots[i] > 0 else '-'}{'XYZ'[i]}", abs(dots[i])


def box_tilt(model, data, q0) -> float:
    """Degrees the block has turned since q0, about any axis."""
    q_inv, q_rel, w = np.zeros(4), np.zeros(4), np.zeros(3)
    mujoco.mju_negQuat(q_inv, q0)
    mujoco.mju_mulQuat(q_rel, data.xquat[model.body("box").id], q_inv)
    mujoco.mju_quat2Vel(w, q_rel, 1.0)
    return float(np.degrees(np.linalg.norm(w)))


def touching(model, data) -> int:
    """Contact points between the block and either hand."""
    bg = model.geom("box_geom").id
    hands = pick_box.hand_collision_geoms(model)
    n = 0
    for i in range(data.ncon):
        c = data.contact[i]
        if bg in (c.geom1, c.geom2):
            if hands.get(c.geom2 if c.geom1 == bg else c.geom1) is not None:
                n += 1
    return n


def run_once(model, jm, poses, table_top, box_half, log=None) -> dict:
    """The whole push, plus two seconds of settling. Returns everything main() reports."""
    box = model.body("box").id
    data = mujoco.MjData(model)
    data.qpos[jm.qadr] = poses["home"]
    mujoco.mj_forward(model, data)

    q0 = data.xquat[box].copy()
    p0 = data.xpos[box].copy()
    face0, _ = down_face(model, data)
    peak, nxt = 0.0, 0.0
    end = PHASES[-1][1] + 3.0
    while data.time < end:
        tgt, phase = target_at(data.time, poses)
        control(model, data, jm, tgt)
        mujoco.mj_step(model, data)
        peak = max(peak, box_tilt(model, data, q0))
        if log is not None and data.time >= nxt:
            nxt += 1.0
            log.append((data.time, phase, data.xpos[box][0], data.xpos[box][2],
                        box_tilt(model, data, q0), touching(model, data)))

    face1, square = down_face(model, data)
    p1 = data.xpos[box].copy()
    # lowest corner: the block's half-extents projected onto world Z, which is the bottom
    # row of its rotation matrix -- not the whole matrix
    lowest = p1[2] - float(np.abs(data.xmat[box].reshape(3, 3)[2]) @ box_half)
    on_table = (lowest > table_top - 0.02
                and abs(p1[0]) < TABLE_HALF[0] and abs(p1[1] - TABLE_Y) < TABLE_HALF[1])
    return dict(face0=face0, face1=face1, square=square, peak=peak,
                final=box_tilt(model, data, q0), moved=float(p1[0] - p0[0]),
                z_end=float(p1[2]), on_table=bool(on_table),
                held=touching(model, data), data=data)


def verdict(r, box_half) -> bool:
    """Turned onto a new face, still on the table, lying flat, and the hand is off it."""
    return (r["face1"] != r["face0"] and r["on_table"] and r["held"] == 0
            and r["square"] > 0.999)


def build_and_run(shove=SHOVE, table_top=TABLE_TOP, box_half=BOX_HALF, standoff=STANDOFF,
                  push_frac=None, pads=False, verbose=False, log=None):
    """One complete trial: build the scene, plan the push, run it, report."""
    model = build_scene(table_top, box_half, pads)
    jm = JointMap(model)
    poses = plan(model, jm, box_half, table_top, shove, standoff, push_frac, verbose)
    return model, jm, poses, run_once(model, jm, poses, table_top, box_half, log)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--headless", action="store_true", help="no window; report the outcome")
    ap.add_argument("--shove", type=float, default=SHOVE,
                    help="metres of horizontal travel commanded after the palm touches")
    ap.add_argument("--table", type=float, default=TABLE_TOP, help="tabletop height, m")
    ap.add_argument("--standoff", type=float, default=STANDOFF)
    ap.add_argument("--pads", action="store_true",
                    help="add pick_box's palm pads; the push does not need them")
    ap.add_argument("--sweep-shove", action="store_true",
                    help="the outcome across push distances -- the width of the window")
    ap.add_argument("--sweep-height", action="store_true",
                    help="push height against push distance -- the table in PUSH_FRAC")
    ap.add_argument("--sweep-shape", action="store_true",
                    help="the same push against blocks of different proportions")
    ap.add_argument("--sweep-table", action="store_true",
                    help="the same push at different tabletop heights")
    args = ap.parse_args()

    if args.sweep_shove:
        print("push distance vs outcome. Compare with rotate_around_x_axis_v3.py, where "
              "10 microns\nof release height decides the same question.\n")
        print(f"  {'shove mm':>9} {'peak tilt':>10} {'final tilt':>11} {'moved mm':>9} "
              f"{'face':>6} {'result':>9}")
        for shove in (0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.20, 0.24):
            _, _, _, r = build_and_run(shove=shove, table_top=args.table, pads=args.pads)
            print(f"  {shove * 1000:9.0f} {r['peak']:9.1f}d {r['final']:10.1f}d "
                  f"{r['moved'] * 1000:9.0f} {r['face1']:>6} "
                  f"{'TURNED' if verdict(r, BOX_HALF) else 'failed':>9}")
        return 0

    if args.sweep_height:
        print("where on the face to push, against how far. This is the measurement behind "
              "PUSH_FRAC:\nheight decides the outcome and distance barely matters, which "
              "is the opposite of what\nthe leverage argument alone would suggest. Cells "
              "are peak tilt in degrees.\n")
        shoves = (0.10, 0.12, 0.14, 0.16, 0.18, 0.22)
        print("   frac " + "".join(f"{s * 1000:>8.0f}" for s in shoves) + "   (shove mm)")
        for frac in (0.30, 0.45, 0.60, 0.70, 0.80, 0.90, 0.95, 1.00):
            cells = []
            for s in shoves:
                _, _, _, r = build_and_run(shove=s, table_top=args.table, push_frac=frac,
                                           pads=args.pads)
                ok = verdict(r, BOX_HALF)
                cells.append(f"{'OK' if ok else '.':>2}{r['peak']:5.0f}")
            print(f"  {frac:5.2f} " + "".join(f"{c:>8}" for c in cells))
        return 0

    if args.sweep_shape:
        print("the same push against different blocks. tip angle is atan(a/h) -- the tilt "
              "at which\nthe block goes over on its own; below it the arm has to do all "
              "the work.\n")
        print(f"  {'block mm':>16} {'tip deg':>8} {'peak tilt':>10} {'moved mm':>9} "
              f"{'face':>6} {'result':>9}")
        shapes = [np.array([0.045, 0.045, 0.100]),     # pick_box's original block
                  np.array([0.060, 0.060, 0.130]),     # this file's default
                  np.array([0.070, 0.070, 0.180]),     # taller still
                  np.array([0.090, 0.090, 0.110]),     # squat
                  np.array([0.080, 0.080, 0.080])]     # a cube: tip angle 45 deg
        for half in shapes:
            _, _, _, r = build_and_run(shove=args.shove, table_top=args.table,
                                       box_half=half, pads=args.pads)
            dims = "x".join(f"{d * 2000:.0f}" for d in half)
            print(f"  {dims:>16} {tip_deg(half):7.1f}d {r['peak']:9.1f}d "
                  f"{r['moved'] * 1000:9.0f} {r['face1']:>6} "
                  f"{'TURNED' if verdict(r, half) else 'failed':>9}")
        return 0

    if args.sweep_table:
        print("the same push at different tabletop heights. The palm has to reach "
              f"{PUSH_FRAC:.0%} of the\nway up a {BOX_HALF[2] * 2000:.0f} mm block, so "
              "this is really a question about arm reach.\n")
        print(f"  {'table m':>8} {'touch z':>8} {'IK err mm':>10} {'peak tilt':>10} "
              f"{'face':>6} {'result':>9}")
        for top in (0.90, 0.95, 1.00, 1.05, 1.10, 1.15):
            model, jm, poses, r = build_and_run(shove=args.shove, table_top=top,
                                                pads=args.pads)
            arms = Arms(model)
            scratch = mujoco.MjData(model)
            scratch.qpos[jm.qadr] = poses["shove"]
            mujoco.mj_forward(model, scratch)
            tl, _ = hand_targets("shove", model, BOX_HALF, top, args.shove, args.standoff)
            err = np.linalg.norm(tl - scratch.geom_xpos[arms.geoms[0]])
            print(f"  {top:8.2f} {top + PUSH_FRAC * 2 * BOX_HALF[2]:8.3f} {err * 1000:10.1f} "
                  f"{r['peak']:9.1f}d {r['face1']:>6} "
                  f"{'TURNED' if verdict(r, BOX_HALF) else 'failed':>9}")
        return 0

    box_half = BOX_HALF
    model = build_scene(args.table, box_half, args.pads)
    jm = JointMap(model)
    lo, hi = finger_band(model, "left")
    print(f"block {2000 * box_half[0]:.0f} x {2000 * box_half[1]:.0f} x "
          f"{2000 * box_half[2]:.0f} mm, {BOX_MASS} kg, on a table at {args.table:.2f} m")
    print(f"it goes over on its own past {tip_deg(box_half):.1f} deg of tilt, and it tips "
          f"rather than slides for any push above "
          f"{1000 * box_half[0] / BOX_FRICTION:.0f} mm up the face")
    print(f"the palm's proud fingertip strip measures z {lo * 1000:.0f}..{hi * 1000:.0f} mm "
          f"in the hand's own frame; the push aims it at "
          f"{args.table + PUSH_FRAC * 2 * box_half[2]:.3f} m")
    print("\nplanning the push:")
    poses = plan(model, jm, box_half, args.table, args.shove, args.standoff)

    if not args.headless:
        data = mujoco.MjData(model)
        data.qpos[jm.qadr] = poses["home"]
        mujoco.mj_forward(model, data)
        print(f"\nresting on its {down_face(model, data)[0]} face")
        print("left-drag orbit | scroll zoom | Space pause | Esc quit\n")
        with mujoco.viewer.launch_passive(model, data) as viewer:
            wall = time.perf_counter()
            while viewer.is_running():
                tgt, _ = target_at(data.time, poses)
                control(model, data, jm, tgt)
                mujoco.mj_step(model, data)
                viewer.sync()
                lag = data.time - (time.perf_counter() - wall)
                if lag > 0:
                    time.sleep(lag)
        return 0

    log = []
    r = run_once(model, jm, poses, args.table, box_half, log)
    print(f"\n{'t':>5} {'phase':<16} {'box x':>7} {'box z':>7} {'tilt':>8} "
          f"{'hand contacts':>14}")
    for t, p, x, z, tilt, n in log:
        print(f"{t:5.1f} {p:<16} {x:7.3f} {z:7.3f} {tilt:7.1f}d {n:14d}")

    print(f"\nblock turned {r['peak']:.1f} deg at most and came to rest {r['final']:.1f} "
          f"deg from where it started")
    print(f"it slid/fell {r['moved'] * 1000:.0f} mm along X and settled at "
          f"z={r['z_end']:.3f} m")
    print(f"it now rests on its {r['face1']} face (started on {r['face0']}), "
          f"{np.degrees(np.arccos(min(r['square'], 1.0))):.1f} deg off flat")
    for label, ok in (("landed on a new face", r["face1"] != r["face0"]),
                      ("lying flat, not propped on an edge", r["square"] > 0.999),
                      ("still on the table", r["on_table"]),
                      ("hand clear of it", r["held"] == 0)):
        print(f"  [{'x' if ok else ' '}] {label}")
    ok = verdict(r, box_half)
    print("RESULT:", "TURNED ONTO A NEW FACE" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
