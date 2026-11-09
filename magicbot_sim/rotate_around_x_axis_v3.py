"""Bimanual roll, version 3: grip with a FACE instead of a point, and measure the corner.

    python rotate_around_x_axis_v3.py                # interactive viewer
    python rotate_around_x_axis_v3.py --headless     # no window, reports what the box did
    python rotate_around_x_axis_v3.py --show-pads    # draw the pads instead of hiding them
    python rotate_around_x_axis_v3.py --no-pads      # v2's grasp, with v3's set-down

Same grasp, same roll, same release order as rotate_around_x_axis_v2.py, which this file
imports rather than repeats. Two things differ, and the second exists because of the first.

1. THE PALMS GET A COLLISION FACE. Measured over v2's carry, the LEFT hand holds the box
   with exactly ONE contact point for every step of the lift, and one point on ~50% of the
   steps of the roll and hold. A condim=3 contact carries force but no torque, so a hand on
   one point resists no rotation about that point whatsoever, and the box will not sit
   still: it turned 0.47 deg in a typical displayed frame and REVERSED its direction of
   rotation between 39% of consecutive frames -- a visible shimmer.

   The cause is the hand's shape, not the trajectory or the solver. The collision shape is
   the wrist_pitch_link STL's 611-vertex convex hull, and that hull touches its own
   gripping plane in two separate clusters:

       z -135..-95 mm   the fingertips   within 0.7-1.5 mm of the plane   64 verts
       z  -95..+45 mm   the palm         RECESSED 15-20 mm                 0 verts
       z  +45..+85 mm   the wrist end    within 0.0-0.4 mm                 11 verts

   The robot grips with its fingertips, and the proud fingertip strip is only ~8 mm wide.
   Note what this rules out: no rotation of the hand fixes it, because the fingertips and
   the palm differ by 15-20 mm in DEPTH, not in angle -- any attitude that brings the palm
   onto the box lifts the fingertips off it.

   So pick_box.PALM_PADS lays a flat box geom across the fingertips, its outer face flush
   with the gripping plane. That makes the pair box-vs-BOX, which uses MuJoCo's
   special-cased collider instead of the general convex one. Measured, this file vs v2:

       grasp            contact points L/R      deg per frame, direction reversals
                        lift    roll    hold    lift          roll        hold
       v2 (hull only)   1.0/1.5 1.6/2.6 1.5/3.7 0.47 @ 39%   0.18 @ 39%  0.04 @ 49%
       v3 (with pads)   5.0/5.0 5.2/5.4 4.9/7.3 0.01 @  1%   0.08 @  6%  0.02 @ 10%

   Uncommanded tilt goes from 1.25 deg rms / 15.4 deg peak-to-peak to 0.32 / 1.5. That is
   the same result the condim=6 ablation gets, reached by giving the grasp a real face
   rather than by changing the friction model. Penetration also drops from 17 to 12 mm,
   because a face carries the squeeze instead of letting the hull sink into the box.

   Three things that do NOT work, all measured and all cheaper than the pads:
     * a softer box (solref -200, -50) changes NOTHING -- byte-identical results. When
       either geom's solref is negative MuJoCo takes the elementwise MINIMUM of the pair,
       so the hand's -500 always wins and the box's value is discarded. Softness would
       have to go on the HAND. And it would not help anyway: MuJoCo geoms are rigid, so
       softness buys interpenetration, not deformation, and the collider still runs on the
       undeformed box.
     * a harder squeeze (45, 60 mm of over-travel) makes it WORSE -- the left hand drops
       to 1.00 points and penetration reaches 27-32 mm, past the box's 45 mm half-width.
       Grip force is already 116 N on a 3 N box.
     * grabbing lower. There IS room -- the hull truly clears the tabletop by 35 mm, not
       the 2.7 mm pick_box_v2.palm_clearance reports, because that uses the geom's
       BOUNDING BOX and under-reports by ~33 mm. But lowering the grasp until the palm
       covered the box's whole 200 mm face left the left hand at 1.00 points regardless.
       Coverage was never the problem.

2. THE SET-DOWN MEASURES THE CORNER INSTEAD OF PREDICTING IT. The pads move the box's
   carried attitude by about 34 deg (v2 settles at +21.6 deg of fore-aft lean, v3 at
   -12.8), so v2's descent cannot simply be reused -- and v2's half_height() cannot be
   either, because it models the box as tilted about Y ALONE:

       half_height(tilt) = hx*|sin tilt| + hz*|cos tilt|

   That is exact for a pure Y roll and wrong by the fore-aft lean otherwise, which is
   exactly what the pads change. So this version stops predicting the lowest corner and
   measures it: it transforms all eight box corners into world coordinates and takes the
   minimum z. Same two-pass structure as v2, one less approximation in it.

WHAT IS STILL FRAGILE, AND IT IS NOT THE GRASP. The set-down window is savagely narrow.
Sweeping v2's descent at 10-micron resolution:

    descend 0.11959010235910551 -> 86.1 deg, +X face   OK
    descend 0.11960000000000000 ->  0.0 deg, -Z face   FELL BACK
    descend 0.11950000000000000 ->  0.0 deg, -Z face   FELL BACK
    descend 0.11900000000000000 ->  0.0 deg, -Z face   FELL BACK
    descend 0.11800000000000000 -> 86.0 deg, +X face   OK
    descend 0.11500000000000000 -> 90.0 deg, +X face   OK

Ten microns flips a clean 90 deg turn into complete failure. v2's docstring claims
RELEASE_GAP = 0.010 sits "in the middle of the band that works"; at fine resolution the
outcome is not a band at all. Whether THIS file's grasp lands the box on a new face is
reported by --headless and is not assumed anywhere in this docstring -- and one passing run
would not establish it. Use --sweep-gap to see the outcome across several release gaps,
which is the only honest way to ask the question.
"""

import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np

import pick_box
from pick_box import (
    BOX_HALF_XY,
    BOX_HALF_Z,
    SQUEEZE,
    TABLE_TOP,
    JointMap,
    build_model,
    control,
    grip_force,
    hand_collision_geoms,
)
from pick_box_v2 import PREROLL_DEG
from rotate_around_x_axis import (
    ATTITUDE_FRAC,
    RELEASE,
    box_roll,
    down_face,
    plan,
    target_at,
    timeline,
)
from rotate_around_x_axis_v2 import ANGLE, MIN_ROLL_DEG, RELEASE_GAP, TIP_DEG

BOX_CORNERS = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1)
                        for sz in (-1, 1)], dtype=float) * [BOX_HALF_XY, BOX_HALF_XY,
                                                            BOX_HALF_Z]


def lowest_corner(model, data, box) -> float:
    """World z of the box's lowest corner, measured rather than predicted.

    v2 predicts this with hx*|sin tilt| + hz*|cos tilt|, which is exact for a roll about Y
    alone and wrong by however much the box also leans fore-aft -- and the palm pads change
    that lean by ~34 deg. Eight corners through the box's own rotation matrix costs nothing
    and has no such assumption in it.
    """
    R = data.xmat[box].reshape(3, 3)
    return float((data.xpos[box] + BOX_CORNERS @ R.T)[:, 2].min())


def box_contacts(model, data, hands) -> int:
    """Contact points holding the box, counting the palm pads as part of the hand."""
    bg = model.geom("box_geom").id
    n = 0
    for i in range(data.ncon):
        c = data.contact[i]
        if bg in (c.geom1, c.geom2):
            if hands.get(c.geom2 if c.geom1 == bg else c.geom1) is not None:
                n += 1
    return n


def simulate(model, data, jm, poses, phases, until, box, q0, hands, log=None):
    """Step to `until`, optionally logging a row a second. Returns the peak box height."""
    peak = data.xpos[box][2]
    nxt = data.time
    while data.time < until:
        tgt, phase = target_at(data.time, poses, phases)
        control(model, data, jm, tgt)
        mujoco.mj_step(model, data)
        peak = max(peak, data.xpos[box][2])
        if log is not None and data.time >= nxt:
            nxt += 1.0
            log.append((data.time, phase, data.xpos[box][2],
                        box_roll(model, data, q0)[1], grip_force(model, data),
                        box_contacts(model, data, hands)))
    return peak


def fresh(model, jm, angle, descend, release, squeeze, preroll, frac):
    phases, sched = timeline(angle, 6, 6.0, descend, release)
    poses = plan(model, jm, phases, sched, squeeze=squeeze, preroll_deg=preroll,
                 use_waist=True, frac=frac, verbose=False)
    data = mujoco.MjData(model)
    data.qpos[jm.qadr] = poses["home"]
    mujoco.mj_forward(model, data)
    return phases, dict(phases), poses, data


def derive_descend(model, jm, args, box, gap, verbose=True):
    """Carry the box through the roll, then measure how far its lowest corner has to fall."""
    phases, at, poses, data = fresh(model, jm, args.angle, 0.0, args.release,
                                    args.squeeze, args.preroll, args.attitude_frac)
    q0 = data.xquat[box].copy()
    hands = hand_collision_geoms(model)
    simulate(model, data, jm, poses, phases, at["hold"], box, q0, hands)
    corner = lowest_corner(model, data, box)
    descend = corner - TABLE_TOP - gap
    if verbose:
        print(f"  carried {box_roll(model, data, q0)[1]:.1f} deg of roll; box centre at "
              f"z={data.xpos[box][2]:.3f}, lowest corner measured at z={corner:.3f}")
        print(f"  descending {descend:.3f} m leaves {gap * 1000:.0f} mm of air under that "
              f"corner at the release")
    return descend


def run_once(model, jm, args, box, descend, log=None):
    """The whole manoeuvre with a given descent. Returns everything main() reports."""
    phases, at, poses, data = fresh(model, jm, args.angle, descend, args.release,
                                    args.squeeze, args.preroll, args.attitude_frac)
    hands = hand_collision_geoms(model)
    z0 = data.xpos[box][2]
    q0 = data.xquat[box].copy()
    face0, _ = down_face(model, data)
    lifted = simulate(model, data, jm, poses, phases, at["descend"], box, q0, hands, log)
    at_release = box_roll(model, data, q0)[1]
    peak = max(lifted, simulate(model, data, jm, poses, phases, phases[-1][1] + 2.0,
                                box, q0, hands, log))
    total, rolled = box_roll(model, data, q0)
    face1, square = down_face(model, data)
    return dict(z0=z0, face0=face0, peak=peak, at_release=at_release, total=total,
                rolled=rolled, face1=face1, square=square, z_end=data.xpos[box][2],
                held_by=box_contacts(model, data, hands), phases=phases, poses=poses)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--headless", action="store_true", help="no window; report the outcome")
    ap.add_argument("--angle", type=float, default=ANGLE,
                    help="degrees of roll to command before setting the box down")
    ap.add_argument("--gap", type=float, default=RELEASE_GAP,
                    help="metres of air left under the box's lowest corner at the release")
    ap.add_argument("--descend", type=float, default=None,
                    help="override the derived descent, in metres")
    ap.add_argument("--sweep-gap", action="store_true",
                    help="report the outcome across several release gaps, not just one")
    ap.add_argument("--no-pads", action="store_true",
                    help="v2's point grasp, with v3's measured set-down")
    ap.add_argument("--show-pads", action="store_true",
                    help="draw the palm pads instead of leaving them invisible")
    ap.add_argument("--squeeze", type=float, default=SQUEEZE)
    ap.add_argument("--preroll", type=float, default=PREROLL_DEG)
    ap.add_argument("--attitude-frac", type=float, default=ATTITUDE_FRAC)
    ap.add_argument("--release", type=float, default=RELEASE)
    args = ap.parse_args()

    if args.show_pads:
        pick_box.PAD_RGBA = [0.15, 0.80, 0.35, 0.55]
    model = build_model(palm_pads=not args.no_pads)
    jm = JointMap(model)
    box = model.body("box").id

    print(f"box tips over on its own past {TIP_DEG:.1f} deg; commanding "
          f"{args.angle:.0f} deg of roll, "
          f"{'WITHOUT' if args.no_pads else 'with'} palm pads "
          f"({len(hand_collision_geoms(model))} collidable hand geoms)")

    if args.sweep_gap:
        print("\nthe outcome across release gaps -- one run proves nothing here, see the "
              "module docstring")
        print(f"  {'gap mm':>7} {'descend':>9} {'at release':>11} {'came to rest':>13} "
              f"{'face':>6} {'result':>10}")
        for gap in (0.006, 0.008, 0.010, 0.012, 0.014, 0.018):
            descend = derive_descend(model, jm, args, box, gap, verbose=False)
            r = run_once(model, jm, args, box, descend)
            ok = (abs(r["at_release"]) >= MIN_ROLL_DEG and r["z_end"] > TABLE_TOP
                  and r["held_by"] == 0 and r["face1"] != r["face0"])
            print(f"  {gap * 1000:7.0f} {descend:9.4f} {r['at_release']:10.1f}d "
                  f"{r['rolled']:12.1f}d {r['face1']:>6} "
                  f"{'TURNED' if ok else 'failed':>10}")
        return 0

    if args.descend is None:
        print("\npass 1: rolling, to measure where the box's lowest corner ends up...")
        descend = derive_descend(model, jm, args, box, args.gap)
    else:
        descend = args.descend
        print(f"\ndescent overridden to {descend:.3f} m")

    print("\npass 2: the whole thing, with that descent")
    if not args.headless:
        phases, at, poses, data = fresh(model, jm, args.angle, descend, args.release,
                                        args.squeeze, args.preroll, args.attitude_frac)
        print(f"table top {TABLE_TOP:.3f} m, box centre starts at "
              f"z={data.xpos[box][2]:.3f} m, resting on its {down_face(model, data)[0]} face")
        print("\nleft-drag orbit | scroll zoom | Space pause | Esc quit\n")
        with mujoco.viewer.launch_passive(model, data) as viewer:
            wall = time.perf_counter()
            while viewer.is_running():
                tgt, _ = target_at(data.time, poses, phases)
                control(model, data, jm, tgt)
                mujoco.mj_step(model, data)
                viewer.sync()
                lag = data.time - (time.perf_counter() - wall)
                if lag > 0:
                    time.sleep(lag)
        return 0

    log = []
    r = run_once(model, jm, args, box, descend, log)
    print(f"\n{'t':>5} {'phase':<20} {'box z':>7} {'box roll':>9} {'grip N':>8}"
          f" {'hand contacts':>14}")
    for t, p, z, roll, f, n in log:
        print(f"{t:5.1f} {p:<20} {z:7.3f} {roll:8.1f}d {f:8.1f} {n:14d}")

    print(f"\nbox rose {(max(r['peak'], r['z0']) - r['z0']) * 1000:.0f} mm above its start")
    print(f"box was tilted {r['at_release']:.1f} deg when the hands let go (commanded "
          f"{args.angle:.0f}, needed {TIP_DEG:.1f} to go over), and came to rest "
          f"{r['rolled']:.1f} deg from where it started ({r['total']:.1f} deg total)")
    print(f"box settled at z={r['z_end']:.3f} m, table top {TABLE_TOP:.3f} "
          f"-- {'ON THE TABLE' if r['z_end'] > TABLE_TOP else 'OFF THE TABLE'}")
    print(f"hand/box contacts at the end: {r['held_by']} "
          f"({'released' if r['held_by'] == 0 else 'STILL TOUCHING'})")
    print(f"box now rests on its {r['face1']} face (started on {r['face0']}), "
          f"{np.degrees(np.arccos(min(r['square'], 1.0))):.1f} deg off flat")

    turned = abs(r["at_release"]) >= MIN_ROLL_DEG
    placed = r["z_end"] > TABLE_TOP and r["held_by"] == 0
    new_face = r["face1"] != r["face0"]
    for label, ok in ((f"still past the {TIP_DEG:.1f} deg tipping angle at the release",
                       turned),
                      ("left on the table, hands clear", placed),
                      ("landed on a new face", new_face)):
        print(f"  [{'x' if ok else ' '}] {label}")
    ok = turned and placed and new_face
    print("RESULT:", "TURNED OVER AND PLACED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
