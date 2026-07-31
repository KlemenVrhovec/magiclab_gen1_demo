"""Bimanual roll, version 2: carry it only 45 deg, and let gravity do the rest.

    python rotate_around_x_axis_v2.py                # interactive viewer
    python rotate_around_x_axis_v2.py --headless     # no window, reports what the box did
    python rotate_around_x_axis_v2.py --angle 60     # carry it further before setting down

Same grasp, same roll, same release order as rotate_around_x_axis.py, which this file
imports rather than repeats. Two things differ, both of them about the SET-DOWN.

1. THE CARRY DOES NOT HAVE TO REACH 50 DEG. V1 commands 90 and carries 62.8. It does not
   need to. The box is 9 cm square and 20 cm tall, so once tilted past
   atan(0.045/0.100) = 24.2 deg its centre of mass is beyond its bottom corner, and
   setting it down and letting go is enough -- it goes over on its own. Commanding 45
   carries 32.7, which clears 24.2 with room, and the box still lands square on a new
   face. Measured, commanding 45:

       descend    carried    came to rest    face
        0.18       34.1        0.0 deg       -Z   fell back onto the face it started on
        0.16       33.0        0.0 deg       -Z   same
        0.13       32.3       86.1 deg       +X
        0.11       32.7       89.9 deg       +X   cleanest

2. THE DESCENT HAS TO BE DERIVED, NOT REUSED. That table is the whole point of this
   version. V1's DESCEND = 0.18 is not a constant of the manoeuvre, it is tuned to V1's
   65 deg tilt, and at 45 deg it is catastrophically wrong -- not by a little. A box
   tilted 33 deg is TALLER than one tilted 65: its half-height is
   0.045*sin(33) + 0.100*cos(33) = 0.108 m against 0.083 at 65. So its underside is
   already at 1.232 with the tabletop at 1.100, and descending 0.18 drives it 5 cm INTO
   the table. The palms are soft, so the table simply levers the box back upright before
   either hand lets go, and it settles on -Z having rotated nowhere. It reports a clean
   lift, a clean release and a flat landing while having achieved precisely nothing.

   So this version measures instead of assuming. It plans and simulates once to the end
   of the roll, reads the box's ACTUAL height and tilt at that moment, and only then
   works out how far to descend:

       descend = (box centre - half-height at that tilt) - tabletop - RELEASE_GAP

   RELEASE_GAP leaves the box a short drop rather than pressing it into the table, which
   is what keeps the tilt intact through the release. Costs a second planning pass.

WHY THIS IS THE INTERESTING VERSION. V1 is the stronger manipulation -- it carries the
box nearly twice as far. This one is the more honest result: almost none of the work
needs to be done by the arms at all. The arms only have to get the box past its tipping
angle and put it down without squaring it up. Everything after that is free.
"""

import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np

from pick_box import (
    BOX_HALF_XY,
    BOX_HALF_Z,
    SQUEEZE,
    TABLE_TOP,
    JointMap,
    build_model,
    control,
    grip_force,
)
from pick_box_v2 import PREROLL_DEG
from rotate_around_x_axis import (
    ATTITUDE_FRAC,
    RELEASE,
    box_contacts,
    box_roll,
    down_face,
    plan,
    target_at,
    timeline,
)

ANGLE = 45.0             # commanded roll; the box carries about 33 of it
RELEASE_GAP = 0.02       # metres of air left under the box when the hands let go

# The tilt past which the box's centre of mass clears its bottom corner and it goes over
# on its own. Everything this script does is in service of clearing this number.
TIP_DEG = float(np.degrees(np.arctan(BOX_HALF_XY / BOX_HALF_Z)))
MIN_ROLL_DEG = TIP_DEG + 5.0


def half_height(tilt: float) -> float:
    """How far the box's lowest corner sits below its centre, at a given roll.

    A box is at its tallest standing square and at its shortest on the diagonal, so this
    is not a constant and cannot be treated as one -- see the module docstring.
    """
    return BOX_HALF_XY * abs(np.sin(tilt)) + BOX_HALF_Z * abs(np.cos(tilt))


def descend_for(centre_z: float, tilt: float, gap: float = RELEASE_GAP) -> float:
    """Metres to lower the pair so the box ends `gap` above the tabletop."""
    return (centre_z - half_height(tilt)) - TABLE_TOP - gap


def simulate(model, data, jm, poses, phases, until, box, q0, log=None):
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
                        box_contacts(model, data)))
    return peak


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--headless", action="store_true", help="no window; report the outcome")
    ap.add_argument("--angle", type=float, default=ANGLE,
                    help="degrees of roll to command before setting the box down")
    ap.add_argument("--gap", type=float, default=RELEASE_GAP,
                    help="metres of air left under the box at the release")
    ap.add_argument("--descend", type=float, default=None,
                    help="override the derived descent, in metres")
    ap.add_argument("--squeeze", type=float, default=SQUEEZE)
    ap.add_argument("--preroll", type=float, default=PREROLL_DEG)
    ap.add_argument("--attitude-frac", type=float, default=ATTITUDE_FRAC)
    ap.add_argument("--release", type=float, default=RELEASE)
    args = ap.parse_args()

    model = build_model()
    jm = JointMap(model)
    box = model.body("box").id

    def fresh(descend):
        phases, sched = timeline(args.angle, 6, 6.0, descend, args.release)
        poses = plan(model, jm, phases, sched, squeeze=args.squeeze,
                     preroll_deg=args.preroll, use_waist=True,
                     frac=args.attitude_frac, verbose=False)
        data = mujoco.MjData(model)
        data.qpos[jm.qadr] = poses["home"]
        mujoco.mj_forward(model, data)
        return phases, dict(phases), poses, data

    print(f"box tips over on its own past {TIP_DEG:.1f} deg; "
          f"commanding {args.angle:.0f} deg of roll")

    if args.descend is None:
        # PASS 1: carry the box through the roll and see where it actually ends up.
        # Nothing about the descent is known until this has been measured.
        print("\npass 1: rolling, to measure the box's real height and tilt...")
        phases, at, poses, data = fresh(0.0)
        q0 = data.xquat[box].copy()
        simulate(model, data, jm, poses, phases, at["hold"], box, q0)
        tilt = np.deg2rad(box_roll(model, data, q0)[1])
        centre = data.xpos[box][2]
        descend = descend_for(centre, tilt, args.gap)
        print(f"  carried {np.degrees(tilt):.1f} deg, box centre at z={centre:.3f}, "
              f"half-height {half_height(tilt):.3f} -> underside at "
              f"{centre - half_height(tilt):.3f}")
        print(f"  descending {descend:.3f} m leaves {args.gap*1000:.0f} mm of air "
              f"under it at the release")
    else:
        descend = args.descend
        print(f"\ndescent overridden to {descend:.3f} m")

    print("\npass 2: the whole thing, with that descent")
    phases, at, poses, data = fresh(descend)
    z0 = data.xpos[box][2]
    q0 = data.xquat[box].copy()
    face0, _ = down_face(model, data)
    print(f"table top {TABLE_TOP:.3f} m, box centre starts at z={z0:.3f} m, "
          f"resting on its {face0} face")

    if not args.headless:
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
    simulate(model, data, jm, poses, phases, at["hold"], box, q0, log)
    carried = box_roll(model, data, q0)[1]
    peak = simulate(model, data, jm, poses, phases, phases[-1][1] + 2.0, box, q0, log)

    print(f"\n{'t':>5} {'phase':<20} {'box z':>7} {'box roll':>9} {'grip N':>8}"
          f" {'hand contacts':>14}")
    for t, p, z, r, f, n in log:
        print(f"{t:5.1f} {p:<20} {z:7.3f} {r:8.1f}d {f:8.1f} {n:14d}")

    total, rolled = box_roll(model, data, q0)
    held_by = box_contacts(model, data)
    z_end = data.xpos[box][2]
    face1, square = down_face(model, data)

    print(f"\nbox rose {(max(peak, z0) - z0)*1000:.0f} mm above its start height")
    print(f"box was rolled {carried:.1f} deg while carried (commanded {args.angle:.0f}, "
          f"needed {TIP_DEG:.1f} to go over), and came to rest {rolled:.1f} deg from "
          f"where it started ({total:.1f} deg total rotation)")
    print(f"box settled at z={z_end:.3f} m, table top {TABLE_TOP:.3f} "
          f"-- {'ON THE TABLE' if z_end > TABLE_TOP else 'OFF THE TABLE'}")
    print(f"hand/box contacts at the end: {held_by} "
          f"({'released' if held_by == 0 else 'STILL TOUCHING'})")
    print(f"box now rests on its {face1} face (started on {face0}), "
          f"{np.degrees(np.arccos(min(square, 1.0))):.1f} deg off flat")

    turned = abs(carried) >= MIN_ROLL_DEG
    placed = z_end > TABLE_TOP and held_by == 0
    new_face = face1 != face0
    for label, ok in ((f"carried past the {TIP_DEG:.1f} deg tipping angle", turned),
                      ("left on the table, hands clear", placed),
                      ("landed on a new face", new_face)):
        print(f"  [{'x' if ok else ' '}] {label}")
    ok = turned and placed and new_face
    print("RESULT:", "TURNED OVER AND PLACED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
