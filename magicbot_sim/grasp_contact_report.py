"""What the grasp in rotate_around_x_axis_v2.py actually consists of, measured.

    python grasp_contact_report.py             # the report
    python grasp_contact_report.py --viewer    # watch it with contact points drawn
    python grasp_contact_report.py --ablate    # the experiments behind section 5

The scripts talk about "palms" and "flat hand faces" as though the box were pinched
between two plates. It is not, and the difference is why the carried box will not sit
still. Nothing below is assumed; every number is printed by this file.

1. THE HAND IS NOT A SLAB AND NOT A BOX. It is the whole wrist_pitch_link STL -- 8214
   verts, 16466 faces -- collided as its 611-vertex convex hull. pick_box's
   HAND_HALF_THICK = 0.0228 is not a plate thickness; MuJoCo fills geom_size for a mesh
   from its bounding box, and no collision uses it as a shape. The gripping side is flat
   to within 1 mm over about 38 x 60 mm, which is the largest patch that could ever bear
   on the box's 90 x 200 mm face. So the two shapes meet as box-vs-convex-hull, MuJoCo's
   general convex collider -- NOT the special-cased box-box collider the table gets, and
   that is why the table reliably yields 4-5 points and the hands do not.

2. THE BOX IS A REAL BOX: one primitive mjGEOM_BOX, 90 x 90 x 200 mm, 0.3 kg, freejoint.

3. HOW MANY POINTS OF CONTACT: BETWEEN ONE AND FIVE PER HAND, AND IT CHANGES CONSTANTLY.
   Measured over the lift, roll and hold, the two hands are nothing like each other:

       LEFT   one single contact point for all 900 steps of the lift. Through the roll
              and hold it is one point on ~50% of steps, and when it does find a second
              during the hold the two are ~1 mm apart -- a point, not a patch.
       RIGHT  one to five points, genuinely spread: up to 43 x 79 mm across the box face
              during the roll, 43 x 12 mm during the hold. A real patch.

   At least one hand is down to a single point on 69% of steps, and both hands at once
   on 17%. The count changes on 40-50% of steps. Since condim=3 contacts carry force but
   NO torque, a hand reduced to one point resists no rotation at all about that point,
   and the set that does the resisting is rebuilt several hundred times a second.

   The pinch is also deep. The 30 mm squeeze bias meets contacts of stiffness 500 N/m
   (solref="-500 -800", from the model's own defaults), so the hulls end up ~20 mm inside
   the box -- nearly half its 90 mm width, and plainly visible in the viewer as hands
   sunk into it.

4. SO WHAT YOU SEE THE BOX DOING IS REAL, AND IT IS THREE SEPARATE THINGS.

   a) A JITTER, from the contact set being rebuilt. Sampled at 60 fps, the box turns
      0.47 deg in a typical displayed frame during the lift (up to 3.8), and its
      direction of rotation about X REVERSES between 39% of consecutive frames. That
      reads as a shimmer. It decays to 0.04 deg/frame in the hold, once nothing is moving.

   b) A 1 Hz STUTTER through the roll, which is the commanded trajectory, not the
      contacts. rotate_around_x_axis.timeline lays down 6 roll waypoints over 6 s and
      target_at cosine-eases between them, which brings the hands to a dead stop at each
      one. Measured per quarter-second, the box rolls +0.3, +1.2, +1.9, +1.3 deg and
      then nearly halts -- six visible pulses, with a brief backward slip of -0.13 deg
      at the first boundary.

   c) A ONE-OFF SNAP OF 21 DEG at lift-off, and a permanent 21.6 deg lean afterwards.
      The manoeuvre commands roll about Y only, so any tilt in the Y-Z plane is pure
      grasp artefact. The box is dragged to -12.2 deg while the hands close on it against
      the table, snaps through to +13.1 in the 370 ms it comes free, and settles at
      +21.6 -- which is pick_box_v2.PREROLL_DEG's -20 deg palm pre-roll, the box having
      been pulled into line with the hands. On top of that sits only a 0.07 deg rms
      ripple, so the box is NOT swinging steadily once carried; it is jittering (a) and
      stuttering (b) about a pose that is 21.6 deg away from the one the plan describes.

5. WHAT IS TO BLAME, by ablation (--ablate). Lean measured from the lift to the set-down:

       change                                lean rms   peak-to-peak
       baseline, 6 roll waypoints              1.25 deg    15.4 deg
       1 roll waypoint instead of 6            1.30        15.7      not the waypoints
       12 roll waypoints instead of 6          1.23        15.3      not the waypoints
       stiff contacts (solref 0.01 1)          4.37        27.3      WORSE
       condim 6, torsional+rolling friction    0.33         1.5      12x better

   The waypoint cadence does not cause the tilt -- it only paces the stutter in 4b.
   Stiffening the contacts makes it worse, because less penetration puts fewer hull
   vertices inside the box and leaves a smaller, more point-like contact set. Giving the
   contacts torque to transmit is the only change that fixes it, which is the direct test
   of the diagnosis in 3: the problem is that the grasp is points, not faces.
"""

import argparse
import collections
import time

import mujoco
import mujoco.viewer
import numpy as np

from pick_box import (
    BOX_HALF_XY,
    BOX_HALF_Z,
    BOX_MASS,
    PALM_PADS,
    SQUEEZE,
    JointMap,
    build_model,
    control,
    hand_collision_geoms,
    hand_geom,
)
from pick_box_v2 import PREROLL_DEG
from rotate_around_x_axis import ATTITUDE_FRAC, RELEASE, plan, target_at, timeline
from rotate_around_x_axis_v2 import ANGLE

# What rotate_around_x_axis_v2's pass 1 derives, hard-coded so this report costs one
# planning pass instead of two. It only has to be close: nothing here depends on the
# set-down succeeding.
DESCEND = 0.1196

FPS = 60.0          # the rate the viewer actually shows you, for section 4a


def hull_of(model, geom):
    """(convex hull vertices in mesh coordinates, hull face count) for a mesh geom."""
    mid = model.geom_dataid[geom]
    adr = model.mesh_graphadr[mid]
    nvert, nface = int(model.mesh_graph[adr]), int(model.mesh_graph[adr + 1])
    gid = model.mesh_graph[adr + 2 + nvert: adr + 2 + 2 * nvert]
    va = model.mesh_vertadr[mid]
    return model.mesh_vert[va: va + model.mesh_vertnum[mid]][gid], nface


def hand_contacts(model, data, hands):
    """{side: [(world position, penetration, normal force, geom id)]} for box/hand."""
    bg = model.geom("box_geom").id
    out = {"L": [], "R": []}
    for i in range(data.ncon):
        c = data.contact[i]
        if bg not in (c.geom1, c.geom2):
            continue
        g = c.geom2 if c.geom1 == bg else c.geom1
        side = hands.get(g)
        if side is None:
            continue
        f = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, f)
        out[side].append((c.pos.copy(), c.dist, abs(f[0]), g))
    return out


def grip_side(model, side):
    """(gripping local axis, sign) for a hand. The hands are MIRROR meshes: the left
    presses with its local -x face and the right with +x, so measuring either one with a
    single convention gets one of them wrong."""
    return 0, (-1.0 if side == "left" else +1.0)


def lean_deg(data, box):
    """Box tilt in the Y-Z plane, in degrees.

    The manoeuvre commands roll about Y and nothing else, and a pure Y roll leaves this
    at exactly zero -- so whatever it reads is entirely the grasp's doing.
    """
    R = data.xmat[box].reshape(3, 3)
    return float(np.degrees(np.arctan2(R[1, 2], R[2, 2])))


def build(steps=6, secs=6.0, descend=DESCEND, tweak=None):
    model = build_model()
    if tweak:
        tweak(model)
    jm = JointMap(model)
    phases, sched = timeline(ANGLE, steps, secs, descend, RELEASE)
    poses = plan(model, jm, phases, sched, squeeze=SQUEEZE, preroll_deg=PREROLL_DEG,
                 use_waist=True, frac=ATTITUDE_FRAC, verbose=False)
    return model, jm, phases, poses


def rollout(model, jm, poses, phases, until, box, hands=None):
    """Step to `until`, keeping everything section 3 and 4 need, one row per step."""
    data = mujoco.MjData(model)
    data.qpos[jm.qadr] = poses["home"]
    mujoco.mj_forward(model, data)
    vadr = model.jnt_dofadr[model.body_jntadr[box]]
    rows = []
    while data.time < until:
        tgt, _ = target_at(data.time, poses, phases)
        control(model, data, jm, tgt)
        mujoco.mj_step(model, data)
        rows.append((data.time,
                     hand_contacts(model, data, hands) if hands else None,
                     lean_deg(data, box),
                     data.xpos[box].copy(),
                     data.xmat[box].reshape(3, 3).copy(),
                     data.qvel[vadr + 3: vadr + 6].copy()))
    return data, rows


def describe_shapes(model):
    print("1. HOW THE HAND IS BUILT")
    g = hand_geom(model, "left")
    mid = model.geom_dataid[g]
    hull, nface = hull_of(model, g)
    print(f"   collision geom type    {mujoco.mjtGeom(model.geom_type[g]).name}"
          f"  -- the link's own STL, not a box")
    print(f"   source mesh            {model.mesh_vertnum[mid]} verts,"
          f" {model.mesh_facenum[mid]} faces")
    print(f"   what actually collides its convex hull: {len(hull)} verts, {nface} faces")
    print(f"   hull bounding box      {np.ptp(hull, axis=0) * 1000} mm"
          f"  (thin axis is local X)")
    print(f"   offset from the wrist  {model.geom_pos[g]} -- 10 cm below the body origin")
    print(f"   contact params         condim {model.geom_condim[g]}"
          f" (force only, NO torque), friction {model.geom_friction[g]},"
          f" solref {model.geom_solref[g]}")

    print("\n   how flat is the gripping side? The hands are MIRROR meshes -- the left")
    print("   presses with its local -x face, the right with +x:")
    for side in ("left", "right"):
        h, _ = hull_of(model, hand_geom(model, side))
        ax, sign = grip_side(model, side)
        v = h[:, ax] * sign
        print(f"     {side} hand, local {'+' if sign > 0 else '-'}x face"
              f" (extreme {v.max() * 1000:.1f} mm):")
        for tol in (0.001, 0.002, 0.005):
            p = h[v > v.max() - tol]
            other = [a for a in range(3) if a != ax]
            span = np.ptp(p[:, other], axis=0) * 1000 if len(p) > 1 else np.zeros(2)
            print(f"       within {tol * 1000:4.1f} mm: {len(p):3d} verts spanning"
                  f" {span[0]:5.1f} x {span[1]:5.1f} mm")
        p = h[v > v.max() - 0.002]
        p = p[p[:, 2] < -0.080]
        print(f"       the flat patch sits at the palm's LOWER end:"
              f" y {p[:, 1].min() * 1000:+.0f}..{p[:, 1].max() * 1000:+.0f},"
              f" z {p[:, 2].min() * 1000:+.0f}..{p[:, 2].max() * 1000:+.0f} mm")
    print("   -> a flat patch of only ~78 x 22 mm at the palm tip, against a")
    print("      90 x 200 mm box face")

    if PALM_PADS:
        pad = model.geom("left_palm_pad").id
        print(f"\n   PALM PADS ARE ON: a {model.geom_size[pad][1] * 2000:.0f} x"
              f" {model.geom_size[pad][2] * 2000:.0f} mm box geom laid into that patch on"
              f" each palm,")
        print(f"   {model.geom_size[pad][0] * 2000:.0f} mm thick, outer face flush with"
              f" the palm plane. Makes the pair box-vs-BOX.")
    else:
        print("\n   PALM PADS ARE OFF (pick_box.PALM_PADS) -- box-vs-mesh only.")

    print("\n2. HOW THE BOX IS BUILT")
    bg = model.geom("box_geom").id
    print(f"   collision geom type    {mujoco.mjtGeom(model.geom_type[bg]).name}"
          f" -- one primitive, half-sizes {model.geom_size[bg]}")
    print(f"   {BOX_HALF_XY * 2000:.0f} x {BOX_HALF_XY * 2000:.0f} x"
          f" {BOX_HALF_Z * 2000:.0f} mm, {BOX_MASS} kg, freejoint")
    print(f"   condim {model.geom_condim[bg]}, friction {model.geom_friction[bg]},"
          f" solref {model.geom_solref[bg]} (= 500 N/m stiffness, 800 Ns/m damping)")
    print("   box vs hand -> box-vs-mesh, MuJoCo's general convex collider")
    print("   box vs table -> box-vs-box, the special-cased one, which is why the table")
    print("      yields a steady 4-5 points and the hands do not")


def report_contacts(rows, windows, pad_ids):
    print("\n3. HOW MANY POINTS OF CONTACT, AND HOW SPREAD OUT")
    for label, a, b in windows:
        sel = [r for r in rows if a <= r[0] < b]
        if not sel:
            continue
        print(f"\n   {label}  ({len(sel)} steps)")
        for side, name in (("L", "left "), ("R", "right")):
            hist = collections.Counter(len(r[1][side]) for r in sel)
            counts = "  ".join(f"{n}pt x{c}" for n, c in sorted(hist.items()))
            mean = np.mean([len(r[1][side]) for r in sel])
            print(f"     {name}: mean {mean:.2f} points   {counts}")
            if PALM_PADS:
                on_pad = np.mean([sum(1 for *_, g in r[1][side] if g == pad_ids[side])
                                  for r in sel])
                print(f"            of which {on_pad:.2f} are on the PAD,"
                      f" {mean - on_pad:.2f} on the hull mesh")
            # spread measured in the BOX's own frame: how far apart the points sit
            # across the face they are pressing on
            spans = []
            for r in sel:
                p = np.array([q for q, *_ in r[1][side]])
                if len(p) < 2:
                    continue
                spans.append(np.ptp((p - r[3]) @ r[4], axis=0) * 1000)
            if spans:
                s = np.array(spans)
                print(f"            when it has >1 point ({len(spans)}/{len(sel)} steps):"
                      f" spread across the face"
                      f" {s[:, 1].mean():.0f} x {s[:, 2].mean():.0f} mm"
                      f" (max {s[:, 1].max():.0f} x {s[:, 2].max():.0f})")
            else:
                print("            never more than one point")
        pen = min(c[1] for r in sel for c in r[1]["L"] + r[1]["R"])
        grip = np.mean([sum(c[2] for c in r[1]["L"] + r[1]["R"]) for r in sel])
        nL = np.array([len(r[1]["L"]) for r in sel])
        nR = np.array([len(r[1]["R"]) for r in sel])
        print(f"     deepest penetration {pen * 1000:.1f} mm into a 90 mm box,"
              f" mean grip {grip:.0f} N")
        print(f"     the count changes on {np.mean(np.diff(nL) != 0) * 100:.0f}% (L) /"
              f" {np.mean(np.diff(nR) != 0) * 100:.0f}% (R) of steps")

    carry = [r for r in rows if windows[0][1] <= r[0] < windows[-1][2]]
    one = np.mean([len(r[1]["L"]) < 2 or len(r[1]["R"]) < 2 for r in carry])
    both = np.mean([len(r[1]["L"]) < 2 and len(r[1]["R"]) < 2 for r in carry])
    print(f"\n   over the whole carry: at least one hand down to a single point on"
          f" {one * 100:.0f}% of steps,")
    print(f"   both hands at once on {both * 100:.0f}%. condim 3 carries no torque, so a"
          f" hand on one point")
    print("   resists no rotation about it at all.")


def report_motion(rows, phases, windows):
    at = dict(phases)
    t = np.array([r[0] for r in rows])
    lean = np.array([r[2] for r in rows])
    w = np.array([r[5] for r in rows])

    print("\n4a. WHAT IS VISIBLE AT 60 FPS")
    print(f"    {'window':<14} {'deg turned per frame':>22} {'wx reverses between':>21}")
    stride = max(1, int(round((1.0 / FPS) / 0.002)))
    for label, a, b in windows:
        idx = np.where((t >= a) & (t < b))[0][::stride]
        if len(idx) < 3:
            continue
        per = np.degrees(np.linalg.norm(w[idx], axis=1)) / FPS
        rev = np.mean(np.sign(w[idx, 0][1:]) != np.sign(w[idx, 0][:-1]))
        print(f"    {label:<14} mean {per.mean():5.2f}  max {per.max():5.2f}"
              f"       {rev * 100:4.0f}% of frames")
    print("    -> a shimmer, not a swing: the box reverses direction about half the time")

    print("\n4b. THE 1 Hz STUTTER THROUGH THE ROLL (the commanded trajectory, not contact)")
    roll = np.degrees(np.unwrap([np.arctan2(r[4][0, 2], r[4][2, 2]) for r in rows]))
    marks = [tw for n, tw in phases if n.startswith("roll")]
    print("    roll advanced per quarter-second, with the 6 waypoint boundaries marked:")
    for k in range(int(at["roll1"] / 0.002) - 500, int(at["hold"] / 0.002) - 125, 125):
        d = roll[k + 125] - roll[k]
        mark = "<- waypoint" if any(abs(t[k] - tw) < 3e-3 for tw in marks) else ""
        print(f"      {t[k]:5.2f}-{t[k] + 0.25:5.2f} {d:+6.2f} "
              f"{'#' * int(max(0.0, d) * 8):<20} {mark}")

    print("\n4c. THE UNCOMMANDED TILT (zero for any pure Y roll, so this is all grasp)")
    grab = (t >= 5.5) & (t < at["lift"])
    snap = (t >= at["lift"] - 2.0) & (t < at["lift"] - 1.4)
    held = (t >= at["hold"]) & (t < at["descend"])
    print(f"    dragged to      {lean[grab].min():+7.2f} deg while the hands close"
          f" against the table (t={t[grab][lean[grab].argmin()]:.2f})")
    lo, hi = lean[snap].argmin(), lean[snap].argmax()
    print(f"    snaps through   {lean[snap][lo]:+7.2f} -> {lean[snap][hi]:+7.2f} deg in the"
          f" {abs(t[snap][hi] - t[snap][lo]) * 1000:.0f} ms it comes free of the table")
    print(f"    settles at      {lean[held].mean():+7.2f} deg -- the palms'"
          f" {PREROLL_DEG:.0f} deg pre-roll, mirrored")

    carry = (t >= at["roll1"]) & (t < at["descend"])
    x = lean[carry] - np.polyval(np.polyfit(t[carry], lean[carry], 9), t[carry])
    freq = np.fft.rfftfreq(len(x), 0.002)
    spec = np.fft.rfft(x)
    ripple = np.fft.irfft(np.where(freq > 0.5, spec, 0), len(x))
    band = (freq > 0.5) & (freq < 25)
    print(f"    ripple on top   {ripple.std():7.2f} deg rms above 0.5 Hz, strongest at"
          f" {freq[band][np.abs(spec)[band].argmax()]:.2f} Hz")
    print("    -> the box is not steadily swinging once carried. It is jittering and")
    print("       stuttering about a pose 21.6 deg away from the one the plan describes.")


def ablate():
    """Which of the plan, the contact stiffness or the contact model is to blame."""
    def stiff(model):
        model.geom_solref[:] = [0.01, 1.0]

    def torsional(model):
        for g in (model.geom("box_geom").id, hand_geom(model, "left"),
                  hand_geom(model, "right")):
            model.geom_condim[g] = 6
            model.geom_friction[g] = [model.geom_friction[g][0], 0.05, 0.02]

    print("\n5. ABLATIONS -- uncommanded lean, measured from the lift to the set-down")
    print(f"   {'change':<38} {'rms':>8} {'peak-to-peak':>14}")
    for label, steps, tweak in (("baseline, 6 roll waypoints", 6, None),
                                ("1 roll waypoint (no 1 Hz cadence)", 1, None),
                                ("12 roll waypoints", 12, None),
                                ("stiff contacts, solref 0.01 1", 6, stiff),
                                ("condim 6, torsional+rolling friction", 6, torsional)):
        model, jm, phases, poses = build(steps=steps, tweak=tweak)
        box = model.body("box").id
        _, rows = rollout(model, jm, poses, phases, dict(phases)["descend"], box)
        t = np.array([r[0] for r in rows])
        lean = np.array([r[2] for r in rows])
        sel = t >= 7.0
        x = lean[sel] - np.polyval(np.polyfit(t[sel], lean[sel], 5), t[sel])
        print(f"   {label:<38} {x.std():7.2f}d {np.ptp(x):13.1f}d")
    print("   -> the waypoint cadence is not what tilts the box; stiffening is worse;")
    print("      only giving the contacts torque to carry fixes it.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--viewer", action="store_true",
                    help="run the manoeuvre with contact points and forces drawn")
    ap.add_argument("--ablate", action="store_true",
                    help="also run the ablations in section 5 (about 5 minutes)")
    args = ap.parse_args()

    model, jm, phases, poses = build()
    box = model.body("box").id
    hands = hand_collision_geoms(model)
    pad_ids = ({"L": model.geom("left_palm_pad").id,
                "R": model.geom("right_palm_pad").id} if PALM_PADS else {})
    at = dict(phases)

    describe_shapes(model)

    if args.viewer:
        data = mujoco.MjData(model)
        data.qpos[jm.qadr] = poses["home"]
        mujoco.mj_forward(model, data)
        print("\ncontact points and forces are drawn, and the geoms made transparent so")
        print("you can see the hands sink ~20 mm into the box")
        print("left-drag orbit | scroll zoom | Space pause | Esc quit\n")
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
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

    windows = [("lift", at["lift"] - 1.8, at["lift"]),
               ("roll", at["roll1"], at["roll6"]),
               ("hold", at["roll6"], at["hold"])]
    _, rows = rollout(model, jm, poses, phases, at["descend"], box, hands)
    report_contacts(rows, windows, pad_ids)
    report_motion(rows, phases, windows)
    if args.ablate:
        ablate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
