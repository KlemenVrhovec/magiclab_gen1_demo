# Why the carried box will not sit still

Findings on the grasp used by `rotate_around_x_axis_v2.py`. Every number here is printed
by [`grasp_contact_report.py`](grasp_contact_report.py); nothing is assumed.

```
python grasp_contact_report.py             # the report (one planning pass, ~1 min)
python grasp_contact_report.py --viewer    # watch it with contact points drawn
python grasp_contact_report.py --ablate    # the ablation table below (~5 min)
```

## 1. The hand is not a square, and not a box at all

It is the whole `wrist_pitch_link` STL — 8214 verts, 16466 faces — collided as its
**611-vertex convex hull**. Its bounding box is 45 x 140 x 248 mm, thin axis local X.

`HAND_HALF_THICK = 0.0228` in `pick_box.py` is misleading as a name: MuJoCo fills
`geom_size` for a mesh from its bounding box, and no collision uses it as a shape. It is
half the hull's bounding box on the thin axis, not a plate thickness.

The gripping side is nearly flat, but only over a small part of the box face:

| hull verts within | count | spanning |
|---|---|---|
| 0.5 mm of the extreme plane | 10 | 21 x 45 mm |
| 1.0 mm | 22 | 38 x 60 mm |
| 2.0 mm | 39 | 51 x 70 mm |
| 5.0 mm | 69 | 63 x 117 mm |

So the largest patch that could ever bear is roughly 38 x 60 mm, against a
90 x 200 mm box face.

## 2. The box is a real box

One primitive `mjGEOM_BOX`, 90 x 90 x 200 mm, 0.3 kg, freejoint, sliding friction 2.0.

Consequence: **box vs hand is box-vs-mesh**, MuJoCo's general convex collider — *not* the
special-cased box-box collider that box vs table uses. That is exactly why the table
yields a steady 4-5 points and the hands do not.

## 3. Points of contact: 1 to 5 per hand, and the set is rebuilt constantly

The two hands behave nothing like each other.

| | contacts | spread across the box face |
|---|---|---|
| left, lift | **1 point for all 900 steps** | never more than one |
| left, roll | 1pt x1261, 2pt x952, 3pt x283, 4pt x4 | 13 x 38 mm when >1 (max 43 x 78) |
| left, hold | 1pt x526, 2pt x460, 3pt x14 | ~1 mm apart when >1 — a point, not a patch |
| right, lift | 1pt x503, 2pt x322, 3pt x61, 4pt x14 | 35 x 3 mm when >1 |
| right, roll | 1pt x339, 2pt x850, 3pt x878, 4pt x399, 5pt x34 | 42 x 47 mm (max 43 x 79) |
| right, hold | 2pt x78, 3pt x331, 4pt x435, 5pt x156 | 43 x 12 mm |

- At least one hand is down to a **single point on 69%** of steps; both at once on 17%.
- The count changes on 17-41% of steps (65% on the right during the lift).
- `condim=3` contacts carry force but **no torque**, so a hand reduced to one point
  resists no rotation about that point at all.

The pinch is also deep. The 30 mm `SQUEEZE` bias meets contacts of stiffness 500 N/m
(`solref="-500 -800"`, from the model's own `<default>` in `MAGICBOT.xml`), so the hulls
end up **~20 mm inside** the box — nearly half its 90 mm width, and plainly visible in
the viewer as hands sunk into it. Mean grip 109-118 N.

## 4. What you see is three separate things, not one oscillation

### 4a. A jitter, from the contact set being rebuilt

Sampled at 60 fps — what the viewer actually shows you:

| window | deg turned per frame | wx reverses between |
|---|---|---|
| lift | mean 0.47, max 3.84 | 39% of frames |
| roll | mean 0.18, max 0.75 | 39% of frames |
| hold | mean 0.04, max 0.15 | 49% of frames |

The box reverses its direction of rotation about X on roughly half of consecutive frames.
That reads as a shimmer, not a swing. It decays once nothing is moving.

### 4b. A 1 Hz stutter through the roll — the trajectory, not the physics

`rotate_around_x_axis.timeline` lays down 6 roll waypoints over 6 s, and `target_at`
cosine-eases between them, which brings the hands to a **dead stop at every waypoint**.
Roll advanced per quarter-second:

```
 9.00- 9.25  +3.24 #########################
 9.25- 9.50  +0.34 ##
 9.50- 9.75  +0.07
 9.75-10.00  +0.06
10.00-10.25  -0.08                      <- waypoint
10.25-10.50  +0.25 #
10.50-10.75  +0.57 ####
10.75-11.00  +0.46 ###
11.00-11.25  +0.07                      <- waypoint
11.25-11.50  +0.69 #####
11.50-11.75  +1.61 ############
11.75-12.00  +1.33 ##########
12.00-12.25  +0.42 ###                  <- waypoint
```

Six visible pulses, with a small backward slip of -0.08 deg at the first boundary.

### 4c. A one-off 21 deg snap, then a permanent lean

The manoeuvre commands roll about Y only, so any tilt in the Y-Z plane is pure grasp
artefact (it is exactly zero for a pure Y roll). Measured:

```
dragged to      -12.24 deg while the hands close against the table (t=6.54)
snaps through    -8.01 -> +13.11 deg in the 368 ms it comes free of the table
settles at      +21.60 deg
ripple on top     0.07 deg rms above 0.5 Hz, strongest at 0.95 Hz
```

The +21.6 deg is `PREROLL_DEG = -20` from `pick_box_v2.py`, mirrored — the box has been
dragged into line with the pre-rolled palms.

**So the box is not steadily swinging once carried.** It is jittering (4a) and stuttering
(4b) about a pose 21.6 deg away from the one the plan describes (4c).

## 5. What is to blame — ablated

Uncommanded lean, measured from the lift to the set-down:

| change | rms | peak-to-peak | verdict |
|---|---|---|---|
| baseline, 6 roll waypoints | 1.25 deg | 15.4 deg | |
| 1 roll waypoint instead of 6 | 1.30 | 15.7 | not the waypoints |
| 12 roll waypoints instead of 6 | 1.23 | 15.3 | not the waypoints |
| stiff contacts (`solref 0.01 1`) | 4.37 | 27.3 | **worse** |
| `condim 6`, torsional+rolling friction | 0.33 | 1.5 | **12x better** |

- The waypoint cadence does not cause the tilt. It only paces the stutter in 4b.
- Stiffening the contacts makes it **worse**: less penetration puts fewer hull vertices
  inside the box, leaving a smaller and more point-like contact set.
- The only change that fixes it is giving the contacts torque to transmit, which is the
  direct test of the diagnosis in section 3: **the grasp is points, not faces.**

## Open decisions

- `condim=6` with torsional and rolling friction on the box and hand geoms is a one-line
  fix for the wobble. Not applied — it changes the contact model for the whole scene.
- The 21.6 deg lean is `PREROLL_DEG` doing exactly what it is written to do. It is not a
  bug, but nothing downstream of the grasp accounts for the box being tilted by it.
