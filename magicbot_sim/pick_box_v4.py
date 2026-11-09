"""
Bimanual grasp, version 4: the real box, 20 x 27 x 27 cm, gripped on its 27 x 27 faces.

    python pick_box_v4.py               # interactive viewer
    python pick_box_v4.py --headless    # no window, reports the grasp and body clearance

Same motion as pick_box_v3.py -- hands lifted forward beside the body, fingers turned to
point away from the robot, palms closing on the box -- planned for a bigger box.

THE BOX. 20 cm across (world X), 27 cm deep (Y), 27 cm tall (Z). The palms squeeze along
X, so they land on the two large 27 x 27 faces and the grip spans 20 cm instead of V3's
9 cm.

WHERE IT SITS. 27 cm deep does not fit at V3's y = -0.36: it would hang 6.5 cm off the
front edge of the table (y = -0.29). At BIG_BOX_Y = -0.44 it spans y = -0.575..-0.305,
fully on the table, and the palm -- 25 cm long now that it lies fingers-forward --
covers almost the whole depth of the face.

HOW HIGH. The wider grip makes fingers-forward easier to reach than in V3: measured at
this box position, the fingers end 14 deg off forward at z = 1.26, 6 deg at 1.30, 0 at
1.33. The hand's vertical extent with the fingers forward is 54 mm below the aimed point
and 86 mm above, and the box top is at 1.37. GRASP_Z = 1.32 puts the palm on the face from
z = 1.27 to the top edge, ~10 cm of contact, with the fingers within a few degrees of
forward.

The box mass is still pick_box's 0.3 kg; nothing is lifted yet, so it does not matter.
"""

import argparse

import numpy as np

from pick_box import HAND_HALF_THICK, SQUEEZE, JointMap, build_model
from pick_box_v3 import FINGER_ROLL_DEG, plan, simulate

BIG_BOX_HALF = (0.10, 0.135, 0.135)   # half-extents (x, y, z): 20 x 27 x 27 cm
BIG_BOX_Y = -0.44                     # box centre; see WHERE IT SITS
GRASP_Z = 1.32                        # hand geom centre height at the grip; see HOW HIGH

TOUCH_X = BIG_BOX_HALF[0] + HAND_HALF_THICK   # 0.123: palms just kissing the box
OPEN_X = TOUCH_X + 0.075                      # approach from outside, as V1-V3
GRIP_X = TOUCH_X - 0.018                      # commanded inside the box -> squeeze

# Where the left hand's path passes on the way in (right mirrors it). V3's point, which
# measured clear of both the body and this box.
VIA = np.array([0.30, -0.15, 1.35])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--headless", action="store_true", help="no window; report the grasp")
    ap.add_argument("--duration", type=float, default=4.0)
    ap.add_argument("--grasp-z", type=float, default=GRASP_Z,
                    help="height of the hand centres at the grip")
    ap.add_argument("--squeeze", type=float, default=SQUEEZE,
                    help="metres of inward over-travel commanded into the box (grip force)")
    args = ap.parse_args()

    model = build_model(box_half=BIG_BOX_HALF, box_y=BIG_BOX_Y)
    jm = JointMap(model)
    print("planning grasp:")
    poses = plan(model, jm, squeeze=args.squeeze, roll_deg=FINGER_ROLL_DEG,
                 grasp_z=args.grasp_z, via=VIA, box_y=BIG_BOX_Y,
                 open_x=OPEN_X, grip_x=GRIP_X)
    return simulate(model, jm, poses, args.headless, args.duration)


if __name__ == "__main__":
    raise SystemExit(main())
