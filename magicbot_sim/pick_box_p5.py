"""
Bimanual grasp on OUR robot: pick_box_v4's motion, on the P5 v1.0 kinematics with rubber hands.

    python pick_box_p5.py               # interactive viewer
    python pick_box_p5.py --headless    # no window, reports the grasp and body clearance

Same task as pick_box_v4.py -- a 20 x 27 x 27 cm box gripped on its 27 x 27 faces,
hands lifted forward beside the body, fingers pointing away from the robot, no lift --
but on p5_model.build_p5_model(): every joint where the robot's own URDF puts it, the
P5 joint limits, and magiclab's measured rubber-hand slab in place of Gen1's hand.

THE HAND. The rubber hand is a 100 x 30 x 210 mm slab hanging from the wrist centre,
inner face on link -y (left) / +y (right), fingers along link -z. Fingers forward with
the inner faces on the box gives the SAME attitude for both hands (GRASP_ATT): link x ->
world up, link y -> world +X, link z -> world +Y (towards the robot), so the fingers
(-z) point away from it and each inner face points at the box.

HOW HIGH. On the P5 the wrist pitch has -49..+69 deg instead of Gen1's +/-18, and that
is what limited Gen1: measured here, the grip is reached exactly (0.0 mm, 0.0 deg) at
hand-centre z = 1.25 and 1.30, and misses by ~30 mm at 1.20. The slab's width is
vertical when the fingers point forward, so the palm covers z +/- 50 mm around the
aimed point; GRASP_Z = 1.25 puts it at 1.20..1.30 on the 1.10..1.37 face.

WHERE ALONG THE BOX. The slab hangs 210 mm forward of the wrist, and the forearm and
wrist housings are wider than the 30 mm slab. Aiming the hand centre at the middle of the
box's depth (V4's choice) puts the wrist beside the box, and then it is the FOREARMS that
squeeze it -- measured: left wrist_yaw 25 N + wrist_roll 30 N, right 14 + 40 N, the
rubber hands not touching at all. HAND_Y keeps the wrist 40 mm in front of the box's
near face (wrist y = -0.265, face at -0.305): only the rubber touches, symmetric, along
~170 mm of the 270 mm face. 20 mm further in and the right forearm already catches the
box's near edge.

HOW HARD. An empty cardboard box needs little: 2 * mu * N >= m * g, so ~5 N per hand for
0.5 kg with rubber on cardboard at mu ~0.5, ~10 N with a 2x margin. The hands are aimed
GRIP_DEPTH into the box and nothing more (no add_squeeze over-travel). Measured per hand
with the rubber contact: 0 mm -> 2 N, 4 -> 7, 8 -> 11, 12 -> 15 N, ~1.1 N/mm. On the real
robot that rate depends on the vendor joint gains, not on this simulation's kp = 300.

THE CONTACT SURFACE IS NOT CALIBRATED. magiclab's proxy centres the slab on the wrist
axis (their assumption A5); the real inner rubber surface may sit a few mm either way.
TOUCH_X assumes the proxy, so the squeeze depth on the real hand is uncertain by that.
"""

import argparse

import numpy as np

from p5_model import HAND_THICK, build_p5_model
from pick_box import JointMap
from pick_box_v3 import plan, simulate
from pick_box_v4 import BIG_BOX_HALF, BIG_BOX_Y

GRASP_Z = 1.25

# World attitude of both hand slabs at the grip, as columns (link x, link y, link z).
# Left inner face is link -y, so link y = +X points it at the box from the left (+X)
# side; right inner face is link +y, which also points at the box (+X) from the right.
GRASP_ATT = np.array([[0.0, 1.0, 0.0],
                      [0.0, 0.0, 1.0],
                      [1.0, 0.0, 0.0]])

TOUCH_X = BIG_BOX_HALF[0] + HAND_THICK / 2    # 0.115: inner rubber faces kissing the box
OPEN_X = TOUCH_X + 0.075
GRIP_DEPTH = 0.008                            # aimed into the box -> ~11 N per hand
GRIP_X = TOUCH_X - GRIP_DEPTH
SQUEEZE = 0.0                                 # no extra over-travel; see HOW HARD

# Hand-centre y at the grip: 70 mm behind the box centre, so the wrist (105 mm behind the
# hand centre) sits 40 mm in front of the box's near face. See WHERE ALONG THE BOX.
HAND_Y = BIG_BOX_Y + 0.07                     # -0.37

VIA = np.array([0.30, -0.15, 1.35])           # as V3/V4; measured 100 mm over the table


def p5_scene():
    return build_p5_model(box_half=BIG_BOX_HALF, box_y=BIG_BOX_Y)


def make_plan(model, jm, verbose=True, squeeze=SQUEEZE, grasp_z=GRASP_Z, grip_x=GRIP_X):
    # plan()'s box_y is only where the hand centres are aimed along y
    return plan(model, jm, verbose=verbose, squeeze=squeeze, grasp_z=grasp_z, via=VIA,
                box_y=HAND_Y, open_x=OPEN_X, grip_x=grip_x, grasp_att=GRASP_ATT)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--headless", action="store_true", help="no window; report the grasp")
    ap.add_argument("--duration", type=float, default=4.0)
    ap.add_argument("--grasp-z", type=float, default=GRASP_Z,
                    help="height of the hand centres at the grip")
    ap.add_argument("--grip-depth", type=float, default=GRIP_DEPTH,
                    help="metres the hands are aimed into the box (grip force, ~1.1 N/mm)")
    ap.add_argument("--squeeze", type=float, default=SQUEEZE,
                    help="extra metres of inward over-travel on top of --grip-depth")
    args = ap.parse_args()

    model = p5_scene()
    jm = JointMap(model)
    print("planning grasp (P5 v1.0 kinematics, rubber hands):")
    poses = make_plan(model, jm, squeeze=args.squeeze, grasp_z=args.grasp_z,
                      grip_x=TOUCH_X - args.grip_depth)
    return simulate(model, jm, poses, args.headless, args.duration, builder=build_p5_model)


if __name__ == "__main__":
    raise SystemExit(main())
