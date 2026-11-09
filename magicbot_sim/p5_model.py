"""
The MuJoCo robot re-dimensioned to OUR robot: the P5 v1.0 kinematics, with rubber hands.

    python p5_model.py        # self-check: every link vs the P5 URDF, printed

WHY. magicbot-gen1_description (MAGICBOT.xml) is the vendor's generic, public Gen1
description. The robot we have publishes its own URDF, p5_humanoid_v1.0.urdf, which the
magiclab team verified against passive TF captures of the real robot on 2026-09-22
(magiclab/configs/robot/magicbot_p5_joint_name_map_tf_verified_2026-09-22.json). They
differ: the waist joint sits 21 mm lower on the pelvis, the hips are 2 cm closer
together, the forearm is 2 mm shorter with a 2.6 deg tilt at the wrist, every joint
limit is different (wrist pitch most: +/-18 deg in Gen1, -49..+69 deg on P5), and the
real hands are fixed rubber rakes, not Gen1's slab.

HOW. The P5 STL meshes are not available (magiclab's own P5 review uses proxies for the
same reason), so this keeps the Gen1 meshes for looks and collision and moves every body
onto the P5 kinematics: for each joint in the P5 URDF, the matching Gen1 body gets the
URDF joint origin as its pos/quat, the joint gets the URDF axis and limits, and the body
gets the URDF mass and inertia. The result has the P5's joint frames exactly -- p5_model
checks that with an independent URDF FK at random poses -- and Gen1-shaped meshes riding
on them, a few mm off the real shells.

HANDS. The Gen1 hand mesh is hidden and switched off, and each wrist_pitch_link (= P5
link_la7/ra7) gets the rubber-hand slab of magiclab sim/p5_rubber_hand_proxy.py, built
from the user's measurements of the right hand: 100 mm wide (link x), 30 mm thick (link
y), 210 mm long hanging down link -z from the wrist centre, inner face on link -y (left)
/ +y (right). Like there, it is UNCALIBRATED: where the rubber contact surface sits
relative to the wrist was not measured, and the slab is centred on the wrist axis.
The slab is added first among collidable geoms on that body, so pick_box.hand_geom()
returns it and every IK target in the pick scripts aims at the slab centre.
"""

import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

import pick_box

HERE = Path(__file__).resolve().parent
P5_URDF = HERE / "p5_reference" / "p5_humanoid_v1.0.urdf"
# the hash magiclab's planner checks (planning/README_P5_BIMANUAL_GRASP.md); LF line endings
P5_URDF_SHA256 = "065fdda596bfd01ec9074e8abb8375721f7744315cf11c44dc682b87effbffc3"

# rubber hand slab, magiclab sim/p5_rubber_hand_proxy.py MEASUREMENTS (right hand; left
# stated to be the mirror image)
HAND_WIDTH = 0.100        # link x
HAND_THICK = 0.030        # link y  -> the inner face is HAND_THICK / 2 from the slab centre
HAND_LENGTH = 0.210       # link -z, from the wrist centre (their assumption A2)
HAND_INNER_SIGN = {"left": -1.0, "right": +1.0}   # inner face normal along link y
HAND_RGBA = {"left": [0.10, 0.60, 0.95, 1.0], "right": [0.95, 0.55, 0.10, 1.0]}
RUBBER_FRICTION = 1.2

# Gen1 body <- P5 link (the joint into that link carries the origin, axis and limits)
_ARM = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow_roll", "wrist_yaw",
        "wrist_roll", "wrist_pitch")
_LEG = ("hip_roll", "hip_yaw", "hip_pitch", "knee_pitch", "ankle_pitch", "ankle_roll")
BODY_MAP = {"waist_roll_link": "link_wr", "waist_yaw_link": "link_wy",
            "head_yaw_link": "link_hy", "head_pitch_link": "link_hp"}
for _i, _n in enumerate(_ARM, 1):
    BODY_MAP[f"left_{_n}_link"] = f"link_la{_i}"
    BODY_MAP[f"right_{_n}_link"] = f"link_ra{_i}"
for _n in _LEG:
    BODY_MAP[f"left_{_n}_link"] = f"LINK_{_n.upper()}_L"
    BODY_MAP[f"right_{_n}_link"] = f"LINK_{_n.upper()}_R"


def _rpy_mat(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def _quat(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).flatten())
    return q


def _vec(text, n=3):
    return np.array([float(v) for v in (text or " ".join(["0"] * n)).split()])


def load_urdf(path=P5_URDF):
    """{child link: joint record} and {link: inertial record} from the P5 URDF."""
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if digest != P5_URDF_SHA256:
        raise RuntimeError(f"{path} is not the verified P5 v1.0 URDF (sha256 {digest})")
    root = ET.parse(path).getroot()
    joints, inertials = {}, {}
    for j in root.findall("joint"):
        o, a, lim = j.find("origin"), j.find("axis"), j.find("limit")
        joints[j.find("child").get("link")] = {
            "name": j.get("name"), "type": j.get("type"),
            "xyz": _vec(o.get("xyz") if o is not None else None),
            "rpy": _vec(o.get("rpy") if o is not None else None),
            "axis": _vec(a.get("xyz")) if a is not None else None,
            "range": (float(lim.get("lower")), float(lim.get("upper")))
            if lim is not None and j.get("type") == "revolute" else None}
    for link in root.findall("link"):
        i = link.find("inertial")
        if i is None:
            continue
        o, m, t = i.find("origin"), i.find("mass"), i.find("inertia")
        inertials[link.get("name")] = {
            "xyz": _vec(o.get("xyz") if o is not None else None),
            "rpy": _vec(o.get("rpy") if o is not None else None),
            "mass": float(m.get("value")),
            "I": np.array([[float(t.get("ixx")), float(t.get("ixy")), float(t.get("ixz"))],
                           [float(t.get("ixy")), float(t.get("iyy")), float(t.get("iyz"))],
                           [float(t.get("ixz")), float(t.get("iyz")), float(t.get("izz"))]])}
    return joints, inertials


def p5_spec_hook(spec):
    """Move every mapped Gen1 body onto the P5 kinematics and fit the rubber hands."""
    joints, inertials = load_urdf()
    for gen1, link in BODY_MAP.items():
        body = spec.body(gen1)
        rec = joints[link]
        body.pos = rec["xyz"].tolist()
        body.quat = _quat(_rpy_mat(*rec["rpy"])).tolist()
        jnt = body.joints[0]
        jnt.axis = rec["axis"].tolist()
        jnt.range = list(rec["range"])
        inert = inertials.get(link)
        if inert is not None and inert["mass"] > 0:
            # principal axes of the URDF inertia, rotated into the body frame
            R0 = _rpy_mat(*inert["rpy"])
            vals, vecs = np.linalg.eigh(R0 @ inert["I"] @ R0.T)
            if np.linalg.det(vecs) < 0:
                vecs[:, 0] *= -1
            body.mass = inert["mass"]
            body.ipos = inert["xyz"].tolist()
            body.iquat = _quat(vecs).tolist()
            body.inertia = np.maximum(vals, 1e-7).tolist()
            body.explicitinertial = True

    pel = inertials.get("pelvis")
    if pel is not None:
        body = spec.body("pelvis")
        vals, vecs = np.linalg.eigh(pel["I"])
        if np.linalg.det(vecs) < 0:
            vecs[:, 0] *= -1
        body.mass, body.ipos = pel["mass"], pel["xyz"].tolist()
        body.iquat, body.inertia = _quat(vecs).tolist(), np.maximum(vals, 1e-7).tolist()
        body.explicitinertial = True

    for side in ("left", "right"):
        hand = spec.body(f"{side}_wrist_pitch_link")
        old = list(hand.geoms)
        for g in old:                       # Gen1's hand: no collision, not drawn
            g.contype, g.conaffinity = 0, 0
            g.rgba = [0, 0, 0, 0]
        slab = hand.add_geom()
        slab.name = f"{side}_rubber_hand"
        slab.type = mujoco.mjtGeom.mjGEOM_BOX
        slab.size = [HAND_WIDTH / 2, HAND_THICK / 2, HAND_LENGTH / 2]
        slab.pos = [0.0, 0.0, -HAND_LENGTH / 2]
        slab.rgba = HAND_RGBA[side]
        slab.friction = [RUBBER_FRICTION, 0.02, 0.001]
        slab.density = 0.0
        # same pairing as the Gen1 hand it replaces: collides with the box and table
        slab.contype, slab.conaffinity = 0, 1
        slab.group = 0


def build_p5_model(box_half=None, box_y=pick_box.BOX_Y) -> mujoco.MjModel:
    """pick_box.build_model, on the P5 robot."""
    return pick_box.build_model(palm_pads=False, box_half=box_half, box_y=box_y,
                                spec_hook=p5_spec_hook)


# ---------------------------------------------------------------- self-check

def urdf_fk(joints, link, q):
    """Pose of `link` in the pelvis frame, straight from the URDF records."""
    chain = []
    while link in joints:
        chain.append(joints[link])
        link = _parent_of(link)
    T = np.eye(4)
    for rec in reversed(chain):
        step = np.eye(4)
        R = _rpy_mat(*rec["rpy"])
        if rec["type"] == "revolute":
            ax = rec["axis"] / np.linalg.norm(rec["axis"])
            K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
            a = q.get(rec["name"], 0.0)
            R = R @ (np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K)
        step[:3, :3], step[:3, 3] = R, rec["xyz"]
        T = T @ step
    return T


_PARENTS = {}


def _parent_of(link):
    if not _PARENTS:
        root = ET.parse(P5_URDF).getroot()
        for j in root.findall("joint"):
            _PARENTS[j.find("child").get("link")] = j.find("parent").get("link")
    return _PARENTS.get(link)


def self_check(samples=20, seed=0):
    """Max position / angle error of every mapped body vs the URDF over random poses."""
    joints, _ = load_urdf()
    model = build_p5_model()
    data = mujoco.MjData(model)
    rng = np.random.default_rng(seed)
    pel = model.body("pelvis").id
    worst_p, worst_a = 0.0, 0.0
    for _ in range(samples):
        q = {}
        for gen1, link in BODY_MAP.items():
            rec = joints[link]
            jid = model.body(gen1).jntadr[0]
            v = rng.uniform(*rec["range"])
            data.qpos[model.jnt_qposadr[jid]] = v
            q[rec["name"]] = v
        mujoco.mj_kinematics(model, data)
        Rp = data.xmat[pel].reshape(3, 3)
        for gen1, link in BODY_MAP.items():
            b = model.body(gen1).id
            p_sim = Rp.T @ (data.xpos[b] - data.xpos[pel])
            R_sim = Rp.T @ data.xmat[b].reshape(3, 3)
            T = urdf_fk(joints, link, q)
            worst_p = max(worst_p, np.linalg.norm(p_sim - T[:3, 3]))
            c = np.clip((np.trace(R_sim.T @ T[:3, :3]) - 1) / 2, -1, 1)
            worst_a = max(worst_a, np.degrees(np.arccos(c)))
    return worst_p, worst_a, model


def main() -> int:
    p, a, model = self_check()
    print(f"P5 URDF verified (sha256 {P5_URDF_SHA256[:12]}...)")
    print(f"{len(BODY_MAP)} bodies re-dimensioned; vs URDF FK at 20 random poses: "
          f"worst position {p * 1000:.4f} mm, worst angle {a:.5f} deg")
    print(f"total robot mass {sum(model.body_mass[1:model.body('table').id]):.1f} kg")
    for side in ("left", "right"):
        g = pick_box.hand_geom(model, side)
        print(f"  {side} IK hand geom: {model.geom(g).name}  size {np.round(model.geom_size[g], 3)}")
    return 0 if p < 1e-6 and a < 1e-4 else 1


if __name__ == "__main__":
    raise SystemExit(main())
