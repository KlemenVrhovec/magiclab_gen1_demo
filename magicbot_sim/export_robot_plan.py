"""
Export pick_box_v4's grasp as a joint-keyframe plan for the real MagicBot (P5), for REVIEW.

    python export_robot_plan.py                       # writes into the magiclab repo
    python export_robot_plan.py --out-dir some/dir    # anywhere else

Writes two files: pick_box_v4_keyframes.json (the plan) and README.md (what it is, the
numbers, and what has to happen before any of it may move a robot). Nothing here talks
to a robot; the JSON is marked not_a_robot_command / hardware_execution_allowed: false.

WHAT IS EXPORTED. pick_box_v4 drives the arms along a dense IK path (80 samples) that is
replayed at 500 Hz in 2.5 s. A robot cannot use that as-is: the real arm controller is
capped at 0.15 rad/s and 0.30 rad/s^2 per joint (robot/p5_motion_backend.py), the start
pose is the robot's live pose rather than the simulation's, and the vendor behaviour of a
streamed upper-body trajectory is not established (docs/p5_motion_incident_20260929.md).
So the dense path is reduced to a few KEYFRAMES -- added greedily wherever a straight
joint-space blend between neighbours would leave the planned hand path by more than
MAX_HAND_DEVIATION_M -- and every segment is retimed as a rest-to-rest quintic inside the
robot's limits, which is the shape robot/p5_phase_executor.py streams per phase.

JOINTS. The simulation's 7 arm joints per side are the robot's joint_la1..7 / ra1..7 in
the same order, same axes and same link offsets to ~2 mm (checked against
external/p5_robot_reference/p5_humanoid_v1.0.urdf). The one real difference is above the
arms: the robot's waist joint sits 21 mm lower on the pelvis (z 0.062 vs 0.083), so the
same joint angles put the hands ~2 cm lower. The exporter does FK on the robot's own URDF
for every keyframe and reports both, and shifts the recommended box position to match.

Every segment of the keyframe path is checked in the simulation: forearm/hand vs body
(pick_box_v3.body_clearance), hand vs tabletop and hand vs box.
"""

import argparse
import hashlib
import json
import subprocess
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import mujoco
import numpy as np

import pick_box_v3 as v3
import pick_box_v4 as v4
from pick_box import TABLE_HALF, TABLE_TOP, TABLE_Y, Arms, JointMap, build_model
from pick_box_v2 import palm_clearance

HERE = Path(__file__).resolve().parent
MAGICLAB = Path(r"C:\Work\13 Magiclab repo\magiclab")
DEFAULT_OUT = MAGICLAB / "sim" / "exports" / "pick_box_v4_fingers_forward"
P5_URDF = MAGICLAB / "external" / "p5_robot_reference" / "p5_humanoid_v1.0.urdf"

# robot/p5_motion_backend.py: MAX_JOINT_VELOCITY_RAD_S, MAX_JOINT_ACCELERATION_RAD_S2
MAX_VEL = 0.15
MAX_ACC = 0.30
MAX_HAND_DEVIATION_M = 0.015   # keyframe blend vs planned hand path
MIN_BODY_CLEARANCE_M = 0.010   # forearm/hand vs body, on the blended path
SUBSTEPS = 25                  # checks per segment

P5_ARM = [f"joint_la{i}" for i in range(1, 8)] + [f"joint_ra{i}" for i in range(1, 8)]
JSON_NAME = "pick_box_v4_keyframes.json"
FORMAT = "magicbot_sim_joint_keyframe_plan_v1"


# ---------------------------------------------------------------- robot URDF FK

def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def _axis_angle(axis, a):
    k = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


class UrdfFK:
    """Forward kinematics on a URDF's joint tree, in the root link's (pelvis) frame."""

    def __init__(self, path):
        root = ET.parse(path).getroot()
        self.by_child, self.limits = {}, {}
        for j in root.findall("joint"):
            o = j.find("origin")
            xyz = np.array([float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()])
            rpy = [float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()]
            a = j.find("axis")
            axis = [float(v) for v in a.get("xyz").split()] if a is not None else [1, 0, 0]
            self.by_child[j.find("child").get("link")] = (
                j.get("name"), j.find("parent").get("link"), xyz, _rpy(*rpy), axis, j.get("type"))
            lim = j.find("limit")
            if lim is not None and j.get("type") == "revolute":
                self.limits[j.get("name")] = (float(lim.get("lower")), float(lim.get("upper")))

    def link(self, name, q):
        """4x4 pose of `name` in the root frame; joints missing from q are at 0."""
        T = np.eye(4)
        chain = []
        while name in self.by_child:
            chain.append(self.by_child[name])
            name = self.by_child[name][1]
        for jname, _, xyz, R0, axis, jtype in reversed(chain):
            step = np.eye(4)
            step[:3, :3] = R0 @ (_axis_angle(axis, q.get(jname, 0.0)) if jtype == "revolute" else np.eye(3))
            step[:3, 3] = xyz
            T = T @ step
        return T


# ---------------------------------------------------------------- simulation side

def _v4_scene():
    return build_model(box_half=v4.BIG_BOX_HALF, box_y=v4.BIG_BOX_Y)


class Sim:
    """A grasp scene plus everything needed to express it in the robot's pelvis frame.

    scene_builder() builds the scene being exported (default: pick_box_v4's); body_builder
    is the robot used for the forearm/hand-vs-body probe (pick_box_v3.body_clearance).
    """

    def __init__(self, scene_builder=_v4_scene, body_builder=build_model):
        self.scene_builder = scene_builder
        self.model = scene_builder()
        self.jm = JointMap(self.model)
        self.arms = Arms(self.model)
        self.data = mujoco.MjData(self.model)
        self.idx = [self.jm.index(n) for n in self.arms.names]
        self.body_check = v3.body_clearance(body_builder)
        self.box_check = self._box_probe()

        mujoco.mj_kinematics(self.model, self.data)
        pel = self.model.body("pelvis").id
        self.R_pel = self.data.xmat[pel].reshape(3, 3).copy()
        self.p_pel = self.data.xpos[pel].copy()
        # the hand geom centre, fixed in the wrist_pitch_link frame -- the same point is
        # then placed on the robot's link_la7 / link_ra7
        self.hand_offset = []
        for body, geom in zip(self.arms.bodies, self.arms.geoms):
            R = self.data.xmat[body].reshape(3, 3)
            self.hand_offset.append(R.T @ (self.data.geom_xpos[geom] - self.data.xpos[body]))

    def to_pelvis(self, p_world):
        return self.R_pel.T @ (np.asarray(p_world) - self.p_pel)

    def set_arms(self, arm_q):
        self.data.qpos[self.jm.qadr] = 0.0
        self.data.qpos[self.jm.qadr[self.idx]] = arm_q
        mujoco.mj_kinematics(self.model, self.data)

    def hands_world(self, arm_q):
        self.set_arms(arm_q)
        return [self.data.geom_xpos[g].copy() for g in self.arms.geoms]

    def _box_probe(self):
        """Signed distance to the box out to 50 mm, from a probe copy with a box margin.

        check(q) -> (hand distance, closest OTHER robot part's distance, that part's name).
        The second number is what catches a forearm or wrist housing reaching the box
        before the hand does.
        """
        m = self.scene_builder()
        d = mujoco.MjData(m)
        box = m.geom("box_geom").id
        table = m.geom("table_geom").id
        m.geom_margin[box] = 0.05
        hands = set(Arms(m).geoms)
        jm = JointMap(m)
        idx = [jm.index(n) for n in Arms(m).names]

        def check(arm_q):
            d.qpos[:] = m.qpos0
            d.qpos[jm.qadr[idx]] = arm_q
            mujoco.mj_forward(m, d)
            hand, other, who = np.inf, np.inf, ""
            for i in range(d.ncon):
                c = d.contact[i]
                if box not in (c.geom1, c.geom2):
                    continue
                g = c.geom2 if c.geom1 == box else c.geom1
                if g == table:
                    continue
                if g in hands:
                    hand = min(hand, c.dist)
                elif c.dist < other:
                    other, who = c.dist, m.body(m.geom_bodyid[g]).name
            return hand, other, who

        return check

    def segment_checks(self, qa, qb):
        """Worst body / table / box clearance along the straight joint blend qa -> qb.

        box is the hand's distance; self.last_other_box is (distance, part) for the
        closest robot part that is not the hand.
        """
        body = table = box = np.inf
        who = ""
        other = (np.inf, "")
        for s in np.linspace(0.0, 1.0, SUBSTEPS):
            q = qa + s * (qb - qa)
            self.set_arms(q)
            mujoco.mj_forward(self.model, self.data)
            table = min(table, palm_clearance(self.model, self.data, self.arms))
            hand_d, other_d, other_who = self.box_check(q)
            box = min(box, hand_d)
            if other_d < other[0]:
                other = (other_d, other_who)
            full = self.data.qpos.copy()
            gap, w = self.body_check(full)
            if gap < body:
                body, who = gap, w
        self.last_other_box = other
        return body, who, table, box


def p5_hands(fk, arm_q, hand_offset):
    """Hand-geom-centre equivalent on the robot (pelvis frame), via its own URDF."""
    q = dict(zip(P5_ARM, arm_q))
    out = []
    for link, off in (("link_la7", hand_offset[0]), ("link_ra7", hand_offset[1])):
        T = fk.link(link, q)
        out.append(T[:3, :3] @ off + T[:3, 3])
    return out


# ---------------------------------------------------------------- keyframes and timing

def pick_keyframes(sim, path):
    """Greedy: add the dense-path sample the current blend misses worst, until it fits."""
    dense_hands = [sim.hands_world(q[sim.idx]) for q in path]
    keys = [0, len(path) - 1]
    while True:
        worst, at = 0.0, None
        for a, b in zip(keys, keys[1:]):
            for k in range(a + 1, b):
                f = (k - a) / (b - a)
                q = path[a][sim.idx] * (1 - f) + path[b][sim.idx] * f
                h = sim.hands_world(q)
                dev = max(np.linalg.norm(h[i] - dense_hands[k][i]) for i in range(2))
                if dev > worst:
                    worst, at = dev, k
        if worst <= MAX_HAND_DEVIATION_M or at is None:
            break
        keys = sorted(keys + [at])
    # body clearance is checked on the blend too: split any segment that comes too close
    changed = True
    while changed:
        changed = False
        for a, b in zip(keys, keys[1:]):
            if b - a < 2:
                continue
            body, _, _, _ = sim.segment_checks(path[a][sim.idx], path[b][sim.idx])
            if body < MIN_BODY_CLEARANCE_M:
                keys = sorted(keys + [(a + b) // 2])
                changed = True
                break
    return keys, worst


def quintic_duration(dq):
    """Shortest rest-to-rest quintic within MAX_VEL and MAX_ACC on every joint."""
    d = float(np.max(np.abs(dq)))
    if d == 0.0:
        return 0.0
    # quintic peaks: velocity 1.875 d/T, acceleration 5.7735 d/T^2
    return max(1.875 * d / MAX_VEL, np.sqrt(5.7735 * d / MAX_ACC))


# ---------------------------------------------------------------- provenance

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_state():
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE, capture_output=True,
                              text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "."], cwd=HERE,
                               capture_output=True, text=True, check=True).stdout.strip()
        return {"commit": head, "uncommitted_changes": bool(dirty)}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "uncommitted_changes": None}


def r(x, n=4):
    return [round(float(v), n) for v in np.ravel(x)]


# ---------------------------------------------------------------- main

def build_export():
    sim = Sim()
    fk = UrdfFK(P5_URDF)
    common = dict(verbose=False, grasp_z=v4.GRASP_Z, via=v4.VIA, box_y=v4.BIG_BOX_Y,
                  open_x=v4.OPEN_X)
    print("planning (three IK solves: path, contact pose, squeeze pose) ...")
    touch = v3.plan(sim.model, sim.jm, squeeze=0.0, grip_x=v4.TOUCH_X, **common)
    squeeze = v3.plan(sim.model, sim.jm, squeeze=v4.SQUEEZE, grip_x=v4.GRIP_X, **common)
    path = touch["path"]

    print("choosing keyframes ...")
    keys, fit = pick_keyframes(sim, path)

    frames = []
    for n, k in enumerate(keys):
        name = "sim_start" if n == 0 else ("beside_box" if k == len(path) - 1 else f"reach_{n}")
        frames.append({"name": name, "q": path[k][sim.idx], "path_fraction": k / (len(path) - 1),
                       "contact": False})
    frames.append({"name": "palms_touch_box", "q": touch["close"][sim.idx], "contact": True})
    frames.append({"name": "squeeze", "q": squeeze["close"][sim.idx], "contact": True})

    # limits, FK on both models, clearances and timing
    lo = np.array([fk.limits[n][0] for n in P5_ARM])
    hi = np.array([fk.limits[n][1] for n in P5_ARM])
    rows = []
    for i, f in enumerate(frames):
        q = f["q"]
        hs = [sim.to_pelvis(h) for h in sim.hands_world(q)]
        hp = p5_hands(fk, q, sim.hand_offset)
        margin = np.minimum(q - lo, hi - q)
        row = {"name": f["name"], "q": q, "hand_sim": hs, "hand_p5": hp,
               "p5_minus_sim": [hp[j] - hs[j] for j in range(2)],
               "min_limit_margin": float(margin.min()), "tightest": P5_ARM[int(margin.argmin())],
               "contact": f["contact"], "path_fraction": f.get("path_fraction")}
        if i == 0:
            row.update(duration=None, body=None, table=None, box=None)
        else:
            prev = frames[i - 1]["q"]
            body, who, table, box = sim.segment_checks(prev, q)
            row.update(duration=quintic_duration(q - prev), body=body, body_pair=who,
                       table=table, box=box, max_joint_change=float(np.max(np.abs(q - prev))),
                       max_joint_change_joint=P5_ARM[int(np.argmax(np.abs(q - prev)))])
        rows.append(row)
    return sim, fk, rows, fit, len(path)


def scene_in_pelvis(sim, shift):
    box_c = sim.to_pelvis([0.0, v4.BIG_BOX_Y, TABLE_TOP + v4.BIG_BOX_HALF[2]])
    # box/table half-extents re-expressed along pelvis axes (pelvis x = world -Y)
    box_half = np.abs(sim.R_pel.T @ np.array(v4.BIG_BOX_HALF))
    table_c = sim.to_pelvis([0.0, TABLE_Y, TABLE_TOP / 2.0])
    table_half = np.abs(sim.R_pel.T @ np.array(TABLE_HALF))
    table_top = sim.to_pelvis([0, 0, TABLE_TOP])[2]
    near_edge = table_c[0] - table_half[0]
    return {
        "frame": "robot pelvis: +x forward, +y robot's left, +z up (metres)",
        "box": {"size_m": r(2 * box_half, 3), "center_sim_geometry": r(box_c),
                "center_for_p5": r(box_c + shift), "rotation": "axis-aligned with the pelvis",
                "mass_kg_in_sim": 0.3,
                "gripped_faces": "the two 27 x 27 cm faces normal to pelvis y"},
        "table": {"top_z_sim_geometry": round(float(table_top), 4),
                  "top_z_for_p5": round(float(table_top + shift[2]), 4),
                  "near_edge_x": round(float(near_edge), 4),
                  "center": r(table_c), "size_m": r(2 * table_half, 3)},
        "p5_shift_note": ("*_for_p5 = the simulation's position moved by the mean hand offset "
                          "(robot URDF FK minus simulation FK) at palms_touch_box, so the box "
                          "sits where the robot's hands will actually be."),
    }


def write(out_dir, sim, fk, rows, fit, n_dense):
    out_dir.mkdir(parents=True, exist_ok=True)
    touch = next(rw for rw in rows if rw["name"] == "palms_touch_box")
    shift = np.mean(touch["p5_minus_sim"], axis=0)
    scene = scene_in_pelvis(sim, shift)
    reach = [rw for rw in rows[1:] if not rw["contact"]]
    reach_t = sum(rw["duration"] for rw in reach)

    sources = ["pick_box.py", "pick_box_v2.py", "pick_box_v3.py", "pick_box_v4.py",
               "export_robot_plan.py"]
    plan = {
        "format": FORMAT,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "passive": True,
        "not_a_robot_command": True,
        "hardware_execution_allowed": False,
        "status": "offline_review_only",
        "summary": ("Bimanual box grasp from MuJoCo (pick_box_v4): hands lifted forward beside the body, "
                    "fingers turned to point forward, palms on the two 27x27 cm faces of a 20x27x27 cm box."),
        "source": {"repo": "6_MagicLab_Gen1_Simulation", **git_state(),
                   "files_sha256": {f: sha256(HERE / f) for f in sources},
                   "sim_model": "magicbot-gen1_description/mjcf/MAGICBOT.xml (pelvis pinned)"},
        "robot_model_check": {"urdf": "external/p5_robot_reference/p5_humanoid_v1.0.urdf",
                              "urdf_sha256": sha256(P5_URDF)},
        "joint_order": P5_ARM,
        "joint_map_from_sim": dict(zip(P5_ARM, sim.arms.names)),
        "units": "rad",
        "not_commanded": {"waist_and_head": "hold the live measured values (joint_wy, joint_wr, joint_hy, joint_hp)",
                          "hands": "no hand joints; the robot has fixed rubber rake hands"},
        "timing": {"model": "rest-to-rest quintic per segment (stops at every keyframe)",
                   "max_joint_velocity_rad_s": MAX_VEL, "max_joint_acceleration_rad_s2": MAX_ACC,
                   "limits_from": "robot/p5_motion_backend.py",
                   "reach_total_s": round(reach_t, 2),
                   "sim_reach_s": v3.REACH_T},
        "keyframe_fit": {"dense_path_samples": n_dense,
                         "max_hand_deviation_m": round(float(fit), 4),
                         "deviation_threshold_m": MAX_HAND_DEVIATION_M,
                         "min_body_clearance_threshold_m": MIN_BODY_CLEARANCE_M},
        "scene": scene,
        "null_values": {"min_forearm_hand_to_body_m": "clearer than the 20 mm probe range",
                        "min_hand_to_tabletop_m": "no hand was over the table in this segment",
                        "min_hand_to_box_m": "clearer than the 50 mm probe range"},
        "keyframes": [],
    }
    for rw in rows:
        kf = {"name": rw["name"],
              "q": dict(zip(P5_ARM, r(rw["q"], 5))),
              "q_vector": r(rw["q"], 5),
              "segment_duration_s": None if rw["duration"] is None else round(rw["duration"], 2),
              "contact_with_box": rw["contact"],
              "use": ("reference only: the robot starts from its LIVE pose" if rw["name"] == "sim_start" else
                      "NOT for a first test: deliberate position error into the box" if rw["name"] == "squeeze" else
                      "contact phase" if rw["contact"] else "reach phase"),
              "hand_center_pelvis_m": {"sim": {"left": r(rw["hand_sim"][0]), "right": r(rw["hand_sim"][1])},
                                       "p5_urdf": {"left": r(rw["hand_p5"][0]), "right": r(rw["hand_p5"][1])}},
              "p5_limit_margin_rad": round(rw["min_limit_margin"], 3),
              "p5_tightest_joint": rw["tightest"]}
        if rw["duration"] is not None:
            kf["segment_checks_sim"] = {
                "min_forearm_hand_to_body_m": None if not np.isfinite(rw["body"]) else round(rw["body"], 4),
                "closest_body_pair": rw.get("body_pair") or None,
                "min_hand_to_tabletop_m": None if not np.isfinite(rw["table"]) else round(rw["table"], 4),
                "min_hand_to_box_m": None if not np.isfinite(rw["box"]) else round(rw["box"], 4),
                "max_joint_change_rad": round(rw["max_joint_change"], 3),
                "max_joint_change_joint": rw["max_joint_change_joint"]}
        plan["keyframes"].append(kf)
    (out_dir / JSON_NAME).write_text(json.dumps(plan, indent=2), encoding="utf-8")
    (out_dir / "README.md").write_text(readme(plan, rows, shift), encoding="utf-8")
    return plan


def _mm(x, none="> 20"):
    return none if x is None or not np.isfinite(x) else f"{x * 1000:.0f}"


def readme(plan, rows, shift):
    sc = plan["scene"]
    kf_rows = []
    for rw, kf in zip(rows, plan["keyframes"]):
        dz = np.mean([d[2] for d in rw["p5_minus_sim"]]) * 1000
        if rw["duration"] is None:
            kf_rows.append(f"| `{rw['name']}` | - | - | - | - | - | {dz:+.0f} | {rw['min_limit_margin']:.2f} |")
        else:
            box = f"contact ({_mm(rw['box'], '> 50')})" if rw["contact"] else _mm(rw["box"], "> 50")
            kf_rows.append(f"| `{rw['name']}` | {rw['duration']:.1f} | {rw['max_joint_change']:.2f} | "
                           f"{_mm(rw['body'])} | {_mm(rw['table'], 'not over table')} | {box} | {dz:+.0f} | "
                           f"{rw['min_limit_margin']:.2f} |")
    t = plan["timing"]
    touch = next(rw for rw in rows if rw["name"] == "palms_touch_box")
    spread = np.mean([abs(touch["hand_p5"][j][1]) - abs(touch["hand_sim"][j][1]) for j in range(2)])
    return f"""# Simulated box grasp (pick_box_v4) — joint keyframe export

**Status: offline review only. Not a robot command.** The JSON is marked
`not_a_robot_command: true` and `hardware_execution_allowed: false`. Nothing in it has been
sent to or tried on a robot, and no robot code reads it.

Generated {plan["created_utc"]} by `magicbot_sim/export_robot_plan.py` in the
`6_MagicLab_Gen1_Simulation` repo (commit `{plan["source"]["commit"]}`, uncommitted changes:
{plan["source"]["uncommitted_changes"]}; file hashes are in the JSON). Re-run that script to regenerate both files.

## What the motion is

A bimanual grasp of a **20 × 27 × 27 cm box** on a table, simulated in MuJoCo with the
Gen1 model and the pelvis pinned:

1. Both hands lift forward and up beside the body, then come in over the table. The hand
   path was planned in Cartesian space, so the forearms stay clear of the torso and hips.
2. Along the way each hand rolls 90° about its palm normal. The palms keep facing each
   other, and the fingers end up pointing forward, away from the robot.
3. The palms close on the two **27 × 27 cm faces** (the grip spans the 20 cm side).
4. Optional squeeze: the simulation commands the hands about 5 cm into the box and lets
   the position error become grip force. That gave about 44 N in simulation, with the
   simulation's own PD gains (kp 300). It says nothing about the force on the robot.

There is no lift in this export.

## Files

| File | Content |
|---|---|
| `{JSON_NAME}` | Keyframes (14 arm joints, rad), segment timing, scene in the pelvis frame, per-segment checks, provenance |
| `README.md` | This file |

## Joints

The simulation's arm joints map one-to-one onto the robot's `joint_la1..7` / `joint_ra1..7`,
in the same order. The exporter checked this against
`external/p5_robot_reference/p5_humanoid_v1.0.urdf`: same axes, link offsets within about 2 mm,
mirror signs included.

| Robot | Simulation (left; right is `right_*`) |
|---|---|
| `joint_la1` | `left_shoulder_pitch_joint` |
| `joint_la2` | `left_shoulder_roll_joint` |
| `joint_la3` | `left_shoulder_yaw_joint` |
| `joint_la4` | `left_elbow_roll_joint` |
| `joint_la5` | `left_wrist_yaw_joint` |
| `joint_la6` | `left_wrist_roll_joint` |
| `joint_la7` | `left_wrist_pitch_joint` |

The robot's joint limits are wider than the simulation's, so every keyframe is inside them.
The last column below gives the smallest margin. Waist and head are not in the plan: hold
them at their live values, as `build_command` already does for the 18-slot command. No hand
joints are included.

**The one real geometric difference:** the robot's waist joint sits **21 mm lower** on the
pelvis (z 0.062 vs 0.083 m). The same arm angles therefore put the robot's hands about 2 cm
lower than in the simulation. Every keyframe stores the hand position computed both ways (in
the pelvis frame), and the scene below gives the box position shifted to match the robot.

## Keyframes

Each segment is a rest-to-rest quintic that stays within **{t["max_joint_velocity_rad_s"]} rad/s** and
**{t["max_joint_acceleration_rad_s2"]} rad/s²** per joint, the `MAX_JOINT_VELOCITY_RAD_S` / `MAX_JOINT_ACCELERATION_RAD_S2`
limits in `robot/p5_motion_backend.py`. That is the shape `robot/p5_phase_executor.py`
streams per phase. The arm stops at every keyframe. The reach takes **{t["reach_total_s"]:.0f} s**
this way, against {t["sim_reach_s"]} s in the simulation, where the fastest joint peaked at about 6 rad/s.

Keyframes were picked from the {plan["keyframe_fit"]["dense_path_samples"]}-sample simulated path wherever a
straight joint-space blend between neighbours would leave the planned hand path by more
than {plan["keyframe_fit"]["deviation_threshold_m"] * 1000:.0f} mm. The largest remaining deviation is
{plan["keyframe_fit"]["max_hand_deviation_m"] * 1000:.0f} mm. Each blended segment was then checked in the
simulation:

| Keyframe | Segment s | Max joint Δ rad | Body clearance mm | Table clearance mm | Box clearance mm | Robot hand z vs sim mm | Limit margin rad |
|---|---|---|---|---|---|---|---|
{chr(10).join(kf_rows)}

How the clearances were measured:
- **Body:** forearm and hand against the pelvis, waist, head and legs, using the simulation's
  convex collision hulls. The upper arm is excluded because the torso hull bridges the
  armpit.
- **Table:** the lowest hand corner above the tabletop, counted only while the hand is over
  the table.
- **Box:** a hand against the box. The reach segments must not touch it. In the contact rows a
  negative value is commanded penetration into the box: the squeeze.

`sim_start` is the simulation's start pose: arms hanging, elbows bent 20°. It is a
**reference only**. A real executor starts from the robot's live pose, and the move from
there to the first reach keyframe has to be planned and checked separately.

## Scene, robot pelvis frame (+x forward, +y left, +z up, metres)

| | Simulation geometry | Shifted for the robot's hands |
|---|---|---|
| Box centre | {sc["box"]["center_sim_geometry"]} | {sc["box"]["center_for_p5"]} |
| Tabletop z | {sc["table"]["top_z_sim_geometry"]} | {sc["table"]["top_z_for_p5"]} |

- Box size along pelvis x, y, z: {sc["box"]["size_m"]}. It is axis-aligned and grasped on the
  faces normal to pelvis y.
- The table's near edge is at x = {sc["table"]["near_edge_x"]}; its size is {sc["table"]["size_m"]}.
- The shift is the mean robot-minus-simulation hand offset at `palms_touch_box`:
  {[round(float(v), 4) for v in shift]} m.
- At `palms_touch_box` the robot's hands also come out **{spread * 1000:.0f} mm further apart per side**
  than the simulation's (pelvis y). On the robot that pose would stop short of the box faces
  rather than touch them; the shift above does not correct it.
- The simulation's pelvis is pinned **1.25 m** above the floor with straight legs. What
  matters is the table and box relative to the **standing** robot's pelvis, not to the floor.

## Before this can move a robot

These are open, and each one blocks a physical test:

1. **No executor reads this file.** `robot/p5_phase_executor.py` only sends plans it generated
   and sealed itself from a fresh capture, and its permit is closed. Using these keyframes
   needs a reviewed import into that pipeline (new phase types, its MuJoCo review, the
   live-start check, feedback supervision). The import is not written. The permit and
   `PHYSICAL_MOTION_LOCKED` must not be changed as a side effect.
2. **The hands differ.** The robot has fixed rubber rake hands; the simulation hand is the
   `wrist_pitch_link` slab. The contact keyframes (palm thickness, grip width) are tuned to
   the slab and need re-checking for the rake hands.
3. **The squeeze is a deliberate position error.** Its force on the robot depends on the
   vendor kp/kd, not on simulation numbers. It holds load on the arm motors, and a left
   forearm motor ran warm before the 2026-09-29 incident. Leave `squeeze` out until the
   unloaded motion is verified.
4. **Stream behaviour.** What the controller does when the upper-body stream stops is still
   unverified (`docs/p5_motion_incident_20260929.md`).
5. **Box pose.** These keyframes assume the box exactly where the scene table says. For a
   real box, re-plan from the measured box pose rather than moving the box to fit.

Suggested first physical step, once 1–4 are resolved: the reach keyframes only, ending at
`beside_box` with **no box and no table**, at the limits above, with an operator at the E-stop.
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    sim, fk, rows, fit, n_dense = build_export()
    plan = write(args.out_dir, sim, fk, rows, fit, n_dense)
    print(f"\n{len(plan['keyframes'])} keyframes, reach {plan['timing']['reach_total_s']} s at "
          f"<= {MAX_VEL} rad/s; hand fit {plan['keyframe_fit']['max_hand_deviation_m']*1000:.0f} mm")
    for kf in plan["keyframes"]:
        c = kf.get("segment_checks_sim", {})
        print(f"  {kf['name']:<16} {kf['segment_duration_s'] or 0:5.1f} s  body "
              f"{c.get('min_forearm_hand_to_body_m')}  table {c.get('min_hand_to_tabletop_m')}  "
              f"box {c.get('min_hand_to_box_m')}  limit margin {kf['p5_limit_margin_rad']}")
    print(f"wrote {args.out_dir / JSON_NAME}\n      {args.out_dir / 'README.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
