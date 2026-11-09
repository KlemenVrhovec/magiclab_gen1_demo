"""
Export pick_box_p5's grasp (our robot: P5 v1.0 kinematics, rubber hands) for REVIEW.

    python export_robot_plan_p5.py                       # writes into the magiclab repo
    python export_robot_plan_p5.py --out-dir some/dir    # anywhere else

Writes pick_box_p5_keyframes.json and README.md. Same machinery as export_robot_plan.py
(keyframes picked off the dense IK path, rest-to-rest quintic timing inside the robot's
0.15 rad/s / 0.30 rad/s^2, per-segment checks), with two differences:

* The simulation now IS the robot's kinematics (p5_model), so the robot-URDF FK of every
  keyframe should agree with the simulation exactly; the export records the residual as
  a check instead of shifting the scene to compensate.
* The main sequence (reach_keyframes) runs all the way to grip_0mm: the hands come in
  from beside_box until the inner rubber faces just touch the box's side faces, no
  deeper. A program that plays reach_keyframes start to end therefore closes on the box
  by itself -- the first robot run played the v2 file, whose list stopped at beside_box,
  75 mm out on each side, and the closing move was never made.
* optional_tighter_grips then offers 4, 8 and 12 mm into the box, each a separate step
  FROM grip_0mm. Every grip is simulated in full and exported with the force each rubber
  hand puts on the box and with what ELSE touches the box (it must be nothing: in the
  first P5 attempt the forearms did the squeezing). No add_squeeze over-travel is used.

Nothing here talks to a robot; the JSON is marked not_a_robot_command /
hardware_execution_allowed: false.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import mujoco
import numpy as np

import export_robot_plan as ex
import pick_box_p5 as p5
import pick_box_v3 as v3
from p5_model import HAND_LENGTH, HAND_THICK, HAND_WIDTH, P5_URDF, build_p5_model
from pick_box import TABLE_HALF, TABLE_TOP, TABLE_Y, control

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = ex.MAGICLAB / "sim" / "exports" / "pick_box_p5_rubber_hands"
JSON_NAME = "pick_box_p5_keyframes.json"
FORMAT = "magicbot_sim_joint_keyframe_plan_v3"
END_DEPTHS = (0.0, 0.004, 0.008, 0.012)     # metres the hands are aimed into the box
FINAL_DEPTH = 0.0                            # the main sequence ends here: just touching
RECOMMENDED_DEPTH = p5.GRIP_DEPTH            # 8 mm, ~11 N per hand
HOLD_S = 4.0                                 # simulated time for each force measurement


def tcp_pelvis(fk, q):
    """The robot's own TCP frames (joint_tcp_l / joint_tcp_r), pelvis frame."""
    qd = dict(zip(ex.P5_ARM, q))
    return [fk.link(link, qd)[:3, 3] for link in ("link_tcp_l", "link_tcp_r")]


def measure_grip(depth):
    """Simulate the whole motion ending `depth` into the box; forces and touching parts."""
    model = p5.p5_scene()
    jm = p5.JointMap(model)
    poses = p5.make_plan(model, jm, verbose=False, squeeze=0.0, grip_x=p5.TOUCH_X - depth)
    data = mujoco.MjData(model)
    data.qpos[jm.qadr] = poses["home"]
    mujoco.mj_forward(model, data)
    box = model.body("box").id
    p0 = data.xpos[box].copy()
    while data.time < HOLD_S:
        tgt, _ = v3.target_at(data.time, poses)
        control(model, data, jm, tgt)
        mujoco.mj_step(model, data)
    parts = v3.box_contacts(model, data)
    idx = [jm.index(n) for n in v3.Arms(model).names]
    return {"q": poses["close"][idx], "path": poses["path"], "parts": parts,
            "box_moved": float(np.linalg.norm(data.xpos[box] - p0)),
            "fingers": v3.finger_angle(data, v3.Arms(model))}


def build_export():
    sim = ex.Sim(scene_builder=p5.p5_scene, body_builder=build_p5_model)
    fk = ex.UrdfFK(P5_URDF)

    print("simulating the end positions (full motion each) ...")
    grips = {d: measure_grip(d) for d in END_DEPTHS}
    path = grips[END_DEPTHS[0]]["path"]
    for d in END_DEPTHS[1:]:   # the reach does not depend on the grip depth
        assert np.allclose(grips[d]["path"], path), "reach path differs between grip depths"

    print("choosing keyframes ...")
    keys, fit = ex.pick_keyframes(sim, path)
    lo = np.array([fk.limits[n][0] for n in ex.P5_ARM])
    hi = np.array([fk.limits[n][1] for n in ex.P5_ARM])

    def describe(name, q, prev):
        hs = [sim.to_pelvis(h) for h in sim.hands_world(q)]
        hp = ex.p5_hands(fk, q, sim.hand_offset)
        margin = np.minimum(q - lo, hi - q)
        row = {"name": name, "q": q, "hand_sim": hs, "hand_p5": hp, "tcp": tcp_pelvis(fk, q),
               "fk_mismatch": max(np.linalg.norm(hp[j] - hs[j]) for j in range(2)),
               "min_limit_margin": float(margin.min()), "tightest": ex.P5_ARM[int(margin.argmin())]}
        if prev is None:
            row.update(duration=None)
        else:
            body, who, table, box = sim.segment_checks(prev, q)
            row.update(duration=ex.quintic_duration(q - prev), body=body, body_pair=who,
                       table=table, box=box, other=sim.last_other_box,
                       max_joint_change=float(np.max(np.abs(q - prev))),
                       max_joint_change_joint=ex.P5_ARM[int(np.argmax(np.abs(q - prev)))])
        return row

    reach = []
    for n, k in enumerate(keys):
        name = "sim_start" if n == 0 else ("beside_box" if k == len(path) - 1 else f"reach_{n}")
        q = path[k][sim.idx]
        reach.append(describe(name, q, reach[-1]["q"] if reach else None))
        reach[-1]["path_fraction"] = k / (len(path) - 1)

    def grip_row(d, prev):
        g = grips[d]
        row = describe(f"grip_{d * 1000:.0f}mm", g["q"], prev)
        row.update(depth=d, parts=g["parts"], box_moved=g["box_moved"], fingers_sim=g["fingers"])
        return row

    # the closing move is part of the main sequence: beside_box -> grip_0mm
    reach.append(grip_row(FINAL_DEPTH, reach[-1]["q"]))
    final = reach[-1]["q"]
    ends = [grip_row(d, final) for d in END_DEPTHS if d != FINAL_DEPTH]
    return sim, fk, reach, ends, fit, len(path)


def _fin(x, nd=4):
    return None if x is None or not np.isfinite(x) else round(float(x), nd)


def scene(sim):
    box_c = sim.to_pelvis([0.0, p5.BIG_BOX_Y, TABLE_TOP + p5.BIG_BOX_HALF[2]])
    box_half = np.abs(sim.R_pel.T @ np.array(p5.BIG_BOX_HALF))
    table_c = sim.to_pelvis([0.0, TABLE_Y, TABLE_TOP / 2.0])
    table_half = np.abs(sim.R_pel.T @ np.array(TABLE_HALF))
    return {
        "frame": "robot pelvis: +x forward, +y robot's left, +z up (metres)",
        "box": {"size_m": ex.r(2 * box_half, 3), "center": ex.r(box_c),
                "near_face_x": round(float(box_c[0] - box_half[0]), 4),
                "rotation": "axis-aligned with the pelvis",
                "gripped_faces": "the two 27 x 27 cm faces normal to pelvis y",
                "mass_kg_in_sim": 0.3, "rigid_in_sim": True},
        "table": {"top_z": round(float(sim.to_pelvis([0, 0, TABLE_TOP])[2]), 4),
                  "near_edge_x": round(float(table_c[0] - table_half[0]), 4),
                  "center": ex.r(table_c), "size_m": ex.r(2 * table_half, 3)},
        "hand_contact_along_box_x": "the rubber hand spans from the wrist forward 210 mm; with the "
                                    "wrist 40 mm in front of the box's near face, ~170 mm of the "
                                    "270 mm face is in contact",
        "sim_pelvis_height_above_floor_m": 1.25,
    }


def kf_json(rw):
    kf = {"name": rw["name"], "q": dict(zip(ex.P5_ARM, ex.r(rw["q"], 5))), "q_vector": ex.r(rw["q"], 5),
          "segment_duration_s": None if rw["duration"] is None else round(rw["duration"], 2),
          "hand_center_pelvis_m": {"left": ex.r(rw["hand_p5"][0]), "right": ex.r(rw["hand_p5"][1])},
          "tcp_pelvis_m": {"left": ex.r(rw["tcp"][0]), "right": ex.r(rw["tcp"][1])},
          "urdf_vs_sim_hand_mismatch_m": round(float(rw["fk_mismatch"]), 6),
          "p5_limit_margin_rad": round(rw["min_limit_margin"], 3), "p5_tightest_joint": rw["tightest"]}
    if rw["duration"] is not None:
        kf["segment_checks_sim"] = {
            "min_forearm_hand_to_body_m": _fin(rw["body"]),
            "closest_body_pair": rw.get("body_pair") or None,
            "min_hand_to_tabletop_m": _fin(rw["table"]),
            "min_hand_to_box_m": _fin(rw["box"]),
            "min_other_robot_part_to_box_m": _fin(rw["other"][0]),
            "closest_other_part": rw["other"][1] or None,
            "max_joint_change_rad": round(rw["max_joint_change"], 3),
            "max_joint_change_joint": rw["max_joint_change_joint"]}
    return kf


def hold_json(rw):
    hands = {k: round(v, 1) for k, v in rw["parts"].items() if k.endswith("rubber_hand")}
    others = {k: round(v, 1) for k, v in rw["parts"].items() if not k.endswith("rubber_hand")}
    return {"aim_into_box_m": rw["depth"],
            "simulated_hold": {"normal_force_per_hand_N": hands,
                               "other_robot_parts_on_box_N": others,
                               "box_displacement_m": round(rw["box_moved"], 4),
                               "fingers_off_forward_deg": [round(float(f), 1) for f in rw["fingers_sim"]],
                               "hold_time_s": HOLD_S,
                               "note": "simulation kp=300, rigid box; force on the robot depends on "
                                       "the vendor joint gains and the real box"}}


def side_gap(rw):
    """Planned gap between each inner rubber face and the box face it closes on (m)."""
    return round(float(abs(rw["hand_p5"][0][1]) - HAND_THICK / 2 - p5.BIG_BOX_HALF[0]), 4) + 0.0


def write(out_dir, sim, fk, reach, ends, fit, n_dense):
    out_dir.mkdir(parents=True, exist_ok=True)
    reach_t = sum(rw["duration"] for rw in reach[1:])
    beside = next(rw for rw in reach if rw["name"] == "beside_box")
    sources = ["pick_box.py", "pick_box_v2.py", "pick_box_v3.py", "pick_box_v4.py", "pick_box_p5.py",
               "p5_model.py", "export_robot_plan.py", "export_robot_plan_p5.py"]
    plan = {
        "format": FORMAT,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "passive": True, "not_a_robot_command": True, "hardware_execution_allowed": False,
        "status": "offline_review_only",
        "summary": ("Bimanual box grasp simulated on the P5 v1.0 kinematics with the rubber-hand proxy "
                    "(pick_box_p5): hands lifted forward beside the body, fingers pointing forward, inner "
                    "rubber faces on the two 27x27 cm faces of a 20x27x27 cm box. reach_keyframes run "
                    "from the start to grip_0mm (hands just touching the box); optional_tighter_grips "
                    "add grip depth from there."),
        "source": {"repo": "6_MagicLab_Gen1_Simulation", **ex.git_state(),
                   "files_sha256": {f: ex.sha256(HERE / f) for f in sources},
                   "sim_model": "p5_model.build_p5_model: Gen1 meshes on P5 v1.0 joint frames, P5 limits "
                                "and inertias, rubber-hand slab proxy"},
        "robot_model": {"urdf": "external/p5_robot_reference/p5_humanoid_v1.0.urdf",
                        "urdf_sha256_lf": ex.sha256(P5_URDF)},
        "hand_proxy": {"source": "magiclab sim/p5_rubber_hand_proxy.py MEASUREMENTS",
                       "size_m": [HAND_WIDTH, HAND_THICK, HAND_LENGTH],
                       "frame": "link_la7/ra7: width along x, thickness along y, length down -z from the "
                                "wrist centre; inner face -y (left) / +y (right)",
                       "calibration_status": "uncalibrated (inner surface assumed 15 mm from the wrist axis)"},
        "joint_order": ex.P5_ARM, "units": "rad",
        "not_commanded": {"waist_and_head": "hold the live measured values (joint_wy, joint_wr, joint_hy, joint_hp)",
                          "hands": "no hand joints; fixed rubber rake hands"},
        "timing": {"model": "rest-to-rest quintic per segment (stops at every keyframe)",
                   "max_joint_velocity_rad_s": ex.MAX_VEL, "max_joint_acceleration_rad_s2": ex.MAX_ACC,
                   "limits_from": "robot/p5_motion_backend.py",
                   "total_to_final_keyframe_s": round(reach_t, 2)},
        "keyframe_fit": {"dense_path_samples": n_dense, "max_hand_deviation_m": round(float(fit), 4),
                         "deviation_threshold_m": ex.MAX_HAND_DEVIATION_M},
        "scene": scene(sim),
        "null_values": {"min_forearm_hand_to_body_m": "clearer than the 20 mm probe range",
                        "min_hand_to_tabletop_m": "no hand was over the table in this segment",
                        "min_hand_to_box_m": "clearer than the 50 mm probe range",
                        "min_other_robot_part_to_box_m": "clearer than the 50 mm probe range"},
        "how_to_play": ("Play reach_keyframes in order, each to its q over its segment_duration_s "
                        "(sim_start is a reference only: start from the live pose). The last entry, "
                        "grip_0mm, is the closing move: hands just touching the box. optional_tighter_grips "
                        "are single further steps FROM grip_0mm; play at most one."),
        "sequence": [rw["name"] for rw in reach],
        "final_keyframe": reach[-1]["name"],
        "expected_gap_per_side_m": {"beside_box": round(side_gap(beside), 4),
                                    reach[-1]["name"]: round(side_gap(reach[-1]), 4),
                                    "meaning": "inner rubber face to box side face, from the planned hand "
                                               "centre, half the hand thickness and half the box width; "
                                               "measure it on the robot at beside_box"},
        "reach_keyframes": [],
        "optional_tighter_grips": [],
        "recommended_for_lifting": f"grip_{RECOMMENDED_DEPTH * 1000:.0f}mm",
    }
    for rw in reach:
        kf = kf_json(rw)
        if "depth" in rw:
            kf.update(hold_json(rw))
        plan["reach_keyframes"].append(kf)
    for rw in ends:
        kf = kf_json(rw)
        kf["from"] = reach[-1]["name"]
        kf.update(hold_json(rw))
        plan["optional_tighter_grips"].append(kf)
    (out_dir / JSON_NAME).write_text(json.dumps(plan, indent=2), encoding="utf-8")
    (out_dir / "README.md").write_text(readme(plan, reach, ends), encoding="utf-8")
    return plan


def _mm(x, none="> 20"):
    return none if x is None or not np.isfinite(x) else f"{x * 1000:.0f}"


def readme(plan, reach, ends):
    sc, t = plan["scene"], plan["timing"]
    final = reach[-1]
    gaps = plan["expected_gap_per_side_m"]
    worst_fk = max(rw["fk_mismatch"] for rw in reach + ends)

    def force(rw):
        f = {k: v for k, v in rw["parts"].items() if k.endswith("rubber_hand")}
        return f.get("left_rubber_hand", 0.0), f.get("right_rubber_hand", 0.0)

    def others(rw):
        o = {k: round(v, 1) for k, v in rw["parts"].items() if not k.endswith("rubber_hand")}
        return "none" if not o else str(o)

    seq_rows = []
    for i, rw in enumerate(reach):
        if rw["duration"] is None:
            seq_rows.append(f"| {i} | `{rw['name']}` | - | - | - | - | - | - | {rw['min_limit_margin']:.2f} |")
        else:
            seq_rows.append(
                f"| {i} | `{rw['name']}` | {rw['duration']:.1f} | {rw['max_joint_change']:.2f} | {_mm(rw['body'])} | "
                f"{_mm(rw['table'], 'not over table')} | {_mm(rw['box'], '> 50')} | "
                f"{_mm(rw['other'][0], '> 50')} | {rw['min_limit_margin']:.2f} |")
    fl, fr = force(final)
    grip_rows = [f"| `{final['name']}` (end of sequence) | 0 mm | {fl:.1f} | {fr:.1f} | {others(final)} | "
                 f"{_mm(final['other'][0], '> 50')} | - | {final['box_moved'] * 1000:.1f} |"]
    for rw in ends:
        fl, fr = force(rw)
        rec = " **for lifting**" if rw["name"] == plan["recommended_for_lifting"] else ""
        grip_rows.append(
            f"| `{rw['name']}`{rec} | {rw['depth'] * 1000:.0f} mm | {fl:.1f} | {fr:.1f} | {others(rw)} | "
            f"{_mm(rw['other'][0], '> 50')} | {rw['duration']:.1f} | {rw['box_moved'] * 1000:.1f} |")
    beside_y = abs(reach[-2]["hand_p5"][0][1])
    final_y = abs(final["hand_p5"][0][1])
    return f"""# Simulated box grasp on the P5 robot (pick_box_p5) — joint keyframe export

**Status: offline review only. Not a robot command.** The JSON is marked
`not_a_robot_command: true` and `hardware_execution_allowed: false`. No robot code in this
repository reads it.

Generated {plan["created_utc"]} by `magicbot_sim/export_robot_plan_p5.py` in the
`6_MagicLab_Gen1_Simulation` repo (commit `{plan["source"]["commit"]}`, uncommitted changes:
{plan["source"]["uncommitted_changes"]}; file hashes are in the JSON). Re-run that script to regenerate both files.

**Format `{plan["format"]}`. What changed from v2:** the main sequence `reach_keyframes` now
ends at **`{final["name"]}`**, with the hands closed onto the box's sides and just touching.
In v2 it stopped at `beside_box`, 75 mm out on each side, and the closing move lived in a
separate list. A program that played only `reach_keyframes` therefore never closed in.
`end_positions` is replaced by `optional_tighter_grips`, which start from `{final["name"]}`.

This export supersedes `../pick_box_v4_fingers_forward/`, which was simulated on the generic
Gen1 model.

## How to read the JSON

1. **Play `reach_keyframes` in order.** Move to each entry's `q` (joint name → rad) or
   `q_vector` (14 values in `joint_order`) over its `segment_duration_s`, stopping at every
   keyframe. `sim_start` is a reference only: start from the robot's live pose and plan
   that first move separately.
2. **The last entry, `{final["name"]}`, is the closing move.** The hands come in from `beside_box`
   until the inner rubber faces touch the box's side faces, without pressing into it. The
   order is in `sequence`, and the last name in `final_keyframe`.
3. **Optional:** to grip harder, play **one** entry of `optional_tighter_grips` after
   `{final["name"]}`. Each has `"from": "{final["name"]}"`.
4. Waist and head are not in the plan: hold them at their live values. There are no hand
   joints.

Each keyframe also gives:
- `hand_center_pelvis_m`: the rubber-hand slab centre, from the robot's URDF;
- `tcp_pelvis_m`: the robot's own `link_tcp_l/r`;
- `segment_checks_sim`: the simulated clearances for the move into that keyframe.

## Check on the robot: the gap per side

`expected_gap_per_side_m` gives the planned gap between each inner rubber face and the box
face it closes on:

| Keyframe | Hand centre from the midline | Gap per side |
|---|---|---|
| `beside_box` | {beside_y * 1000:.0f} mm | **{gaps["beside_box"] * 1000:.0f} mm** |
| `{final["name"]}` | {final_y * 1000:.0f} mm | **{gaps[final["name"]] * 1000:.0f} mm** (touching) |

Measure both sides at `beside_box` before running the closing move. If the real gap is not
{gaps["beside_box"] * 1000:.0f} mm, `{final["name"]}` will stop short of the box or press into it by the
difference. In the first robot run the gap looked like about 10 cm.

These things are **not in the file** and have to be measured on the robot:
1. The gap per side at `beside_box`.
2. Where the box actually sits relative to the pelvis. The plan assumes the centre at
   {sc["box"]["center"]}, on the robot's midline.
3. Where the inner rubber surface sits relative to the wrist axis. The plan assumes
   15 mm; this is `MISSING_CONTACT_MEASUREMENT` in `sim/p5_rubber_hand_proxy.py`.
4. The box weight, before using a tighter grip to lift.

## The simulation behind it

- **Kinematics:** the joints, limits and masses are those of
  `external/p5_robot_reference/p5_humanoid_v1.0.urdf` (sha256 `065fdda5…` with LF line
  endings). The P5 shape files are not available, so the drawn and colliding shapes are the
  Gen1 meshes on the P5 joint frames.
- **Model check:** the hand positions computed with the URDF and with the simulation agree to
  **{worst_fk * 1000:.3f} mm** at every keyframe.
- **Hands:** the rubber-hand proxy from `sim/p5_rubber_hand_proxy.py`: a
  {HAND_WIDTH * 1000:.0f} × {HAND_THICK * 1000:.0f} × {HAND_LENGTH * 1000:.0f} mm slab from the `link_la7`/`ra7` origin.
  **Uncalibrated.**

## Scene, robot pelvis frame (+x forward, +y left, +z up, metres)

| | |
|---|---|
| Box | {sc["box"]["size_m"]} m along pelvis x, y, z; centre {sc["box"]["center"]}; near face at x = {sc["box"]["near_face_x"]} |
| Gripped faces | the two 27 × 27 cm faces, normal to pelvis y |
| Tabletop | z = {sc["table"]["top_z"]}; near edge at x = {sc["table"]["near_edge_x"]} |
| Hand centres at `{final["name"]}` | x = {final["hand_p5"][0][0]:.3f}, y = ±{final_y:.3f}, z = {round(float(final["hand_p5"][0][2]), 3) + 0.0:.3f}; each palm covers z ± 50 mm |

The box and table matter **relative to the standing robot's pelvis**. The simulation's
pelvis is pinned at {sc["sim_pelvis_height_above_floor_m"]} m, which is arbitrary.

**Where along the box the hands go.** Each wrist stays 40 mm in front of the box's near
face, so only the rubber touches the box, along about 170 mm of its 270 mm side. With the
hands centred on the box's depth instead, the forearm housings reach the box first and do
the squeezing. That happened in the first P5 simulation: about 50 N per side, with the
hands not touching.

## The sequence (`reach_keyframes`)

Each segment is a rest-to-rest quintic within **{t["max_joint_velocity_rad_s"]} rad/s** and
**{t["max_joint_acceleration_rad_s2"]} rad/s²** per joint, the limits in `robot/p5_motion_backend.py`.
The whole sequence to `{final["name"]}` takes **{t["total_to_final_keyframe_s"]:.0f} s**. The keyframes keep the
hands within {plan["keyframe_fit"]["max_hand_deviation_m"] * 1000:.0f} mm of the planned path.

| # | Keyframe | Segment s | Max joint Δ rad | Body clearance mm | Table clearance mm | Hand to box mm | Other robot part to box mm | Limit margin rad |
|---|---|---|---|---|---|---|---|---|
{chr(10).join(seq_rows)}

How the clearances were measured:
- **Body:** forearm and hand against the pelvis, waist, head and legs.
- **Table:** the lowest hand corner above the tabletop, counted only while the hand is over
  the table.
- **Box:** checked separately for the hands and for every other robot part. In the last row
  the hand reaches the box, which is the intended contact. No other part may.

## Grip force

The force is the normal force each rubber hand puts on the box, measured after the full
motion and a hold of {HOLD_S:.0f} s in simulation, with joint gain kp 300 and a rigid box.

| Grip | Aimed into box | Left hand N | Right hand N | Other parts on box | Other part to box mm | Step from `{final["name"]}` s | Box moved mm |
|---|---|---|---|---|---|---|---|
{chr(10).join(grip_rows)}

Force grows about 1.1 N per mm in simulation. On the robot it depends on the vendor joint
gains.

`{final["name"]}` only touches the box; it doesn't hold it. To lift an **empty** box, friction from
both hands must carry it: 2 × μ × N ≥ m × g.

| Box | Needed per hand | With a 2× margin |
|---|---|---|
| Empty, 0.5 kg, rubber on cardboard μ ≈ 0.5 | about 5 N | about **10 N** |

That is `{plan["recommended_for_lifting"]}`. Go there one step at a time and check by eye that only the
rubber hands touch the box.

## Before this can move a robot

1. **No executor reads this file.** `robot/p5_phase_executor.py` only sends plans it sealed
   itself from a fresh capture, and its permit is closed. Do not change the permit or
   `PHYSICAL_MOTION_LOCKED` as a side effect.
2. **Contact surface:** it is uncalibrated, and the box position is assumed. Measure the gap
   at `beside_box` (see above).
3. **Forearm clearance:** it depends on the box position. If the box sits 2 cm closer to the
   robot, the forearm touches it again.
4. **Stream behaviour:** what the controller does when the stream stops is unverified
   (`docs/p5_motion_incident_20260929.md`). Holding a grip keeps the arm motors loaded, and
   a stall or tracking check may stop a move whose hand is blocked by the box.
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    sim, fk, reach, ends, fit, n_dense = build_export()
    plan = write(args.out_dir, sim, fk, reach, ends, fit, n_dense)
    print(f"\n{len(reach)} keyframes to {plan['final_keyframe']}, "
          f"{plan['timing']['total_to_final_keyframe_s']} s; hand fit "
          f"{plan['keyframe_fit']['max_hand_deviation_m'] * 1000:.0f} mm")
    for kf in plan["reach_keyframes"] + plan["optional_tighter_grips"]:
        c = kf.get("segment_checks_sim", {})
        extra = kf.get("simulated_hold", {}).get("normal_force_per_hand_N", "")
        print(f"  {kf['name']:<12} {kf['segment_duration_s'] or 0:5.1f} s  body "
              f"{c.get('min_forearm_hand_to_body_m')}  table {c.get('min_hand_to_tabletop_m')}  "
              f"other-part-to-box {c.get('min_other_robot_part_to_box_m')} ({c.get('closest_other_part')})  "
              f"urdf-sim {kf['urdf_vs_sim_hand_mismatch_m'] * 1000:.3f} mm  {extra}")
    print(f"wrote {args.out_dir / JSON_NAME}\n      {args.out_dir / 'README.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
