"""
Bimanual pick: the pinned MagicBot Gen1 lifts a box off a table using both hands.

    python pick_box.py               # interactive viewer
    python pick_box.py --headless    # no window, reports whether the lift succeeded

WHY BOTH HANDS. The MJCF we have (MAGICBOT.xml) ends each arm at wrist_pitch_link -- a
solid mesh, no fingers, nothing to close around an object. The repo ships a separate
MAGICBOT_with_hand.urdf but no MJCF equivalent. So the only grasp available is a
bimanual pinch: trap the box between the two flat hand faces and squeeze hard enough
that friction carries its weight. That is a real technique for gripperless robots.

WHY THE TABLE IS CHEST HIGH. With the pelvis pinned at 1.25 m, sampling 30000 random
arm poses shows the hand's collision geom cannot get below z = 1.017 m, and can only
cross to the body midline in the band z = 1.20..1.30. A floor-height table is
unreachable, so the table top sits at 1.205 m and the box is grasped at 1.25 m.

AIM AT THE GEOM, NOT THE BODY. The hand's collision slab is offset [0.008, -0.029,
-0.102] from wrist_pitch_link's origin -- about 10 cm below it. The IK therefore
targets the geom centre via mj_jac, otherwise the hands close 10 cm above the box.

HOW IT WORKS. Four waypoints (home -> reach -> squeeze -> lift) are converted to joint
angles once at startup by a damped-least-squares IK controlling hand position AND
orientation (the flat faces must end up facing each other). The loop interpolates
between those joint vectors and tracks them with the same gravity-compensated PD used
in arm_demo.py. The squeeze comes from commanding the hands slightly closer together
than the box actually is: the box blocks them, and the PD turns that leftover position
error into grip force.

The cloned magicbot-gen1_description repo is never modified; the pinned pelvis, the
table and the box are all added to an in-memory MjSpec copy.
"""

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

HERE = Path(__file__).resolve().parent
MJCF = HERE / "magicbot-gen1_description" / "mjcf" / "MAGICBOT.xml"

# --- controller (same gains as arm_demo.py) ---
KP, KD, TAU_LIMIT = 300.0, 8.0, 100.0

# --- scene layout, world coordinates. the robot faces world -Y ---
# Two clearances drive these numbers, both found by measurement:
#  * the dangling legs sit around y = 0, so the table must start well forward of that
#    or it swallows the feet;
#  * the hand slab is 27 cm long and hangs ~13.5 cm below the point being aimed at, so
#    the grasp height must clear the tabletop -- hence a TALL box, gripped near its top.
PELVIS_H = 1.25
TABLE_TOP = 1.10
TABLE_Y = -0.45
TABLE_HALF = [0.32, 0.16, TABLE_TOP / 2.0]

BOX_HALF_XY = 0.045       # 9 cm square footprint
BOX_HALF_Z = 0.10         # 20 cm tall -> a block, not a cube
BOX_MASS = 0.30           # kg
BOX_Y = -0.36             # near the front edge of the table, to stay inside arm reach
BOX_Z = TABLE_TOP + BOX_HALF_Z              # 1.20: resting on the tabletop
BOX_FRICTION = 2.0        # sliding friction, so the pinch can hold it

# --- grasp geometry (positions of the hand GEOM centres, not the body origins) ---
HAND_HALF_THICK = 0.0228  # geom_size[0] of the hand slab
GRASP_Z = 1.27            # near the top of the box, so the hand clears the tabletop
TOUCH_X = BOX_HALF_XY + HAND_HALF_THICK     # 0.068: faces just kissing the box
OPEN_X = TOUCH_X + 0.075                    # approach from outside
GRIP_X = TOUCH_X - 0.018                    # commanded inside the box -> squeeze
LIFT_H = 0.15

# Approach path. Interpolating straight from the arms-down home pose to the grasp drags
# both hands through the tabletop, so the arms first go OUT to the sides and UP, well
# clear of the table, and only then come forward and inward onto the box.
SPREAD_X, SPREAD_Y, SPREAD_Z = 0.42, -0.15, 1.50

# Metres of inward over-travel commanded once the hands are on the box. The box blocks
# the motion, so this becomes grip force rather than movement. See add_squeeze().
SQUEEZE = 0.030

# Friction-constraint stiffness relative to normal contacts. MuJoCo's default of 1 lets
# a held object creep; manipulation tasks normally want 10-100.
IMPRATIO = 30.0

ARM_L = ["left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
         "left_elbow_roll_joint", "left_wrist_yaw_joint", "left_wrist_roll_joint",
         "left_wrist_pitch_joint"]
ARM_R = [n.replace("left", "right") for n in ARM_L]

PHASES = [("home", 0.0), ("spread", 2.5), ("approach", 5.0), ("close", 6.5),
          ("lift", 9.0), ("hold", 11.5)]
IK_PHASES = ("spread", "approach", "close", "lift")


def build_model() -> mujoco.MjModel:
    """Official MJCF + pinned pelvis + a fixed table + a dynamic box."""
    spec = mujoco.MjSpec.from_file(str(MJCF))

    # MuJoCo models friction as a soft spring-damper, so a gripped object creeps
    # downward at roughly (its weight / the constraint damping) no matter how hard it is
    # squeezed -- measured at 2.3 mm/s with the contacts only 10% of the way to their
    # friction limit. impratio stiffens friction constraints relative to normal ones and
    # is the documented fix for exactly this. It is specified for ELLIPTIC cones; with
    # MuJoCo's default pyramidal cone a high impratio also distorts the normal direction.
    # Measured over a 30 s hold: pyramidal+1 drifts 1.88 mm/s, pyramidal+30 drifts
    # 0.044, elliptic+30 drifts 0.000, for ~14% more CPU.
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = IMPRATIO

    pelvis = spec.body("pelvis")
    spec.delete(pelvis.joints[0])          # remove freejoint -> pelvis welded to world
    pelvis.pos = [0.0, 0.0, PELVIS_H]

    # table: a body with NO joint is rigidly fixed to the world
    table = spec.worldbody.add_body()
    table.name = "table"
    table.pos = [0.0, TABLE_Y, TABLE_TOP / 2.0]
    tg = table.add_geom()
    tg.name = "table_geom"
    tg.type = mujoco.mjtGeom.mjGEOM_BOX
    tg.size = TABLE_HALF
    tg.rgba = [0.55, 0.42, 0.30, 1.0]
    # the model's defaults set contype=0; without this the table would not collide
    tg.contype, tg.conaffinity = 1, 1

    # box: has a freejoint, so it is dynamic
    box = spec.worldbody.add_body()
    box.name = "box"
    box.pos = [0.0, BOX_Y, BOX_Z]
    box.add_freejoint()
    bg = box.add_geom()
    bg.name = "box_geom"
    bg.type = mujoco.mjtGeom.mjGEOM_BOX
    bg.size = [BOX_HALF_XY, BOX_HALF_XY, BOX_HALF_Z]
    bg.rgba = [0.85, 0.25, 0.20, 1.0]
    bg.mass = BOX_MASS
    bg.contype, bg.conaffinity = 1, 1
    bg.friction = [BOX_FRICTION, 0.02, 0.001]

    return spec.compile()


def hand_geom(model, side: str) -> int:
    """The collision geom on that side's wrist_pitch_link."""
    b = model.body(f"{side}_wrist_pitch_link").id
    for g in range(model.body_geomadr[b], model.body_geomadr[b] + model.body_geomnum[b]):
        if model.geom_conaffinity[g] or model.geom_contype[g]:
            return g
    raise RuntimeError(f"no collision geom on {side} hand")


class JointMap:
    """Every actuator -> the joint it drives, and that joint's state indices."""

    def __init__(self, model):
        self.model = model
        self.qadr = np.zeros(model.nu, dtype=int)
        self.vadr = np.zeros(model.nu, dtype=int)
        self.name = []
        for a in range(model.nu):
            j = model.actuator_trnid[a, 0]
            self.qadr[a] = model.jnt_qposadr[j]
            self.vadr[a] = model.jnt_dofadr[j]
            self.name.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j))
        self.lo = model.jnt_range[model.actuator_trnid[:, 0], 0].copy()
        self.hi = model.jnt_range[model.actuator_trnid[:, 0], 1].copy()

    def index(self, n):
        return self.name.index(n)


class Arms:
    """Index bookkeeping plus position+orientation IK on the hand collision geoms."""

    def __init__(self, model):
        self.model = model
        self.names = ARM_L + ARM_R
        self.qids = np.array([model.jnt_qposadr[model.joint(n).id] for n in self.names])
        self.dofs = np.array([model.jnt_dofadr[model.joint(n).id] for n in self.names])
        self.lo = np.array([model.jnt_range[model.joint(n).id, 0] for n in self.names])
        self.hi = np.array([model.jnt_range[model.joint(n).id, 1] for n in self.names])
        self.bodies = [model.body("left_wrist_pitch_link").id,
                       model.body("right_wrist_pitch_link").id]
        self.geoms = [hand_geom(model, "left"), hand_geom(model, "right")]

    def solve(self, data, tl, tr, iters=3000, lam=0.10, step=0.5, w_rot=0.30):
        """Damped least squares on the geom centres. Returns (angles, err_l, err_r)."""
        R_target = np.eye(3).flatten()   # hand thin axis (local X) parallel to world X
        q_tgt = np.zeros(4)
        mujoco.mju_mat2Quat(q_tgt, R_target)

        for _ in range(iters):
            mujoco.mj_forward(self.model, data)
            rows, errs = [], []
            for body, geom, tgt in zip(self.bodies, self.geoms, (tl, tr)):
                point = data.geom_xpos[geom].copy()
                jp = np.zeros((3, self.model.nv))
                jr = np.zeros((3, self.model.nv))
                mujoco.mj_jac(self.model, data, jp, jr, point, body)
                rows.append(jp[:, self.dofs])
                errs.append(tgt - point)

                q_cur, q_inv, q_err, w = np.zeros(4), np.zeros(4), np.zeros(4), np.zeros(3)
                mujoco.mju_mat2Quat(q_cur, data.geom_xmat[geom])
                mujoco.mju_negQuat(q_inv, q_cur)
                mujoco.mju_mulQuat(q_err, q_tgt, q_inv)
                mujoco.mju_quat2Vel(w, q_err, 1.0)
                rows.append(jr[:, self.dofs])
                errs.append(w_rot * w)

            J = np.vstack(rows)
            e = np.concatenate(errs)
            if np.linalg.norm(e) < 1e-4:
                break
            dq = J.T @ np.linalg.solve(J @ J.T + lam**2 * np.eye(J.shape[0]), e)
            data.qpos[self.qids] = np.clip(data.qpos[self.qids] + step * dq, self.lo, self.hi)

        mujoco.mj_forward(self.model, data)
        return (data.qpos[self.qids].copy(),
                np.linalg.norm(tl - data.geom_xpos[self.geoms[0]]),
                np.linalg.norm(tr - data.geom_xpos[self.geoms[1]]))


def hand_targets(phase):
    """World targets for the two hand geom centres (left, right)."""
    if phase == "spread":
        c = np.array([0.0, SPREAD_Y, SPREAD_Z])
        gap = SPREAD_X
    else:
        rise = LIFT_H if phase in ("lift", "hold") else 0.0
        c = np.array([0.0, BOX_Y, GRASP_Z + rise])
        gap = OPEN_X if phase == "approach" else GRIP_X
    return c + [gap, 0, 0], c + [-gap, 0, 0]


def add_squeeze(model, jm, arms, pose, amount):
    """Bias the commanded joint angles inward, past where the hands can actually go.

    IK saturates at the box surface, so asking it for a tighter grip changes nothing.
    Instead take one Jacobian step in the 'both hands move inward' direction and add it
    to the commanded pose. The box physically blocks that motion, so the PD turns the
    resulting permanent joint error into grip force -- amount is in metres of virtual
    over-travel, not a force.
    """
    scratch = mujoco.MjData(model)
    scratch.qpos[jm.qadr] = pose
    mujoco.mj_forward(model, scratch)

    rows, errs = [], []
    for body, geom, inward in zip(arms.bodies, arms.geoms, (-1.0, +1.0)):
        jp = np.zeros((3, model.nv))
        mujoco.mj_jac(model, scratch, jp, None, scratch.geom_xpos[geom].copy(), body)
        rows.append(jp[:, arms.dofs])
        errs.append(np.array([inward * amount, 0.0, 0.0]))

    J = np.vstack(rows)
    e = np.concatenate(errs)
    dq = J.T @ np.linalg.solve(J @ J.T + 0.05**2 * np.eye(6), e)

    out = pose.copy()
    biased = np.clip(scratch.qpos[arms.qids] + dq, arms.lo, arms.hi)
    for n, v in zip(arms.names, biased):
        out[jm.index(n)] = v
    return out


def plan(model, jm, verbose=True, squeeze=0.0) -> dict:
    """Solve IK once per waypoint; return full 30-vector joint targets."""
    scratch = mujoco.MjData(model)
    arms = Arms(model)

    home = np.zeros(model.nu)
    home[jm.index("left_shoulder_roll_joint")] = 0.25
    home[jm.index("right_shoulder_roll_joint")] = -0.25

    poses = {"home": home}
    # seed with arms already lifted out sideways, which is roughly the 'spread' pose --
    # keeps the IK away from the elbow-backwards solutions it otherwise falls into
    seed = np.deg2rad([-20, 70, -20, -40, 0, 0, 0, -20, -70, 20, 40, 0, 0, 0])
    for name in IK_PHASES:
        tl, tr = hand_targets(name)
        mujoco.mj_resetData(model, scratch)
        scratch.qpos[jm.qadr] = home
        scratch.qpos[arms.qids] = np.clip(seed, arms.lo, arms.hi)
        q, el, er = arms.solve(scratch, tl, tr)
        full = home.copy()
        for n, v in zip(arms.names, q):
            full[jm.index(n)] = v
        poses[name] = full
        seed = q                                   # warm-start the next waypoint
        if verbose:
            print(f"  IK {name:<8} left {el*1000:6.1f} mm   right {er*1000:6.1f} mm")

    if squeeze > 0:
        for name in ("close", "lift"):
            poses[name] = add_squeeze(model, jm, arms, poses[name], squeeze)
        if verbose:
            print(f"  squeeze bias: {squeeze*1000:.0f} mm of inward over-travel")

    poses["hold"] = poses["lift"]
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


def control(model, data, jm, target):
    q = data.qpos[jm.qadr]
    v = data.qvel[jm.vadr]
    tau = data.qfrc_bias[jm.vadr] + KP * (target - q) - KD * v
    data.ctrl[:] = np.clip(tau, -TAU_LIMIT, TAU_LIMIT)


def grip_force(model, data):
    """Normal force the hands (not the table) are pressing on the box with."""
    bg = model.geom("box_geom").id
    tg = model.geom("table_geom").id
    total = 0.0
    for i in range(data.ncon):
        c = data.contact[i]
        if bg in (c.geom1, c.geom2):
            other = c.geom2 if c.geom1 == bg else c.geom1
            if other != tg:
                f = np.zeros(6)
                mujoco.mj_contactForce(model, data, i, f)
                total += abs(f[0])
    return total


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--headless", action="store_true", help="no window; report the outcome")
    ap.add_argument("--duration", type=float, default=16.0)
    ap.add_argument("--squeeze", type=float, default=SQUEEZE,
                    help="metres of inward over-travel commanded into the box (grip force)")
    args = ap.parse_args()

    model = build_model()
    jm = JointMap(model)
    print("planning grasp:")
    poses = plan(model, jm, squeeze=args.squeeze)

    data = mujoco.MjData(model)
    data.qpos[jm.qadr] = poses["home"]
    mujoco.mj_forward(model, data)

    box = model.body("box").id
    z0 = data.xpos[box][2]
    print(f"\ntable top {TABLE_TOP:.3f} m, box centre starts at z={z0:.3f} m")

    def step():
        tgt, phase = target_at(data.time, poses)
        control(model, data, jm, tgt)
        mujoco.mj_step(model, data)
        return phase

    if args.headless:
        peak, rows, nxt = z0, [], 0.0
        while data.time < args.duration:
            phase = step()
            peak = max(peak, data.xpos[box][2])
            if data.time >= nxt:
                nxt += 1.0
                rows.append((data.time, phase, data.xpos[box][2], grip_force(model, data)))
        print(f"\n{'t':>5} {'phase':<14} {'box z':>7} {'grip N':>8}")
        for t, p, z, f in rows:
            print(f"{t:5.1f} {p:<14} {z:7.3f} {f:8.1f}")
        lift = peak - z0
        print(f"\nbox rose {lift*1000:.0f} mm above its start height")
        ok = lift > 0.05
        print("RESULT:", "LIFTED" if ok else "not lifted")
        return 0 if ok else 1

    print("left-drag orbit | scroll zoom | ctrl+right-drag push | Space pause | Esc quit\n")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        wall = time.perf_counter()
        while viewer.is_running():
            step()
            viewer.sync()
            lag = data.time - (time.perf_counter() - wall)
            if lag > 0:
                time.sleep(lag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
