"""
A guided tour of the 7 DOF in each MagicBot Gen1 arm — one joint at a time.

Each joint is driven home -> lower limit -> upper limit -> home with cosine easing,
while every other joint is held at the home pose by the same PD loop. So what moves
in the window is exactly one axis, and you see its whole range.

    python arm_dof_tour.py                 # both arms, mirrored, full tour (~60 s)
    python arm_dof_tour.py --arm left      # left arm only
    python arm_dof_tour.py --joint elbow_roll     # one joint, loops until you quit
    python arm_dof_tour.py --speed 2       # twice as fast
    python arm_dof_tour.py --report        # no window: run it and print the ROM table

The 7 axes, shoulder outwards:

    1 shoulder_pitch   Y   arm swings forward / back
    2 shoulder_roll    X   arm lifts out sideways
    3 shoulder_yaw     Z   upper arm twists about its own length
    4 elbow_roll       X   elbow bends
    5 wrist_yaw        Z   forearm twists
    6 wrist_roll       X   hand tilts
    7 wrist_pitch      Y   hand nods

Note on self-collision: every arm geom in MAGICBOT.xml carries contype=0 conaffinity=0,
so the arms collide with nothing, not even the torso. At the extremes of shoulder_roll
and elbow_roll the mesh passes visibly through the body. That is the model, not the
controller — the numbers below are the MJCF joint limits, not a reachability check.
"""

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

HERE = Path(__file__).resolve().parent
MJCF = HERE / "magicbot-gen1_description" / "mjcf" / "MAGICBOT.xml"

KP = 300.0
KD = 8.0
TAU_LIMIT = 100.0  # N*m; the MJCF declares no forcerange, so impose a sane one
PINNED_PELVIS_HEIGHT = 1.25  # m, legs dangle clear of the ground

# The 7 arm axes in shoulder-to-hand order, with the sign that mirrors a left-arm
# command onto the right arm. The left joint is always the driver; the right joint
# follows at MIRROR * the same angle, which lands inside its own (mirrored) limits.
ARM_JOINTS = [
    ("shoulder_pitch", +1.0, "arm swings forward / back"),
    ("shoulder_roll", -1.0, "arm lifts out sideways"),
    ("shoulder_yaw", -1.0, "upper arm twists about its length"),
    ("elbow_roll", -1.0, "elbow bends"),
    ("wrist_yaw", -1.0, "forearm twists"),
    ("wrist_roll", -1.0, "hand tilts"),
    ("wrist_pitch", +1.0, "hand nods"),
]

# Per-joint schedule, in seconds at --speed 1.
SEGMENTS = [
    ("settle", 0.5),
    ("-> lower limit", 1.8),
    ("hold", 0.5),
    ("lower -> upper limit", 3.0),
    ("hold", 0.5),
    ("-> home", 1.8),
    ("rest", 0.4),
]
JOINT_PERIOD = sum(d for _, d in SEGMENTS)


def load_model() -> mujoco.MjModel:
    """Compile the official MJCF with the pelvis pinned to the world.

    The cloned description repo is read-only; the freejoint is dropped from an
    in-memory MjSpec copy and nothing is written back to disk.
    """
    spec = mujoco.MjSpec.from_file(str(MJCF))
    pelvis = spec.body("pelvis")
    spec.delete(pelvis.joints[0])
    pelvis.pos = [0.0, 0.0, PINNED_PELVIS_HEIGHT]
    return spec.compile()


class JointMap:
    """Maps each actuator to the joint it drives, and to that joint's state indices."""

    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self.qadr = np.zeros(model.nu, dtype=int)  # index into data.qpos
        self.vadr = np.zeros(model.nu, dtype=int)  # index into data.qvel / qfrc_bias
        self.jntid = np.zeros(model.nu, dtype=int)
        self.name: list[str] = []
        for a in range(model.nu):
            joint = model.actuator_trnid[a, 0]
            self.jntid[a] = joint
            self.qadr[a] = model.jnt_qposadr[joint]
            self.vadr[a] = model.jnt_dofadr[joint]
            self.name.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint))
        self.lo = model.jnt_range[model.actuator_trnid[:, 0], 0].copy()
        self.hi = model.jnt_range[model.actuator_trnid[:, 0], 1].copy()

    def index(self, joint_name: str) -> int:
        return self.name.index(joint_name)


def control(model, data, jm: JointMap, target: np.ndarray) -> None:
    """Gravity-compensated PD. qfrc_bias is the gravity + Coriolis torque needed just
    to hold still, so feeding it forward lets modest gains track accurately."""
    q = data.qpos[jm.qadr]
    v = data.qvel[jm.vadr]
    tau = data.qfrc_bias[jm.vadr] + KP * (target - q) - KD * v
    data.ctrl[:] = np.clip(tau, -TAU_LIMIT, TAU_LIMIT)


def ease(u: float) -> float:
    """Cosine ease-in-out on u in [0, 1]. Zero velocity at both ends."""
    return 0.5 - 0.5 * np.cos(np.pi * np.clip(u, 0.0, 1.0))


class Tour:
    """Turns wall-clock time into a target vector, one arm joint swept at a time."""

    def __init__(self, jm: JointMap, joints, sides, speed: float):
        self.jm = jm
        self.joints = joints  # [(short_name, mirror_sign, blurb), ...]
        self.sides = sides  # ["left"], ["right"], or both
        self.speed = speed
        self.home = np.zeros(jm.model.nu)  # arms straight down, palms in
        self.period = JOINT_PERIOD / speed

    def limits(self, short: str) -> tuple[float, float]:
        """Sweep range, always read off the left joint — the driver."""
        i = self.jm.index(f"left_{short}_joint")
        return float(self.jm.lo[i]), float(self.jm.hi[i])

    def phase(self, t: float):
        """(joint_index, segment_label, driver_angle) at sim time t."""
        n = len(self.joints)
        k = int(t / self.period) % n
        u = (t / self.period) % 1.0 * JOINT_PERIOD  # seconds into this joint
        lo, hi = self.limits(self.joints[k][0])

        edge = 0.0
        for label, dur in SEGMENTS:
            if u < edge + dur:
                f = (u - edge) / dur
                if label == "settle" or label == "rest":
                    val = 0.0
                elif label == "-> lower limit":
                    val = lo * ease(f)
                elif label == "lower -> upper limit":
                    val = lo + (hi - lo) * ease(f)
                elif label == "-> home":
                    val = hi * (1.0 - ease(f))
                else:  # hold: whichever limit we just arrived at
                    val = lo if edge < JOINT_PERIOD / 2 else hi
                return k, label, val
            edge += dur
        return k, "rest", 0.0

    def target_at(self, t: float) -> np.ndarray:
        k, _, val = self.phase(t)
        short, mirror, _ = self.joints[k]
        target = self.home.copy()
        for side in self.sides:
            i = self.jm.index(f"{side}_{short}_joint")
            sign = 1.0 if side == "left" else mirror
            target[i] = sign * val
        return np.clip(target, self.jm.lo, self.jm.hi)

    def total_time(self) -> float:
        return self.period * len(self.joints)


def draw_overlay(viewer, model, data, jm: JointMap, tour: Tour, t: float) -> None:
    """A caption above the head, and an arrow on the axis currently being swept."""
    scn = viewer.user_scn
    scn.ngeom = 0
    k, label, val = tour.phase(t)
    short, _, blurb = tour.joints[k]
    lo, hi = tour.limits(short)

    head = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "head_pitch_link")]
    caption = (f"{k + 1}/{len(tour.joints)}  {short}   {np.rad2deg(val):+7.1f} deg"
               f"   [{np.rad2deg(lo):+.0f} .. {np.rad2deg(hi):+.0f}]   {label}")
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.001, 0.0, 0.0]),
                        head + np.array([0.0, 0.0, 0.45]), np.eye(3).flatten(),
                        np.array([1.0, 1.0, 1.0, 0.0]))
    g.label = caption
    scn.ngeom += 1

    # The joint's own rotation axis, in world coordinates, at its anchor.
    for side in tour.sides:
        jid = jm.jntid[jm.index(f"{side}_{short}_joint")]
        anchor = data.xanchor[jid]
        axis = data.xaxis[jid]
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3),
                            np.eye(3).flatten(), np.array([1.0, 0.55, 0.0, 0.9]))
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, 0.008,
                             anchor - 0.09 * axis, anchor + 0.09 * axis)
        scn.ngeom += 1


def run_report(model, data, jm: JointMap, tour: Tour) -> int:
    """Run the whole tour headless and print what each axis actually delivered."""
    hand = {s: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{s}_wrist_pitch_link")
            for s in tour.sides}
    n = len(tour.joints)
    reach = {s: [] for s in tour.sides}
    rows = []

    for k, (short, mirror, blurb) in enumerate(tour.joints):
        i = jm.index(f"left_{short}_joint")
        lo, hi = tour.limits(short)
        qmin, qmax, errmax, lagmax = np.inf, -np.inf, 0.0, 0.0
        t_end = tour.period * (k + 1)
        while data.time < t_end:
            tgt = tour.target_at(data.time)
            control(model, data, jm, tgt)
            mujoco.mj_step(model, data)
            q = data.qpos[jm.qadr[i]]
            qmin, qmax = min(qmin, q), max(qmax, q)
            lagmax = max(lagmax, abs(tgt[i] - q))
            if tour.phase(data.time)[1] == "hold":  # settled at a limit
                errmax = max(errmax, abs(tgt[i] - q))
            for s in tour.sides:
                reach[s].append(data.xpos[hand[s]].copy())
        rows.append((short, lo, hi, qmin, qmax, errmax, lagmax, blurb))

    print(f"MagicBot Gen1 arm ROM — {n} DOF per arm, sides: {', '.join(tour.sides)}")
    print(f"pelvis pinned at {PINNED_PELVIS_HEIGHT} m, PD KP={KP} KD={KD} "
          f"tau<={TAU_LIMIT} N*m, {tour.total_time():.1f} s of sim\n")
    print(f"{'#':>2} {'joint':<16} {'limits (deg)':>18} {'span':>7} "
          f"{'reached (deg)':>18} {'held':>6} {'lag':>6}  what it does")
    print("-" * 108)
    for k, (short, lo, hi, qmin, qmax, errmax, lagmax, blurb) in enumerate(rows):
        print(f"{k + 1:>2} {short:<16} "
              f"{np.rad2deg(lo):+8.1f} ..{np.rad2deg(hi):+7.1f} "
              f"{np.rad2deg(hi - lo):>7.1f} "
              f"{np.rad2deg(qmin):+8.1f} ..{np.rad2deg(qmax):+7.1f} "
              f"{np.rad2deg(errmax):>6.2f} {np.rad2deg(lagmax):>6.2f}  {blurb}")
    total = sum(hi - lo for _, lo, hi, _, _, _, _, _ in rows)
    print("-" * 108)
    print(f"{'':>2} {'total':<16} {'':>18} {np.rad2deg(total):>7.1f}")
    print("\nheld = tracking error once parked at a limit; lag = worst error mid-sweep,\n"
          "the PD trailing the moving setpoint. Neither is a limit of the joint.")

    print("\nhand (wrist_pitch_link) envelope swept by the tour, world frame, metres:")
    for s in tour.sides:
        p = np.array(reach[s])
        print(f"  {s:<6} x {p[:, 0].min():+.3f} .. {p[:, 0].max():+.3f}   "
              f"y {p[:, 1].min():+.3f} .. {p[:, 1].max():+.3f}   "
              f"z {p[:, 2].min():+.3f} .. {p[:, 2].max():+.3f}")
    print("\nOne axis moves at a time, so this envelope is a lower bound on the "
          "workspace,\nnot the reachable set — combined joint motion covers far more.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", choices=["left", "right", "both"], default="both",
                    help="which arm to sweep; 'both' mirrors the left command (default)")
    ap.add_argument("--joint", default=None,
                    help="sweep only this axis, e.g. elbow_roll, and repeat it")
    ap.add_argument("--speed", type=float, default=1.0, help="time scale (default 1)")
    ap.add_argument("--loop", action="store_true", help="repeat the tour until you quit")
    ap.add_argument("--report", action="store_true",
                    help="no window: run the tour and print the range-of-motion table")
    args = ap.parse_args()

    joints = ARM_JOINTS
    if args.joint:
        names = [j[0] for j in ARM_JOINTS]
        if args.joint not in names:
            ap.error(f"unknown joint {args.joint!r}; pick one of {', '.join(names)}")
        joints = [ARM_JOINTS[names.index(args.joint)]]
    sides = ["left", "right"] if args.arm == "both" else [args.arm]

    model = load_model()
    data = mujoco.MjData(model)
    jm = JointMap(model)
    tour = Tour(jm, joints, sides, args.speed)

    data.qpos[jm.qadr] = tour.home
    mujoco.mj_forward(model, data)

    if args.report:
        return run_report(model, data, jm, tour)

    print(f"model: {MJCF.name}   arms: {args.arm}   {len(joints)} axis/axes   "
          f"one pass = {tour.total_time():.1f} s")
    print("orbit: left-drag | zoom: scroll | pause: Space | quit: Esc")
    last_phase = None

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.azimuth, viewer.cam.elevation, viewer.cam.distance = 135.0, -12.0, 3.4
        viewer.cam.lookat[:] = [0.0, 0.0, 1.35]
        wall_start = time.perf_counter()
        while viewer.is_running():
            t = data.time
            if not (args.loop or args.joint) and t > tour.total_time():
                break
            control(model, data, jm, tour.target_at(t))
            mujoco.mj_step(model, data)
            draw_overlay(viewer, model, data, jm, tour, t)
            viewer.sync()

            k, label, _ = tour.phase(t)
            if (k, label) != last_phase:
                short, _, blurb = joints[k]
                print(f"  [{t:6.1f}s] {k + 1}/{len(joints)} {short:<16} {label:<22} {blurb}")
                last_phase = (k, label)

            lag = data.time - (time.perf_counter() - wall_start)
            if lag > 0:
                time.sleep(lag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
