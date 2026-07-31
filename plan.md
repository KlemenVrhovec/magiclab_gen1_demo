# Plan — MagicBot Gen1 basic MuJoCo simulation

Derived from [claude_code_prompt.md](claude_code_prompt.md). Scope: load the official
MagicLab Gen1 model in MuJoCo and see it react to gravity in an interactive viewer.
No controller, no locomotion, no RL, no ROS2.

## Environment check (done up front)

| Requirement | Found | OK |
|---|---|---|
| Python 3.10+ | 3.14.6 (`C:\Python314`; 3.13 also present via Store) | yes |
| git | 2.52.0.windows.1 | yes |
| pip | 26.1.2 | yes |
| mujoco wheel for this interpreter | resolves to 3.11.0 | yes |

No Python-version fallback needed. Working dir started with only `.idea/` and the prompt file.

## Steps

1. **Project setup** — create `magicbot_sim/` inside
   `c:\Work\6_MagicLab_Gen1_Simulation` (the prompt allows "wherever we're working",
   so not the home dir). Create a venv at `magicbot_sim\.venv` and upgrade pip inside it.

2. **Install MuJoCo** — `pip install mujoco` in the venv only, no other packages.
   Record the resolved version.

3. **Clone the model** — `git clone --depth 1
   https://github.com/MagiclabRobotics/magicbot-gen1_description.git` into
   `magicbot_sim/`. Treat it as read-only from then on — no edits to anything inside it.
   Verify `mjcf/MAGICBOT.xml` and `meshes/` landed together.

4. **Headless sanity check before opening any window** — load the model in Python and
   compare against the expected `nq: 37  nv: 36  njnt: 31  nbody: 34  nu: 30`.
   If any number differs: stop and investigate rather than continuing to the viewer.
   Likely causes would be a partial clone (missing meshes) or an MJCF/mujoco
   version mismatch.

5. **Launch the interactive viewer** —
   `python -m mujoco.viewer --mjcf=magicbot-gen1_description/mjcf/MAGICBOT.xml`,
   started in the background so it doesn't block, then check stderr for a clean start.

6. **Hand off for acceptance** — confirm the first two acceptance criteria
   (opens with no errors, loads as a 31-joint humanoid). The other two
   (orbit/zoom the camera, ctrl+right-drag to poke a body) can only be confirmed at
   the physical display, so those are for the user to verify in the open window.

## Notes

- The robot sagging/collapsing under gravity is expected and correct here — no
  controller will be added.
- If pip or the viewer fails: retry once with a fresh venv and upgraded pip, then
  report the verbatim error rather than guessing around it.

## Acceptance criteria (from the prompt)

- [x] Viewer window opens with no errors — window title `MuJoCo : Magicbot-Gen1`,
      empty stdout/stderr, render process actively consuming CPU
- [x] Model geometry visibly matches a humanoid — 31 joints / 34 bodies / 30 actuators
      across legs, torso, head and arms; exact match to the expected dimensions
- [x] Robot visibly responds to gravity/physics over time — headless 2 s rollout drops
      the floating-base root from z=0.950 m to z=0.239 m under gravity [0, 0, -9.81]
- [ ] Camera can be orbited/zoomed and the model nudged with the mouse — **for the user
      to confirm at the physical display**

## Results

- venv: `magicbot_sim\.venv` (Python 3.14.6), pip 26.2
- mujoco **3.11.0** — native `cp314` wheel, no fallback interpreter needed
- model repo cloned at depth 1; `mjcf/MAGICBOT.xml` + 56 meshes; left unmodified
- sanity check output: `nq: 37  nv: 36  njnt: 31  nbody: 34  nu: 30` (exact match)

Relaunch command (from `magicbot_sim\`):

```
.\.venv\Scripts\python.exe -m mujoco.viewer --mjcf=magicbot-gen1_description\mjcf\MAGICBOT.xml
```

Note: `mujoco.viewer` spawns a child process that owns the actual window — the parent
stays near-idle, so check the child when verifying the viewer is alive.
