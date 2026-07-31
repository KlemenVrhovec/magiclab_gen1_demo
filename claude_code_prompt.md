# Task: Basic physics simulation of the MagicLab MagicBot Gen1

## Goal
Get the official MagicBot Gen1 robot model loaded and running in MuJoCo on this
machine (Windows laptop), so I can see it as a physically accurate 3D model
reacting to gravity in an interactive viewer window. This is the "most basic"
tier — no walking, no balance controller, no RL yet. Just prove the real
model loads and behaves like real physics.

## Context
MagicLab (the company making the robot we're buying) publishes an official,
open-source robot description for the Gen1 on GitHub, under the
`MagiclabRobotics` organization. It's BSD-3-Clause licensed. Use it directly —
do not build a placeholder or substitute model.

Repo: https://github.com/MagiclabRobotics/magicbot-gen1_description

It contains:
- `mjcf/MAGICBOT.xml` — the MuJoCo model (use this one)
- `urdf/MAGICBOT.urdf` and `urdf/MAGICBOT_with_hand.urdf` — URDF versions (not needed for this task)
- `meshes/` — STL mesh files the MJCF references (must stay alongside the xml)
- `rviz/`, `launch/` — ROS2/RViz visualization support (not needed for this task)

## Environment
- Windows 10/11 laptop, real display attached (not headless)
- Need: Python 3.10+, pip, Git
- No GPU or CUDA needed — MuJoCo runs on CPU and uses standard OpenGL for the
  viewer window, so integrated graphics is fine.

## Steps

1. Check prerequisites: confirm `python --version` is 3.10+, `git --version`
   works, and `pip` is available. Install/upgrade anything missing.

2. Create a project folder (e.g. `magicbot_sim` in my home directory or
   wherever we're working) and set up a Python virtual environment inside it.

3. Inside the venv, run `pip install mujoco` (only dependency needed — no
   ROS2, no Isaac Lab, no other packages for this basic step).

4. Clone the model repo into the project folder:
   `git clone --depth 1 https://github.com/MagiclabRobotics/magicbot-gen1_description.git`

5. Sanity-check the model loads correctly before opening any window. Run a
   short Python snippet that does:
   ```python
   import mujoco
   m = mujoco.MjModel.from_xml_path("magicbot-gen1_description/mjcf/MAGICBOT.xml")
   print("nq:", m.nq, "nv:", m.nv, "njnt:", m.njnt, "nbody:", m.nbody, "nu:", m.nu)
   ```
   Expected output: `nq: 37 nv: 36 njnt: 31 nbody: 34 nu: 30` (30 actuated
   joints across legs, torso, head, and arms). If these numbers don't match,
   something is wrong with the clone or the mujoco version — stop and
   investigate before continuing.

6. Launch the real interactive viewer (this needs the real Windows desktop,
   not a headless/remote session):
   `python -m mujoco.viewer --mjcf=magicbot-gen1_description/mjcf/MAGICBOT.xml`

   A window should open showing the Gen1 humanoid model. With no controller
   active, it will sag/collapse under gravity rather than stand — that's
   expected and correct for this basic step, not a bug. You can left-drag to
   orbit the camera, scroll to zoom, and ctrl+right-drag on a body part to
   poke it with a force and see it react physically.

## Acceptance criteria
- Viewer window opens with no errors
- Model geometry visibly matches a humanoid (head, torso, two arms, two legs,
  hands)
- Robot visibly responds to gravity/physics over time (settles or sags)
- I can orbit/zoom the camera and nudge the model with the mouse

## Explicitly out of scope for this task (don't build unless I ask)
- Any standing/balance controller (PD control, etc.)
- Walking or locomotion policies
- Isaac Lab / reinforcement learning training (`magiclab_rl_lab`)
- ROS2 / RViz integration
- Modifying any files inside the cloned `magicbot-gen1_description` repo

If `pip install mujoco` or the viewer fails on Windows, first try a fresh
virtual environment and an upgraded `pip` before troubleshooting further, and
tell me the exact error rather than guessing around it.
