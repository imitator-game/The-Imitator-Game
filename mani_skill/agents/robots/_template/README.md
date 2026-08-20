# YourRobot — Minimal Robot Agent Template

Copy this folder to `mani_skill/agents/robots/your_robot/` and edit the fields
marked **EDIT ME** to import your own robot from a URDF.

```
mani_skill/agents/robots/your_robot/
├── __init__.py       # from .your_robot import YourRobot
├── your_robot.py     # the agent class
└── README.md
```

Robot assets go under:

```
mani_skill/assets/robots/your_robot/
├── your_robot.urdf
└── meshes/...
```

## What to edit (only these)

```python
uid = "your_robot_uid"                                   # unique robot name
urdf_path = f"{ASSET_DIR}/robots/your_robot/your_robot.urdf"

keyframes = dict(rest=Keyframe(pose=sapien.Pose(),       # default state
                               qpos=np.array([...])))    # arm+then+gripper order

arm_joint_names     = ["..."]        # arm joint names from the URDF
gripper_joint_names = ["..."]        # gripper joint names ([] if none)
ee_link_name        = "..."          # TCP / end-effector link
finger_link_names   = ("...", "...") # (None, None) if no gripper

gripper_lower, gripper_upper = 0.0, 0.04   # gripper action range (URDF limits)
use_mimic_gripper = True                   # False for independently-driven fingers
```

No gripper? Use `gripper_joint_names=[]` and `finger_link_names=(None, None)` and
remove the gripper values from `qpos`.

## Usage

```python
import gymnasium as gym
import mani_skill.envs
import mani_skill.agents.robots.your_robot   # import runs @register_agent()

env = gym.make(
    "PickCube-v1",
    robot_uids="your_robot_uid",             # matches uid above
    obs_mode="state",
    control_mode="pd_joint_delta_pos",       # or pd_joint_pos / pd_ee_delta_pose
)
```

## Common issues

- **Robot not found** → the module must be imported so `@register_agent()` runs.
- **URDF not found** → check `urdf_path` and the asset folder layout.
- **Invalid joint/link name** → names must exactly match the URDF.
- **Robot moves oddly** → check active-joint order, `qpos` length, and stiffness.
- **EE control fails** → start with `pd_joint_delta_pos`, then check `ee_link_name`.
