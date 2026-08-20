import os

import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat as t3d_euler2quat, quat2euler as t3d_quat2euler

from mani_skill.utils.building.actors.robotwin import get_model_id
from mani_skill.utils.geometry.rotation_conversions import (
    euler_angles_to_matrix,
    matrix_to_euler_angles,
    matrix_to_quaternion,
    quaternion_multiply,
    quaternion_to_matrix,
)
from mani_skill.utils.structs.pose import Pose as MSPose

_L1_ENABLED = False
_L1_XY_OFFSET = (0.1, 0.1)
_L2_ENABLED = False
_L3_ENABLED = False
_LR_MIRROR_ENABLED = None
_LR_MIRROR_MODE = None
_LR_MIRROR_ROBOT_POSE_ENABLED = True
_LR_MIRROR_EULER_SXYZ_REGISTRY: dict[int, tuple] = {}


def set_l1_enabled(enabled: bool) -> None:
    global _L1_ENABLED
    _L1_ENABLED = bool(enabled)


def is_l1_enabled() -> bool:
    global_val = _L1_ENABLED
    env_val = os.getenv("MANI_SKILL_L1", "<unset>")
    result = global_val or (env_val == "1")
    return result


def apply_l1_offset_xy(xy, offset=_L1_XY_OFFSET):
    if not is_l1_enabled():
        return xy
    if isinstance(xy, torch.Tensor):
        offset_tensor = torch.tensor(offset, device=xy.device, dtype=xy.dtype)
        xy = xy.clone()
        xy[..., :2] += offset_tensor
        return xy
    arr = np.array(xy, dtype=np.float32, copy=True)
    arr[..., :2] += np.array(offset, dtype=arr.dtype)
    if isinstance(xy, list):
        return arr.tolist()
    return arr

def apply_l2_offset_xy(xy, offset=_L1_XY_OFFSET):
    if not is_l2_enabled():
        return xy
    if isinstance(xy, torch.Tensor):
        offset_tensor = torch.tensor(offset, device=xy.device, dtype=xy.dtype)
        xy = xy.clone()
        xy[..., :2] += offset_tensor
        return xy
    arr = np.array(xy, dtype=np.float32, copy=True)
    arr[..., :2] += np.array(offset, dtype=arr.dtype)
    if isinstance(xy, list):
        return arr.tolist()
    return arr


def set_l2_enabled(enabled: bool) -> None:
    global _L2_ENABLED
    _L2_ENABLED = bool(enabled)


def is_l2_enabled() -> bool:
    global_val = _L2_ENABLED
    env_val = os.getenv("MANI_SKILL_L2", "<unset>")
    result = global_val or (env_val == "1")
    return result


def apply_l2_model_id(model_id, override_id=None):
    return apply_l2_ycb_model_id(model_id, override_id=override_id)


def set_l3_enabled(enabled: bool) -> None:
    global _L3_ENABLED
    _L3_ENABLED = bool(enabled)


def is_l3_enabled() -> bool:
    global_val = _L3_ENABLED
    env_val = os.getenv("MANI_SKILL_L3", "<unset>")
    result = global_val or (env_val == "1")
    return result


def set_lr_mirror_enabled(enabled: bool | None) -> None:
    """Override L2/L3 mirror behavior. None means follow L2/L3 flags."""
    global _LR_MIRROR_ENABLED
    _LR_MIRROR_ENABLED = enabled


def is_lr_mirror_enabled() -> bool:
    if _LR_MIRROR_ENABLED is not None:
        return bool(_LR_MIRROR_ENABLED)
    return is_l2_enabled()


def set_lr_mirror_robot_pose_enabled(enabled: bool | None) -> None:
    """Control whether mirror also applies to robot root poses.

    True  -> mirror robot articulations (legacy behavior)
    False -> keep robot articulations at original side while still mirroring scene objects
    None  -> reset to default (True)
    """
    global _LR_MIRROR_ROBOT_POSE_ENABLED
    if enabled is None:
        _LR_MIRROR_ROBOT_POSE_ENABLED = True
    else:
        _LR_MIRROR_ROBOT_POSE_ENABLED = bool(enabled)


def is_lr_mirror_robot_pose_enabled() -> bool:
    return bool(_LR_MIRROR_ROBOT_POSE_ENABLED)


def set_lr_mirror_mode(mode: str | None) -> None:
    """Set mirror mode: 'pos_only', 'pos_yaw', or 'full'. None resets to default."""
    global _LR_MIRROR_MODE
    if mode is None:
        _LR_MIRROR_MODE = None
        return
    if mode not in ("pos_only", "pos_yaw", "full"):
        raise ValueError(f"Invalid mirror mode: {mode}")
    _LR_MIRROR_MODE = mode


def get_lr_mirror_mode() -> str:
    if _LR_MIRROR_MODE is not None:
        return _LR_MIRROR_MODE
    return "pos_only"


def mirror_xyz(xyz):
    if not is_lr_mirror_enabled():
        return xyz
    if isinstance(xyz, torch.Tensor):
        out = xyz.clone()
        out[..., 1] = -out[..., 1]
        return out
    arr = np.array(xyz, dtype=np.float32, copy=True)
    arr[..., 1] = -arr[..., 1]
    if isinstance(xyz, list):
        return arr.tolist()
    return arr


def mirror_xy(xy):
    return mirror_xyz(xy)


def mirror_quat(q):
    def _mirror_quat_tensor(q_tensor: torch.Tensor) -> torch.Tensor:
        mirror = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]],
            device=q_tensor.device,
            dtype=q_tensor.dtype,
        )
        mat = quaternion_to_matrix(q_tensor)
        mat = mirror @ mat
        mat = mat @ mirror
        return matrix_to_quaternion(mat)

    if isinstance(q, torch.Tensor):
        return _mirror_quat_tensor(q)
    arr = np.array(q, dtype=np.float32, copy=True)
    out = _mirror_quat_tensor(torch.from_numpy(arr)).numpy()
    if isinstance(q, list):
        return out.tolist()
    return out


def mirror_quat_yaw(q):
    def _mirror_quat_tensor(q_tensor: torch.Tensor) -> torch.Tensor:
        mat = quaternion_to_matrix(q_tensor)
        # Use ZYX (yaw-pitch-roll) and flip yaw only.
        euler = matrix_to_euler_angles(mat, "ZYX")
        euler[..., 0] = -euler[..., 0]  # yaw (world Z)
        mat = euler_angles_to_matrix(euler, "ZYX")
        return matrix_to_quaternion(mat)

    if isinstance(q, torch.Tensor):
        return _mirror_quat_tensor(q)
    arr = np.array(q, dtype=np.float32, copy=True)
    out = _mirror_quat_tensor(torch.from_numpy(arr)).numpy()
    if isinstance(q, list):
        return out.tolist()
    return out


def mirror_quat_xyz_flip_z(q):
    def _mirror_quat_tensor(q_tensor: torch.Tensor) -> torch.Tensor:
        mat = quaternion_to_matrix(q_tensor)
        # Use XYZ (world axes) and flip Z only: (rx, ry, rz) -> (rx, ry, -rz).
        euler = matrix_to_euler_angles(mat, "XYZ")
        euler[..., 2] = -euler[..., 2]
        mat = euler_angles_to_matrix(euler, "XYZ")
        return matrix_to_quaternion(mat)

    if isinstance(q, torch.Tensor):
        return _mirror_quat_tensor(q)
    arr = np.array(q, dtype=np.float32, copy=True)
    out = _mirror_quat_tensor(torch.from_numpy(arr)).numpy()
    if isinstance(q, list):
        return out.tolist()
    return out


def mirror_euler_sxyz_z(euler):
    """Flip the 3rd parameter of transforms3d euler2quat(axes='sxyz')."""
    if isinstance(euler, torch.Tensor):
        out = euler.clone()
        out[..., 2] = -out[..., 2]
        return out
    arr = np.array(euler, dtype=np.float32, copy=True)
    arr[..., 2] = -arr[..., 2]
    if isinstance(euler, list):
        return arr.tolist()
    return arr


def mirror_quat_euler_sxyz_z(q, euler_override=None):
    """Mirror by flipping the 3rd parameter of transforms3d euler2quat(axes='sxyz').

    If euler_override is provided, use it directly to avoid gimbal ambiguity.
    """
    def _to_numpy(arr):
        if isinstance(arr, torch.Tensor):
            return arr.detach().cpu().numpy()
        return np.array(arr, dtype=np.float32, copy=False)

    if euler_override is not None:
        euler = mirror_euler_sxyz_z(_to_numpy(euler_override))
        q_out = t3d_euler2quat(*euler)
        if isinstance(q, torch.Tensor):
            return torch.tensor(q_out, device=q.device, dtype=q.dtype)
        if isinstance(q, list):
            return np.array(q_out, dtype=np.float32).tolist()
        return np.array(q_out, dtype=np.float32)

    arr = _to_numpy(q)
    if arr.ndim == 1:
        euler = t3d_quat2euler(arr)
        euler = mirror_euler_sxyz_z(euler)
        q_out = t3d_euler2quat(*euler)
        if isinstance(q, torch.Tensor):
            return torch.tensor(q_out, device=q.device, dtype=q.dtype)
        if isinstance(q, list):
            return np.array(q_out, dtype=np.float32).tolist()
        return np.array(q_out, dtype=np.float32)

    # Batched: loop per quaternion (transforms3d is not vectorized).
    q_list = []
    for row in arr:
        euler = t3d_quat2euler(row)
        euler = mirror_euler_sxyz_z(euler)
        q_list.append(t3d_euler2quat(*euler))
    q_out = np.stack(q_list, axis=0)
    if isinstance(q, torch.Tensor):
        return torch.tensor(q_out, device=q.device, dtype=q.dtype)
    if isinstance(q, list):
        return q_out.tolist()
    return q_out


def register_lr_mirror_euler_sxyz(obj, euler):
    """Attach a canonical sxyz Euler triple for exact Z-parameter mirroring."""
    if obj is None:
        return
    euler_val = tuple(euler)
    _LR_MIRROR_EULER_SXYZ_REGISTRY[id(obj)] = euler_val
    try:
        setattr(obj, "_lr_mirror_euler_sxyz", euler_val)
    except Exception:
        pass
    # Propagate to underlying entities for merged Actor/Articulation views.
    if hasattr(obj, "_objs"):
        for ent in obj._objs:
            _LR_MIRROR_EULER_SXYZ_REGISTRY[id(ent)] = euler_val
            try:
                setattr(ent, "_lr_mirror_euler_sxyz", euler_val)
            except Exception:
                pass


def get_lr_mirror_euler_sxyz(obj):
    """Fetch registered sxyz Euler triple for mirroring, if available."""
    if obj is None:
        return None
    euler = _LR_MIRROR_EULER_SXYZ_REGISTRY.get(id(obj))
    if euler is not None:
        return euler
    try:
        return getattr(obj, "_lr_mirror_euler_sxyz", None)
    except Exception:
        return None


def mirror_pose(pose, mode: str | None = None):
    if not is_lr_mirror_enabled():
        return pose
    mode = get_lr_mirror_mode() if mode is None else mode
    if isinstance(pose, MSPose):
        p = mirror_xyz(pose.p)
        if mode == "full":
            q = mirror_quat(pose.q)
        elif mode == "pos_yaw":
            q = mirror_quat_yaw(pose.q)
        elif mode == "pos_xyz_z":
            q = mirror_quat_xyz_flip_z(pose.q)
        else:
            q = pose.q
        return MSPose.create_from_pq(p=p, q=q, device=pose.device)
    if isinstance(pose, sapien.Pose):
        if mode == "full":
            q = mirror_quat(pose.q)
        elif mode == "pos_yaw":
            q = mirror_quat_yaw(pose.q)
        elif mode == "pos_xyz_z":
            q = mirror_quat_xyz_flip_z(pose.q)
        else:
            q = pose.q
        return sapien.Pose(p=mirror_xyz(pose.p), q=q)
    raise TypeError(f"Unsupported pose type for mirroring: {type(pose)}")


def mirror_pose_full(pose):
    """Mirror a pose across the y=0 plane, including orientation (M*R*M)."""
    return mirror_pose(pose, mode="full")


def mirror_sign(value):
    """Negate a scalar when mirror is enabled. No-op otherwise."""
    if not is_lr_mirror_enabled():
        return value
    return -value


def mirror_offset_pose(pose):
    """Mirror a local offset pose: flip y position + reflect orientation."""
    if not is_lr_mirror_enabled():
        return pose
    if isinstance(pose, MSPose):
        p = mirror_xyz(pose.p)
        q = mirror_quat(pose.q)
        return MSPose.create_from_pq(p=p, q=q, device=pose.device)
    if isinstance(pose, sapien.Pose):
        p = mirror_xyz(pose.p)
        q = mirror_quat(pose.q)
        return sapien.Pose(p=p, q=q)
    raise TypeError(f"Unsupported pose type for mirroring: {type(pose)}")


def mirror_panda_qpos_y(qpos):
    """Mirror Panda arm joint positions across the y=0 plane (heuristic).

    Assumes 7 arm joints + 2 gripper joints. Flips joints 0,2,4,6.
    """
    signs = np.array([-1, 1, -1, 1, -1, 1, -1, 1, 1], dtype=np.float32)
    if isinstance(qpos, torch.Tensor):
        out = qpos.clone()
        if out.shape[-1] >= 9:
            sign_t = torch.tensor(signs, device=out.device, dtype=out.dtype)
            out[..., :9] = out[..., :9] * sign_t
            return out
        return out * torch.tensor(signs[: out.shape[-1]], device=out.device, dtype=out.dtype)
    arr = np.array(qpos, dtype=np.float32, copy=True)
    if arr.shape[-1] >= 9:
        arr[..., :9] *= signs
    else:
        arr *= signs[: arr.shape[-1]]
    if isinstance(qpos, list):
        return arr.tolist()
    return arr


def apply_l3_robotwin_model(modelname, model_id=None, override_name=None, override_id=None):
    return _apply_robotwin_model(
        is_l3_enabled(),
        modelname,
        model_id=model_id,
        override_name=override_name,
        override_id=override_id,
    )


def apply_l2_ycb_model_id(model_id, override_id=None):
    if is_l2_enabled() and override_id:
        return override_id
    return model_id


def apply_l3_ycb_model_id(model_id, override_id=None):
    if is_l3_enabled() and override_id:
        return override_id
    return model_id


def apply_l2_robotwin_model(modelname, model_id=None, override_name=None, override_id=None):
    return _apply_robotwin_model(
        is_l2_enabled(),
        modelname,
        model_id=model_id,
        override_name=override_name,
        override_id=override_id,
    )


def apply_l3_robotwin_config(
    modelname,
    model_id=None,
    base_scale=None,
    base_replace_scale=False,
    override_name=None,
    override_id=None,
    override_scale=None,
    override_replace_scale=False,
):
    use_override = is_l3_enabled() and override_name
    if use_override:
        modelname = override_name
        model_id = override_id
        scale = override_scale
        replace_scale = override_replace_scale
    else:
        scale = base_scale
        replace_scale = base_replace_scale
    resolved_id = get_model_id(modelname, model_id=model_id)
    if resolved_id is None:
        resolved_id = model_id
    return modelname, resolved_id, scale, replace_scale


def apply_l2_robotwin_config(
    modelname,
    model_id=None,
    base_scale=None,
    base_replace_scale=False,
    override_name=None,
    override_id=None,
    override_scale=None,
    override_replace_scale=False,
):
    use_override = is_l2_enabled() and override_name
    if use_override:
        modelname = override_name
        model_id = override_id
        scale = override_scale
        replace_scale = override_replace_scale
    else:
        scale = base_scale
        replace_scale = base_replace_scale
    resolved_id = get_model_id(modelname, model_id=model_id)
    if resolved_id is None:
        resolved_id = model_id
    return modelname, resolved_id, scale, replace_scale


def apply_l2_partnet_ids(model_ids, override_ids=None):
    if is_l2_enabled() and override_ids:
        return np.array(override_ids)
    return np.array(model_ids)


def apply_l3_partnet_ids(model_ids, override_ids=None):
    if is_l3_enabled() and override_ids:
        return np.array(override_ids)
    return np.array(model_ids)


def apply_l2_quat_offset(q, offset_q=None):
    if not is_l2_enabled() or offset_q is None:
        return q
    if isinstance(q, torch.Tensor):
        offset = torch.tensor(offset_q, device=q.device, dtype=q.dtype)
        if offset.ndim == 1 and q.ndim > 1:
            offset = offset.expand(q.shape[0], -1)
        return quaternion_multiply(q, offset)
    arr = np.asarray(q, dtype=np.float32)
    offset = np.asarray(offset_q, dtype=arr.dtype)
    q_tensor = torch.from_numpy(arr)
    offset_tensor = torch.from_numpy(offset)
    if offset_tensor.ndim == 1 and q_tensor.ndim > 1:
        offset_tensor = offset_tensor.expand(q_tensor.shape[0], -1)
    out = quaternion_multiply(q_tensor, offset_tensor).numpy()
    if isinstance(q, list):
        return out.tolist()
    return out


def _apply_robotwin_model(
    enabled, modelname, model_id=None, override_name=None, override_id=None
):
    if enabled and override_name:
        modelname = override_name
        model_id = override_id
    resolved_id = get_model_id(modelname, model_id=model_id)
    if resolved_id is None:
        return modelname, model_id
    return modelname, resolved_id