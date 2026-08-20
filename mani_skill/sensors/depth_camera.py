# TODO (stao): reimplement this


import numpy as np
import sapien
from mani_skill.utils import common
import sapien.physx as physx
from gymnasium import spaces

from mani_skill.render import SAPIEN_RENDER_SYSTEM

if SAPIEN_RENDER_SYSTEM == "3.0":
    from sapien.sensor import StereoDepthSensor, StereoDepthSensorConfig

from mani_skill.utils import sapien_utils

from .camera import Camera, CameraConfig


class StereoDepthCameraConfig(CameraConfig):
    def __init__(self, *args, min_depth: float = 0.05, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_depth = min_depth

    @property
    def rgb_resolution(self):
        return (self.width, self.height)

    @property
    def rgb_intrinsic(self):
        fy = (self.height / 2) / np.tan(self.fov / 2)
        return np.array([[fy, 0, self.width / 2], [0, fy, self.height / 2], [0, 0, 1]])

    @classmethod
    def fromCameraConfig(cls, cfg: CameraConfig):
        return cls(**cfg.__dict__)


class StereoDepthCamera(Camera):
    def __init__(
        self,
        config: StereoDepthCameraConfig,
        scene: sapien.Scene,
        renderer_type: str = "sapien",
        articulation: physx.PhysxArticulation = None,
    ):
        self.config = config
        assert renderer_type == "sapien", renderer_type
        self.renderer_type = renderer_type

        actor_uid = config.entity_uid
        if actor_uid is None:
            self.actor = None
        else:
            if articulation is None:
                self.actor = sapien_utils.get_obj_by_name(
                    scene.get_all_actors(), actor_uid
                )
            else:
                self.actor = sapien_utils.get_obj_by_name(
                    articulation.get_links(), actor_uid
                )
            if self.actor is None:
                raise RuntimeError(f"Mount actor ({actor_uid}) is not found")

        # Add camera
        sensor_config = StereoDepthSensorConfig()
        sensor_config.rgb_resolution = config.rgb_resolution
        sensor_config.rgb_intrinsic = config.rgb_intrinsic
        sensor_config.min_depth = config.min_depth
        if self.actor is None:
            camera_mount = sapien.Entity()
            camera_mount.add_to_scene(scene.sub_scenes[0])
            camera_mount.set_pose(sapien.Pose(config.pose.p[0], config.pose.q[0]))
            self.camera = StereoDepthSensor(
                sensor_config, mount_entity=camera_mount
            )
        else:
            self.camera = StereoDepthSensor(
                sensor_config,
                mount_entity=self.actor,
                pose=sapien.Pose(config.pose.p[0], config.pose.q[0]),
            )

        # Filter texture names according to renderer type if necessary (legacy for Kuafu)
        self.texture_names = config.shader_config.texture_names

    def get_images(self, take_picture=False):
        """Get (raw) images from the camera."""
        if take_picture:
            self.camera.take_picture()

        if self.renderer_type == "client":
            return {}

        images = []
        for name in self.texture_names:
            if name == "Color":
                image = self.camera._cam_rgb.get_picture("Color")
            elif name == "depth" or "depth" in self.texture_names[name]:
                self.camera.compute_depth()
                image = self.camera.get_depth()[..., None]
            elif name == "Position":
                self.camera.compute_depth()
                position = self.camera._cam_rgb.get_picture("Position")
                depth = self.camera.get_depth()
                position[..., 2] = -depth
                image = position
            elif name == "Segmentation":
                image = self.camera._cam_rgb.get_picture("Segmentation")
            elif name == "Normal":
                image = self.camera._cam_rgb.get_picture("Normal")
            elif name == "Albedo":
                image = self.camera._cam_rgb.get_picture("Albedo")
            else:
                raise NotImplementedError(name)
            images.append(common.to_tensor(image))
        return images

    def get_params(self):
        """Get camera parameters."""
        return dict(
            extrinsic_cv=self.camera._cam_rgb.get_extrinsic_matrix(),
            cam2world_gl=self.camera._cam_rgb.get_model_matrix(),
            intrinsic_cv=self.camera._cam_rgb.get_intrinsic_matrix(),
        )

    @property
    def observation_space(self) -> spaces.Dict:
        obs_spaces = dict()
        width, height = self.camera._cam_rgb.width, self.camera._cam_rgb.height
        for name in self.texture_names:
            if name == "Color":
                obs_spaces[name] = spaces.Box(
                    low=0, high=1, shape=(height, width, 4), dtype=np.float32
                )
            elif name == "Position":
                obs_spaces[name] = spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(height, width, 4),
                    dtype=np.float32,
                )
            elif name == "Segmentation":
                obs_spaces[name] = spaces.Box(
                    low=np.iinfo(np.uint32).min,
                    high=np.iinfo(np.uint32).max,
                    shape=(height, width, 4),
                    dtype=np.uint32,
                )
            else:
                raise NotImplementedError(name)
        return spaces.Dict(obs_spaces)
