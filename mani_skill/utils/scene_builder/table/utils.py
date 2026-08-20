import sapien
import sapien.core as sapien
import numpy as np
from pathlib import Path
import transforms3d as t3d
import sapien.physx as sapienp
import json
import os, re
import os.path as osp
from mani_skill.utils.structs.robotwin_actor import Actor, ArticulationActor
from mani_skill import ASSET_DIR

def _transform_point(self, origin_base: sapien.Pose, new_base: sapien.Pose, origin_pos: sapien.Pose):
    from scipy.spatial.transform import Rotation
    rotation_origin = Rotation.from_quat([origin_base.q[1], origin_base.q[2], origin_base.q[3], origin_base.q[0]])
    rotation_origin_new = Rotation.from_quat([new_base.q[1], new_base.q[2], new_base.q[3], new_base.q[0]])
    relative_pos = origin_pos.p - origin_base.p
    rotation_point = Rotation.from_quat([origin_pos.q[1], origin_pos.q[2], origin_pos.q[3], origin_pos.q[0]])
    relative_rot = rotation_origin.inv() * rotation_point
    relative_pos_rotated = rotation_origin_new.apply(rotation_origin.inv().apply(relative_pos))
    position_point_new = new_base.p + relative_pos_rotated
    rotation_point_new = rotation_origin_new * relative_rot
    quaternion_point_new = rotation_point_new.as_quat()  # Returns (x,y,z,w)
    quaternion_point_new = np.array([quaternion_point_new[3], quaternion_point_new[0], quaternion_point_new[1], quaternion_point_new[2]])
    new_pos = sapien.Pose(
        p=position_point_new,
        q=quaternion_point_new
    )
    return new_pos



class UnStableError(Exception):

    def __init__(self, msg):
        super().__init__(msg)

# create box
def create_entity_box(
    scene,
    pose: sapien.Pose,
    half_size,
    color=None,
    is_static=False,
    name="",
    texture_id=None,
) -> sapien.Entity:

    entity = sapien.Entity()
    entity.set_name(name)
    entity.set_pose(pose)

    # create PhysX dynamic rigid body
    rigid_component = (sapien.physx.PhysxRigidDynamicComponent()
                       if not is_static else sapien.physx.PhysxRigidStaticComponent())
    rigid_component.attach(
        sapien.physx.PhysxCollisionShapeBox(half_size=half_size, material=None))

    # Add texture
    if texture_id is not None:

        # test for both .png and .jpg
        texturepath = str(Path.home() / ".maniskill" / "data" / "robotwin" / "background_texture"/ f"{texture_id}.png")
        # create texture from file
        texture2d = sapien.render.RenderTexture2D(texturepath)
        material = sapien.render.RenderMaterial()
        material.set_base_color_texture(texture2d)
        # renderer.create_texture_from_file(texturepath)
        # material.set_diffuse_texture(texturepath)
        material.base_color = [1, 1, 1, 1]
        material.metallic = 0.1
        material.roughness = 0.3
    else:
        model_dir = Path(osp.dirname(__file__)) / "assets"
        texturepath = str(model_dir / "wall.png")
        texture2d = sapien.render.RenderTexture2D(texturepath)
        material = sapien.render.RenderMaterial()
        material.set_base_color_texture(texture2d)
        # renderer.create_texture_from_file(texturepath)
        # material.set_diffuse_texture(texturepath)
        material.base_color = [1, 1, 1, 1]
        material.metallic = 0.1
        material.roughness = 0.3

    # create render body for visualization
    render_component = sapien.render.RenderBodyComponent()
    render_component.attach(
        # add a box visual shape with given size and rendering material
        sapien.render.RenderShapeBox(half_size, material))

    entity.add_component(rigid_component)
    entity.add_component(render_component)
    entity.set_pose(pose)

    # in general, entity should only be added to scene after it is fully built
    for sub_scene in scene.sub_scenes:
        sub_scene.add_entity(entity)
    return entity


def create_box(
    scene,
    pose: sapien.Pose,
    half_size,
    color=None,
    is_static=False,
    name="",
    texture_id=None,
    boxtype="default",
) -> Actor:
    entity = create_entity_box(
        scene=scene,
        pose=pose,
        half_size=half_size,
        color=color,
        is_static=is_static,
        name=name,
        texture_id=texture_id,
    )
    if boxtype == "default":
        data = {
            "center": [0, 0, 0],
            "extents":
            half_size,
            "scale":
            half_size,
            "target_pose": [[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]]],
            "contact_points_pose": [
                [
                    [0, 0, 1, 0],
                    [1, 0, 0, 0],
                    [0, 1, 0, 0.0],
                    [0, 0, 0, 1],
                ],  # top_down(front)
                [
                    [1, 0, 0, 0],
                    [0, 0, -1, 0],
                    [0, 1, 0, 0.0],
                    [0, 0, 0, 1],
                ],  # top_down(right)
                [
                    [-1, 0, 0, 0],
                    [0, 0, 1, 0],
                    [0, 1, 0, 0.0],
                    [0, 0, 0, 1],
                ],  # top_down(left)
                [
                    [0, 0, -1, 0],
                    [-1, 0, 0, 0],
                    [0, 1, 0, 0.0],
                    [0, 0, 0, 1],
                ],  # top_down(back)
                # [[0, 0, 1, 0], [0, -1, 0, 0], [1, 0, 0, 0.0], [0, 0, 0, 1]], # front
                # [[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0.0], [0, 0, 0, 1]], # right
                # [[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0.0], [0, 0, 0, 1]], # left
                # [[0, 0, -1, 0], [0, 1, 0, 0], [1, 0, 0, 0.0], [0, 0, 0, 1]], # back
            ],
            "transform_matrix":
            np.eye(4).tolist(),
            "functional_matrix": [
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, -1.0, 0, 0.0],
                    [0.0, 0, -1.0, -1],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, -1.0, 0, 0.0],
                    [0.0, 0, -1.0, 1],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            ],  # functional points matrix
            "contact_points_description": [],  # contact points description
            "contact_points_group": [[0, 1, 2, 3], [4, 5, 6, 7]],
            "contact_points_mask": [True, True],
            "target_point_description": ["The center point on the bottom of the box."],
        }
    else:
        data = {
            "center": [0, 0, 0],
            "extents":
            half_size,
            "scale":
            half_size,
            "target_pose": [[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]]],
            "contact_points_pose": [
                [[0, 0, 1, 0], [0, -1, 0, 0], [1, 0, 0, 0.7], [0, 0, 0, 1]],  # front
                [[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0.7], [0, 0, 0, 1]],  # right
                [[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0.7], [0, 0, 0, 1]],  # left
                [[0, 0, -1, 0], [0, 1, 0, 0], [1, 0, 0, 0.7], [0, 0, 0, 1]],  # back
                [[0, 0, 1, 0], [0, -1, 0, 0], [1, 0, 0, -0.7], [0, 0, 0, 1]],  # front
                [[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, -0.7], [0, 0, 0, 1]],  # right
                [[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, -0.7], [0, 0, 0, 1]],  # left
                [[0, 0, -1, 0], [0, 1, 0, 0], [1, 0, 0, -0.7], [0, 0, 0, 1]],  # back
            ],
            "transform_matrix":
            np.eye(4).tolist(),
            "functional_matrix": [
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, -1.0, 0, 0.0],
                    [0.0, 0, -1.0, -1.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, -1.0, 0, 0.0],
                    [0.0, 0, -1.0, 1.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            ],  # functional points matrix
            "contact_points_description": [],  # contact points description
            "contact_points_group": [[0, 1, 2, 3, 4, 5, 6, 7]],
            "contact_points_mask": [True, True],
            "target_point_description": ["The center point on the bottom of the box."],
        }
    return Actor(entity, data)


# create spere
def create_sphere(
    scene,
    pose: sapien.Pose,
    radius: float,
    color=None,
    is_static=False,
    name="",
    texture_id=None,
) -> sapien.Entity:

    entity = sapien.Entity()
    entity.set_name(name)
    entity.set_pose(pose)

    # create PhysX dynamic rigid body
    rigid_component = (sapien.physx.PhysxRigidDynamicComponent()
                       if not is_static else sapien.physx.PhysxRigidStaticComponent())
    rigid_component.attach(
        sapien.physx.PhysxCollisionShapeSphere(radius=radius, material=scene.default_physical_material))

    # Add texture
    if texture_id is not None:

        # test for both .png and .jpg
        texturepath = str(Path.home() / ".maniskill" / "data" / "robotwin" .join(f"textures/{texture_id}.png"))
        # create texture from file
        texture2d = sapien.render.RenderTexture2D(texturepath)
        material = sapien.render.RenderMaterial()
        material.set_base_color_texture(texture2d)
        # renderer.create_texture_from_file(texturepath)
        # material.set_diffuse_texture(texturepath)
        material.base_color = [1, 1, 1, 1]
        material.metallic = 0.1
        material.roughness = 0.3
    else:
        material = sapien.render.RenderMaterial(base_color=[*color[:3], 1])

    # create render body for visualization
    render_component = sapien.render.RenderBodyComponent()
    render_component.attach(
        # add a box visual shape with given size and rendering material
        sapien.render.RenderShapeSphere(radius=radius, material=material))

    entity.add_component(rigid_component)
    entity.add_component(render_component)
    entity.set_pose(pose)

    # in general, entity should only be added to scene after it is fully built
    scene.add_entity(entity)
    return entity


# create cylinder
def create_cylinder(
    scene,
    pose: sapien.Pose,
    radius: float,
    half_length: float,
    color=None,
    name="",
) -> sapien.Entity:

    entity = sapien.Entity()
    entity.set_name(name)
    entity.set_pose(pose)

    # create PhysX dynamic rigid body
    rigid_component = sapien.physx.PhysxRigidDynamicComponent()
    rigid_component.attach(
        sapien.physx.PhysxCollisionShapeCylinder(
            radius=radius,
            half_length=half_length,
            material=scene.default_physical_material,
        ))

    # create render body for visualization
    render_component = sapien.render.RenderBodyComponent()
    render_component.attach(
        # add a box visual shape with given size and rendering material
        sapien.render.RenderShapeCylinder(
            radius=radius,
            half_length=half_length,
            material=sapien.render.RenderMaterial(base_color=[*color[:3], 1]),
        ))

    entity.add_component(rigid_component)
    entity.add_component(render_component)
    entity.set_pose(pose)

    # in general, entity should only be added to scene after it is fully built
    scene.add_entity(entity)
    return entity


# create box
def create_visual_box(
    scene,
    pose: sapien.Pose,
    half_size,
    color=None,
    name="",
) -> sapien.Entity:

    entity = sapien.Entity()
    entity.set_name(name)
    entity.set_pose(pose)

    # create render body for visualization
    render_component = sapien.render.RenderBodyComponent()
    render_component.attach(
        # add a box visual shape with given size and rendering material
        sapien.render.RenderShapeBox(half_size, sapien.render.RenderMaterial(base_color=[*color[:3], 1])))

    entity.add_component(render_component)
    entity.set_pose(pose)

    # in general, entity should only be added to scene after it is fully built
    scene.add_entity(entity)
    return entity


def create_table(
        scene,
        pose: sapien.Pose,
        length: float,
        width: float,
        height: float,
        thickness=0.1,
        color=(1, 1, 1),
        name="table",
        is_static=True,
        texture_id=None,
) -> sapien.Entity:
    """Create a table with specified dimensions."""

    builder = scene.create_actor_builder()

    if is_static:
        builder.set_physx_body_type("static")
    else:
        builder.set_physx_body_type("dynamic")

    tabletop_pose = sapien.Pose([0.0, 0.0, height])
    tabletop_half_size = [length / 2, width / 2, thickness / 2]
    builder.add_box_collision(
        pose=tabletop_pose,
        half_size=tabletop_half_size,
        material=None,
    )

    # Add texture
    if texture_id is not None:
        texturepath = str(Path.home() / ".maniskill" / "data" / "robotwin" / f"background_texture/{texture_id}.png")

        if not Path(texturepath).exists():
            texturepath = str(Path.home() / ".maniskill" / "data" / "robotwin" / f"background_texture/{texture_id}.jpg")

        texture2d = sapien.render.RenderTexture2D(texturepath)
        material = sapien.render.RenderMaterial()
        material.set_base_color_texture(texture2d)
        material.base_color = [1, 1, 1, 1]
        material.metallic = 0.1
        material.roughness = 0.3
        builder.add_box_visual(pose=tabletop_pose, half_size=tabletop_half_size, material=material)
    else:
        builder.add_box_visual(
            pose=tabletop_pose,
            half_size=tabletop_half_size,
            material=color,
        )

    leg_offset = 0.03  # inset distance of the table legs from the tabletop edge
    leg_thickness = min(thickness, 0.08)  # thickness of the table legs, no more than 0.08

    leg_height = height - thickness / 2  # actual height of the table legs
    leg_center_z = leg_height / 2  # z coordinate of the leg center
    leg_half_height = leg_height / 2  # half height of the table legs

    for i in [-1, 1]:
        for j in [-1, 1]:
            x = i * (length / 2 - leg_offset - leg_thickness / 2)
            y = j * (width / 2 - leg_offset - leg_thickness / 2)

            table_leg_pose = sapien.Pose([x, y, leg_center_z])
            table_leg_half_size = [leg_thickness / 2, leg_thickness / 2, leg_half_height]

            builder.add_box_collision(pose=table_leg_pose, half_size=table_leg_half_size)
            builder.add_box_visual(pose=table_leg_pose, half_size=table_leg_half_size, material=color)

    builder.initial_pose = pose
    table = builder.build_kinematic(name=name)
    return table


# create obj model
def create_obj(
        scene,
        pose: sapien.Pose,
        modelname: str,
        scale=(1, 1, 1),
        convex=False,
        is_static=False,
        model_id=None,
        no_collision=False,
) -> Actor:

    modeldir = Path("assets/objects") / modelname
    if model_id is None:
        file_name = modeldir / "textured.obj"
        json_file_path = modeldir / "model_data.json"
    else:
        file_name = modeldir / f"textured{model_id}.obj"
        json_file_path = modeldir / f"model_data{model_id}.json"

    try:
        with open(json_file_path, "r") as file:
            model_data = json.load(file)
        scale = model_data["scale"]
    except:
        model_data = None

    builder = scene.create_actor_builder()
    builder.initial_pose = pose
    if is_static:
        builder.set_physx_body_type("static")
    else:
        builder.set_physx_body_type("dynamic")

    if not no_collision:
        if convex == True:
            builder.add_multiple_convex_collisions_from_file(filename=str(file_name), scale=scale)
        else:
            builder.add_nonconvex_collision_from_file(filename=str(file_name), scale=scale)

    builder.add_visual_from_file(filename=str(file_name), scale=scale)
    mesh = builder.build(name=modelname)
    mesh.set_pose(pose)

    return Actor(mesh, model_data)


# create glb model
def create_glb(
        scene,
        pose: sapien.Pose,
        modelname: str,
        scale=(1, 1, 1),
        convex=False,
        is_static=False,
        model_id=None,
) -> Actor:

    modeldir = Path("objects") / modelname
    if model_id is None:
        file_name = modeldir / "base.glb"
        json_file_path = modeldir / "model_data.json"
    else:
        file_name = modeldir / f"base{model_id}.glb"
        json_file_path = modeldir / f"model_data{model_id}.json"

    try:
        with open(json_file_path, "r") as file:
            model_data = json.load(file)
        scale = model_data["scale"]
    except:
        model_data = None

    builder = scene.create_actor_builder()
    if is_static:
        builder.set_physx_body_type("static")
    else:
        builder.set_physx_body_type("dynamic")

    if convex == True:
        builder.add_multiple_convex_collisions_from_file(filename=str(file_name), scale=scale)
    else:
        builder.add_nonconvex_collision_from_file(
            filename=str(file_name),
            scale=scale,
        )

    builder.add_visual_from_file(filename=str(file_name), scale=scale)
    mesh = builder.build(name=modelname)
    mesh.set_pose(pose)

    return Actor(mesh, model_data)


def get_glb_or_obj_file(modeldir, model_id):
    modeldir = Path(modeldir)
    if model_id is None:
        file = modeldir / "base.glb"
    else:
        file = modeldir / f"base{model_id}.glb"
    if not file.exists():
        if model_id is None:
            file = modeldir / "textured.obj"
        else:
            file = modeldir / f"textured{model_id}.obj"
    return file


def create_actor(
        scene,
        pose: sapien.Pose,
        modelname: str,
        scale=None,
        replace_scale=False,
        convex=False,
        is_static=False,
        model_id=0,
        _idx_if_repeat=0,
        mass=None,
        name=None,
) -> Actor:

    modeldir = ASSET_DIR / "robotwin" / "objects" / modelname

    if model_id is None:
        json_file_path = modeldir / "model_data.json"
    else:
        json_file_path = modeldir / f"model_data{model_id}.json"

    collision_file = ""
    visual_file = ""
    if (modeldir / "collision").exists():
        collision_file = get_glb_or_obj_file(modeldir / "collision", model_id)
    if collision_file == "" or not collision_file.exists():
        collision_file = get_glb_or_obj_file(modeldir, model_id)

    if (modeldir / "visual").exists():
        visual_file = get_glb_or_obj_file(modeldir / "visual", model_id)
    if visual_file == "" or not visual_file.exists():
        visual_file = get_glb_or_obj_file(modeldir, model_id)

    if not collision_file.exists() or not visual_file.exists():
        print(modelname, "is not exist model file!")
        return None

    with open(json_file_path, "r") as file:
        model_data = json.load(file)
    if scale is None:
        if 'scale' in model_data.keys():
            scale = model_data["scale"]
        else:
            scale = (0.08, 0.08, 0.08)
    else:
        if replace_scale:
            scale = scale
        else:
            if 'scale' in model_data.keys():
                scale = (model_data["scale"][0] * scale[0],
                         model_data["scale"][1] * scale[1],
                         model_data["scale"][2] * scale[2])
            else:
                scale = (0.08 * scale[0],
                         0.08 * scale[1],
                         0.08 * scale[2])

    builder = scene.create_actor_builder()
    builder.initial_pose = pose
    if is_static:
        builder.set_physx_body_type("static")
    else:
        builder.set_physx_body_type("dynamic")

    if convex == True:
        builder.add_multiple_convex_collisions_from_file(filename=str(collision_file), scale=scale)
    else:
        builder.add_nonconvex_collision_from_file(
            filename=str(collision_file),
            scale=scale,
        )

    builder.add_visual_from_file(filename=str(visual_file), scale=scale)
    if name is not None:
        build_name = name
    elif _idx_if_repeat != 0:
        build_name = f"{modelname}_{_idx_if_repeat}"
    else:
        build_name = modelname
    mesh = builder.build(name=build_name)
    mesh.set_pose(pose)

    # FIX A BUG
    mesh.initial_pose = pose

    return Actor(mesh, model_data) if mass is None else Actor(mesh, model_data, mass=mass)

# create urdf model
def create_urdf_obj(scene, pose: sapien.Pose, modelname: str, scale=1.0, fix_root_link=True) -> ArticulationActor:

    modeldir = Path("objects") / modelname
    json_file_path = modeldir / "model_data.json"
    loader: sapien.URDFLoader = scene.create_urdf_loader()
    loader.scale = scale

    try:
        with open(json_file_path, "r") as file:
            model_data = json.load(file)
        loader.scale = model_data["scale"][0]
    except:
        model_data = None

    loader.fix_root_link = fix_root_link
    loader.load_multiple_collisions_from_file = True
    object: sapien.Articulation = loader.load(str(modeldir / "mobility.urdf"))

    object.set_root_pose(pose)
    object.set_name(modelname)
    return ArticulationActor(object, model_data)


def create_sapien_urdf_obj(
    scene,
    pose: sapien.Pose,
    modelname: str,
    scale=1.0,
    modelid: int = None,
    fix_root_link=False,
) -> ArticulationActor:

    modeldir = Path("assets") / "objects" / modelname
    if modelid is not None:
        model_list = [model for model in modeldir.iterdir() if model.is_dir() and model.name != "visual"]

        def extract_number(filename):
            match = re.search(r"\d+", filename.name)
            return int(match.group()) if match else 0

        model_list = sorted(model_list, key=extract_number)

        if modelid >= len(model_list):
            is_find = False
            for model in model_list:
                if modelid == int(model.name):
                    modeldir = model
                    is_find = True
                    break
            if not is_find:
                raise ValueError(f"modelid {modelid} is out of range for {modelname}.")
        else:
            modeldir = model_list[modelid]
    json_file = modeldir / "model_data.json"

    if json_file.exists():
        with open(json_file, "r") as file:
            model_data = json.load(file)
        scale = model_data["scale"]
        trans_mat = np.array(model_data.get("transform_matrix", np.eye(4)))
    else:
        model_data = None
        trans_mat = np.eye(4)

    loader: sapien.URDFLoader = scene.create_urdf_loader()
    loader.scale = scale
    loader.fix_root_link = fix_root_link
    loader.load_multiple_collisions_from_file = True
    object = loader.load_multiple(str(modeldir / "mobility.urdf"))[0][0]

    pose_mat = pose.to_transformation_matrix()
    pose = sapien.Pose(
        p=pose_mat[:3, 3] + trans_mat[:3, 3],
        q=t3d.quaternions.mat2quat(trans_mat[:3, :3] @ pose_mat[:3, :3]),
    )
    object.set_pose(pose)

    if model_data is not None:
        if "init_qpos" in model_data and len(model_data["init_qpos"]) > 0:
            object.set_qpos(np.array(model_data["init_qpos"]))
        if "mass" in model_data and len(model_data["mass"]) > 0:
            for link in object.get_links():
                link.set_mass(model_data["mass"].get(link.get_name(), 0.1))

        bounding_box_file = modeldir / "bounding_box.json"
        if bounding_box_file.exists():
            bounding_box = json.load(open(bounding_box_file, "r", encoding="utf-8"))
            model_data["extents"] = (np.array(bounding_box["max"]) - np.array(bounding_box["min"])).tolist()
    object.set_name(modelname)
    return ArticulationActor(object, model_data)


def create_ground(
        scene,
        floor_width: int = 100,
        floor_length: int = None,
        xy_origin: tuple = (0, 0),
        altitude: float = 0,
        texture_id: str = None,
        texture_square_len: float = 8,  # meters covered by each texture tile
        mipmap_levels: int = 1,
        name: str = "ground",
        add_collision: bool = True,
):
    ground = scene.create_actor_builder()

    # Add collision plane
    if add_collision:
        ground.add_plane_collision(
            sapien.Pose(p=[0, 0, altitude], q=[0.7071068, 0, -0.7071068, 0]),
        )

    ground.initial_pose = sapien.Pose(p=[0, 0, 0], q=[1, 0, 0, 0])

    if scene.parallel_in_single_scene:
        ground.set_scene_idxs([0])

    actor = ground.build_static(name=name)

    if scene.can_render():
        # Prepare the texture material
        if texture_id is not None:
            texturepath = str(Path.home() / ".maniskill" / "data" / "robotwin" /
                              "background_texture" / f"{texture_id}.png")
        else:
            model_dir = Path(osp.dirname(__file__)) / "assets"
            texturepath = str(model_dir / "ground.png")

        mat = sapien.render.RenderMaterial()
        mat.base_color_texture = sapien.render.RenderTexture2D(
            filename=texturepath,
            mipmap_levels=mipmap_levels,
        )
        mat.base_color = [1, 1, 1, 1]
        mat.metallic = 0.1
        mat.roughness = 0.3

        # Generate mesh vertices
        floor_length = floor_width if floor_length is None else floor_length
        num_verts = (floor_width + 1) * (floor_length + 1)
        vertices = np.zeros((num_verts, 3))
        floor_half_width = floor_width / 2
        floor_half_length = floor_length / 2

        xrange = np.arange(start=-floor_half_width, stop=floor_half_width + 1)
        yrange = np.arange(start=-floor_half_length, stop=floor_half_length + 1)
        xx, yy = np.meshgrid(xrange, yrange)
        xys = np.stack((yy, xx), axis=2).reshape(-1, 2)

        vertices[:, 0] = xys[:, 0] + xy_origin[0]
        vertices[:, 1] = xys[:, 1] + xy_origin[1]
        vertices[:, 2] = altitude

        # Normals (pointing upward)
        normals = np.zeros((len(vertices), 3))
        normals[:, 2] = 1

        # UV coordinates (key: control texture repetition)
        uv_scale = floor_width / texture_square_len  # texture repetition count
        uvs = np.zeros((len(vertices), 2))
        uvs[:, 0] = (xys[:, 0] + floor_half_width) / texture_square_len
        uvs[:, 1] = (xys[:, 1] + floor_half_width) / texture_square_len

        # Generate triangle indices
        triangles = []
        for i in range(floor_length):
            # First set of triangles
            triangles.append(
                np.stack(
                    [
                        np.arange(floor_width) + i * (floor_width + 1),
                        np.arange(floor_width) + 1 + floor_width + i * (floor_width + 1),
                        np.arange(floor_width) + 1 + i * (floor_width + 1),
                    ],
                    axis=1,
                )
            )
            # Second set of triangles
            triangles.append(
                np.stack(
                    [
                        np.arange(floor_width) + 1 + floor_width + i * (floor_width + 1),
                        np.arange(floor_width) + floor_width + 2 + i * (floor_width + 1),
                        np.arange(floor_width) + 1 + i * (floor_width + 1),
                    ],
                    axis=1,
                )
            )
        triangles = np.concatenate(triangles)

        # Create the render shape
        shape = sapien.render.RenderShapeTriangleMesh(
            vertices=vertices,
            triangles=triangles,
            normals=normals,
            uvs=uvs,
            material=mat,
        )

        # Add render components to the actor
        for obj in actor._objs:
            floor_comp = sapien.render.RenderBodyComponent()
            floor_comp.attach(shape)
            obj.add_component(floor_comp)

    return actor
