import argparse
import math
import os
import sys

import bpy
from mathutils import Vector


def parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    parser = argparse.ArgumentParser(description="Render a single STL preview for thesis figures.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--size", type=int, default=480)
    return parser.parse_args(argv)


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)
    for block in bpy.data.materials:
        if block.users == 0:
            bpy.data.materials.remove(block)


def import_mesh(path: str):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".stl":
        bpy.ops.import_mesh.stl(filepath=path)
    elif ext == ".ply":
        bpy.ops.import_mesh.ply(filepath=path)
    else:
        raise ValueError(f"Unsupported mesh format: {ext}")
    meshes = [obj for obj in bpy.context.selected_objects if obj.type == "MESH"]
    if not meshes:
        raise RuntimeError("No mesh imported.")
    bpy.context.view_layer.objects.active = meshes[0]
    if len(meshes) > 1:
        bpy.ops.object.join()
    obj = bpy.context.active_object
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode="OBJECT")
    return obj


def normalize_object(obj):
    # Use the same normalization style as the dataset renderer.
    bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
    obj.location = (0.0, 0.0, 0.0)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    max_radius = max(Vector(v).length for v in obj.bound_box)
    scale = 0.95 / max(max_radius, 1e-6)
    obj.scale = (scale, scale, scale)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)


def setup_material(obj):
    mat = bpy.data.materials.new(name="NormalZGray")
    mat.use_nodes = True
    mat.blend_method = "OPAQUE"
    mat.use_screen_refraction = False
    mat.show_transparent_back = False

    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    geom_node = nodes.new(type="ShaderNodeNewGeometry")
    sep_node = nodes.new(type="ShaderNodeSeparateXYZ")
    map_node = nodes.new(type="ShaderNodeMapRange")
    bsdf_node = nodes.new(type="ShaderNodeBsdfDiffuse")
    output_node = nodes.new(type="ShaderNodeOutputMaterial")

    map_node.inputs["From Min"].default_value = -1.0
    map_node.inputs["From Max"].default_value = 1.0
    map_node.inputs["To Min"].default_value = 0.3
    map_node.inputs["To Max"].default_value = 0.8
    map_node.clamp = True

    links.new(geom_node.outputs["Normal"], sep_node.inputs["Vector"])
    links.new(sep_node.outputs["Z"], map_node.inputs["Value"])
    links.new(map_node.outputs["Result"], bsdf_node.inputs["Color"])
    links.new(bsdf_node.outputs["BSDF"], output_node.inputs["Surface"])

    obj.data.materials.clear()
    obj.data.materials.append(mat)


def setup_scene(size: int):
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = size
    scene.render.resolution_y = size
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "BW"
    scene.render.film_transparent = False
    scene.render.use_freestyle = True
    scene.render.line_thickness = 1.0
    scene.eevee.use_bloom = False
    scene.eevee.use_ssr = False
    scene.eevee.use_motion_blur = False
    scene.eevee.use_gtao = False
    linesets = bpy.context.view_layer.freestyle_settings.linesets
    bpy.context.view_layer.freestyle_settings.crease_angle = math.radians(120)
    if linesets:
        lineset = linesets[0]
        for attr, value in {
            "select_silhouette": True,
            "select_border": False,
            "select_crease": True,
            "select_edge_mark": False,
            "select_material_boundary": False,
            "select_suggestive_contour": False,
            "select_ridge_valley": False,
        }.items():
            if hasattr(lineset, attr):
                setattr(lineset, attr, value)
        if hasattr(lineset, "visibility"):
            lineset.visibility = "VISIBLE"
    scene.display_settings.display_device = "sRGB"
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0
    scene.view_settings.gamma = 1
    world = bpy.data.worlds["World"]
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs[0].default_value = (1.0, 1.0, 1.0, 1.0)
    bg.inputs[1].default_value = 1.0
    return scene


def setup_lights():
    light_data = bpy.data.lights.new(name="Sun", type="SUN")
    light_data.energy = 2.2
    light = bpy.data.objects.new(name="Sun", object_data=light_data)
    bpy.context.collection.objects.link(light)
    light.location = (4.0, -4.0, 6.0)
    light.rotation_euler = (math.radians(35), 0.0, math.radians(35))

    fill_data = bpy.data.lights.new(name="Fill", type="AREA")
    fill_data.energy = 1500
    fill_data.shape = "RECTANGLE"
    fill_data.size = 6
    fill = bpy.data.objects.new(name="Fill", object_data=fill_data)
    bpy.context.collection.objects.link(fill)
    fill.location = (-2.5, 2.5, 3.0)
    fill.rotation_euler = (math.radians(60), 0.0, math.radians(-135))


def setup_camera():
    cam_data = bpy.data.cameras.new(name="Camera")
    cam_data.type = "ORTHO"
    cam_data.ortho_scale = 2.0
    cam = bpy.data.objects.new("Camera", cam_data)
    bpy.context.collection.objects.link(cam)
    coord = Vector((1.0, -1.0, 1.0))
    cam.location = coord.normalized() * 5.0
    target = bpy.data.objects.new("Target", None)
    target.location = (0.0, 0.0, 0.0)
    bpy.context.collection.objects.link(target)
    constraint = cam.constraints.new(type="TRACK_TO")
    constraint.target = target
    constraint.track_axis = "TRACK_NEGATIVE_Z"
    constraint.up_axis = "UP_Y"
    bpy.context.scene.camera = cam


def main():
    args = parse_args()
    clear_scene()
    scene = setup_scene(args.size)
    obj = import_mesh(args.input)
    normalize_object(obj)
    setup_material(obj)
    setup_lights()
    setup_camera()
    scene.render.filepath = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(scene.render.filepath), exist_ok=True)
    bpy.ops.render.render(write_still=True)


if __name__ == "__main__":
    main()
