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
    return bpy.context.active_object


def normalize_object(obj):
    bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
    obj.location = (0.0, 0.0, 0.0)
    bpy.context.view_layer.update()
    bbox = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    min_v = Vector((min(v.x for v in bbox), min(v.y for v in bbox), min(v.z for v in bbox)))
    max_v = Vector((max(v.x for v in bbox), max(v.y for v in bbox), max(v.z for v in bbox)))
    center = (min_v + max_v) * 0.5
    dims = max_v - min_v
    max_dim = max(dims.x, dims.y, dims.z, 1e-6)
    obj.location = -center
    scale = 1.6 / max_dim
    obj.scale = (scale, scale, scale)
    bpy.context.view_layer.update()


def setup_material(obj):
    mat = bpy.data.materials.new(name="CadGray")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = (0.72, 0.72, 0.72, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.55
        bsdf.inputs["Metallic"].default_value = 0.0
    obj.data.materials.clear()
    obj.data.materials.append(mat)


def setup_scene(size: int):
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = size
    scene.render.resolution_y = size
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    scene.render.use_freestyle = True
    scene.render.line_thickness = 1.2
    scene.display_settings.display_device = "sRGB"
    scene.view_settings.look = "None"
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
    cam_data.lens = 48
    cam = bpy.data.objects.new("Camera", cam_data)
    bpy.context.collection.objects.link(cam)
    cam.location = (2.8, -2.8, 2.1)
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
