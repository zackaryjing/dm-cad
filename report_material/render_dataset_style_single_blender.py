#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import sys

import bpy
import mathutils


def parse_args():
    argv = sys.argv
    argv = argv[argv.index("--") + 1 :] if "--" in argv else []
    parser = argparse.ArgumentParser(description="Render one mesh with the original Omni-CAD dataset style.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--size", type=int, default=320)
    parser.add_argument("--view-index", type=int, default=4)
    return parser.parse_args(argv)


def setup_scene(size: int):
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = size
    scene.render.resolution_y = size
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "BW"

    scene.display_settings.display_device = "sRGB"
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0
    scene.view_settings.gamma = 1

    scene.eevee.use_bloom = False
    scene.eevee.use_ssr = False
    scene.eevee.use_motion_blur = False
    scene.eevee.use_gtao = False

    if not scene.world:
        scene.world = bpy.data.worlds.new("World")
    scene.world.use_nodes = True
    bg_node = scene.world.node_tree.nodes.get("Background")
    if bg_node:
        bg_node.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
        bg_node.inputs["Strength"].default_value = 1.0

    scene.render.use_freestyle = True
    view_layer = bpy.context.view_layer
    view_layer.freestyle_settings.crease_angle = math.radians(120)
    line_set = view_layer.freestyle_settings.linesets[0]
    line_set.select_silhouette = True
    line_set.select_crease = True
    line_set.select_border = False
    if hasattr(line_set, "visibility"):
        line_set.visibility = "VISIBLE"
    line_style = bpy.data.linestyles.get("LineStyle")
    if line_style:
        line_style.thickness = 1.8
        line_style.color = (0, 0, 0)
    return scene


def import_mesh(path: str):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".ply":
        bpy.ops.import_mesh.ply(filepath=path)
    elif ext == ".stl":
        bpy.ops.import_mesh.stl(filepath=path)
    else:
        raise ValueError(f"Unsupported mesh format: {ext}")
    obj = bpy.context.selected_objects[0]
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode="OBJECT")
    return obj


def load_and_normalize(path: str):
    obj = import_mesh(path)

    mat = bpy.data.materials.new(name="NormalMat")
    mat.use_nodes = True
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
    obj.data.materials.append(mat)

    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
    obj.location = (0, 0, 0)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    max_radius = max(mathutils.Vector(v).length for v in obj.bound_box)
    obj.scale = (0.95 / max(max_radius, 1e-6),) * 3
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    return obj


def setup_camera():
    cam_data = bpy.data.cameras.new("Camera")
    cam_obj = bpy.data.objects.new("Camera", cam_data)
    bpy.context.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj
    cam_data.type = "ORTHO"
    cam_data.ortho_scale = 2.0
    return cam_obj


def render_view(cam_obj, output_path: str, view_index: int):
    coords = [
        (-1, -1, -1),
        (1, -1, -1),
        (-1, 1, -1),
        (1, 1, -1),
        (-1, -1, 1),
        (1, -1, 1),
        (-1, 1, 1),
        (1, 1, 1),
    ]
    coord = coords[view_index % len(coords)]
    pos = mathutils.Vector(coord).normalized() * 5.0
    cam_obj.location = pos
    direction = mathutils.Vector((0, 0, 0)) - pos
    cam_obj.rotation_mode = "QUATERNION"
    cam_obj.rotation_quaternion = direction.to_track_quat("-Z", "Y")
    bpy.context.scene.render.filepath = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(bpy.context.scene.render.filepath), exist_ok=True)
    bpy.ops.render.render(write_still=True)


def main():
    args = parse_args()
    setup_scene(args.size)
    load_and_normalize(args.input)
    cam = setup_camera()
    render_view(cam, args.output, args.view_index)


if __name__ == "__main__":
    main()
