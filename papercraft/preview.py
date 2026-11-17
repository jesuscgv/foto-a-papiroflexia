"""Vista previa del modelo montado (Blender sin interfaz).

blender -b -P preview.py -- modelo.obj salida.png
"""
import sys, math
import bpy
from mathutils import Vector

src, dst = sys.argv[sys.argv.index("--") + 1:][:2]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.wm.obj_import(filepath=src)  # Y arriba en el archivo -> Z arriba en Blender; frente = -Y
ob = bpy.context.selected_objects[0]
for mat in ob.data.materials:
    for n in mat.node_tree.nodes:
        if n.type == "TEX_IMAGE":
            n.interpolation = "Linear"
            n.extension = "EXTEND"
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Roughness"].default_value = 0.9
        bsdf.inputs["Specular IOR Level"].default_value = 0.1

sc = bpy.context.scene
sc.render.engine = "BLENDER_EEVEE_NEXT"
sc.render.resolution_x, sc.render.resolution_y = 520, 620
sc.render.film_transparent = False
w = bpy.data.worlds.new("w"); sc.world = w
w.use_nodes = True
w.node_tree.nodes["Background"].inputs[0].default_value = (0.92, 0.92, 0.92, 1)
w.node_tree.nodes["Background"].inputs[1].default_value = 0.9
sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN"))
sun.data.energy = 3.0
sun.rotation_euler = (math.radians(50), 0, math.radians(30))
sc.collection.objects.link(sun)

bb = [ob.matrix_world @ Vector(c) for c in ob.bound_box]
center = sum(bb, Vector()) / 8
size = max((bb[6] - bb[0]))
cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
cam.data.lens = 60
sc.collection.objects.link(cam)
sc.camera = cam

import os
tmp = []
# vistas: frente, 3/4, perfil, espalda (cámara gira alrededor del eje Y, que es el "arriba" del modelo)
for i, ang in enumerate((0, 35, 90, 180)):
    r = size * 2.6
    a = math.radians(ang)
    cam.location = center + Vector((r * math.sin(a), -r * math.cos(a), size * 0.3))
    d = center - cam.location
    cam.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
    p = f"{dst}.{i}.png"
    sc.render.filepath = p
    bpy.ops.render.render(write_still=True)
    tmp.append(p)
print("PREVIEW", " ".join(tmp))
