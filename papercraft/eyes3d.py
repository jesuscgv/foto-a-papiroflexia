"""Rayos desde la cámara del render 'cara' (ver inspección) a píxeles dados -> puntos 3D en coords del OBJ low.
blender -b -P eyes3d.py -- modelo.glb salida.json"""
import sys, json, bpy
from mathutils import Vector
a = sys.argv[sys.argv.index("--") + 1:]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=a[0])
dg = bpy.context.evaluated_depsgraph_get()
hc = Vector((0.2095, 0.0, 0.323))
loc = hc + Vector((1, 0, 0.35)).normalized() * 0.75
rot = (hc - loc).to_track_quat("-Z", "Y").to_matrix()
RES, LENS, SENSOR = 700, 50.0, 36.0
pix = {"L": (300, 297), "L1": (270, 297), "L2": (335, 297), "R": (505, 310), "R1": (455, 310), "R2": (545, 310)}
out = {}
for k, (u, v) in pix.items():
    d = rot @ Vector(((u - RES / 2) / RES * SENSOR, -(v - RES / 2) / RES * SENSOR, -LENS)).normalized()
    hit, p, n, *_ = bpy.context.scene.ray_cast(dg, loc, d)
    # marco original -> rotado (frente -Y) -> OBJ (Y arriba, frente +Z):  obj = (y, z, x)
    out[k] = [p.y, p.z, p.x] if hit else None
json.dump(out, open(a[1], "w"), indent=1)
print("EYES", out)
