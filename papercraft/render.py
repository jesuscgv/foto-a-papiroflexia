"""Render "cómo quedará montado": foto de producto en 3/4 sobre una mesa + vuelta de 360°.

blender -b -P render.py -- modelo.obj carpeta_salida [frames_vuelta]
Escribe render.png y vuelta/####.png
"""
import sys, math
import bpy
from mathutils import Vector

args = sys.argv[sys.argv.index("--") + 1:]
src, out = args[0], args[1]
frames = int(args[2]) if len(args) > 2 else 0

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.wm.obj_import(filepath=src)
ob = bpy.context.selected_objects[0]
# papel mate: sin brillo, con un pelín de translucidez aparente vía roughness alta
for mat in ob.data.materials:
    nt = mat.node_tree
    for n in nt.nodes:
        if n.type == "TEX_IMAGE":
            n.interpolation, n.extension = "Cubic", "EXTEND"
    bsdf = nt.nodes.get("Principled BSDF")
    bsdf.inputs["Roughness"].default_value = 1.0
    bsdf.inputs["Specular IOR Level"].default_value = 0.05

# mm -> m (tamaño real) y apoyado en el suelo, centrado
ob.scale = (0.001, 0.001, 0.001)
bpy.context.view_layer.update()
bb = [ob.matrix_world @ Vector(c) for c in ob.bound_box]
mn = Vector((min(v.x for v in bb), min(v.y for v in bb), min(v.z for v in bb)))
mx = Vector((max(v.x for v in bb), max(v.y for v in bb), max(v.z for v in bb)))
ob.location -= Vector(((mn.x + mx.x) / 2, (mn.y + mx.y) / 2, mn.z))
size = (mx - mn).length
h = mx.z - mn.z

sc = bpy.context.scene
sc.render.engine = "CYCLES"
sc.cycles.device = "GPU"
prefs = bpy.context.preferences.addons["cycles"].preferences
try:
    prefs.compute_device_type = "METAL"; prefs.get_devices()
    for d in prefs.devices:
        d.use = True
except Exception:
    sc.cycles.device = "CPU"
sc.cycles.samples = 96
sc.cycles.use_denoising = True
sc.view_settings.view_transform = "AgX"
sc.view_settings.look = "AgX - Medium High Contrast"
sc.view_settings.exposure = -0.45

# mesa: plano con color madera clara
bpy.ops.mesh.primitive_plane_add(size=size * 12)
floor = bpy.context.active_object
fm = bpy.data.materials.new("mesa"); fm.use_nodes = True
fb = fm.node_tree.nodes["Principled BSDF"]
fb.inputs["Base Color"].default_value = (0.78, 0.74, 0.68, 1)
fb.inputs["Roughness"].default_value = 0.8
floor.data.materials.append(fm)
floor.is_shadow_catcher = True
sc.render.film_transparent = True

# fondo: pared clara curvada (sin horizonte) -> mundo gris cálido
w = bpy.data.worlds.new("w"); sc.world = w; w.use_nodes = True
w.node_tree.nodes["Background"].inputs[0].default_value = (0.86, 0.84, 0.80, 1)
w.node_tree.nodes["Background"].inputs[1].default_value = 0.35

def area(name, loc, energy, s):
    l = bpy.data.lights.new(name, "AREA"); l.energy = energy; l.size = s
    o = bpy.data.objects.new(name, l); o.location = loc
    o.rotation_euler = (Vector((0, 0, h * 0.5)) - Vector(loc)).to_track_quat("-Z", "Y").to_euler()
    sc.collection.objects.link(o)

k = size
area("clave", (-1.2 * k, -1.6 * k, 1.8 * k), 300 * k * k, 1.2 * k)      # ventana grande, arriba izq.
area("relleno", (1.6 * k, -1.0 * k, 0.8 * k), 90 * k * k, 1.5 * k)
area("contra", (0.6 * k, 1.8 * k, 1.6 * k), 160 * k * k, 0.8 * k)

cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
cam.data.lens = 70
cam.data.dof.use_dof = True
cam.data.dof.aperture_fstop = 5.6
sc.collection.objects.link(cam); sc.camera = cam
target = Vector((0, 0, h * 0.48))

def aim(angle_deg, dist=1.9, elev=0.35):
    a = math.radians(angle_deg)
    cam.location = target + Vector((dist * k * math.sin(a), -dist * k * math.cos(a), elev * k))
    cam.rotation_euler = (target - cam.location).to_track_quat("-Z", "Y").to_euler()
    cam.data.dof.focus_distance = (target - cam.location).length

sc.render.resolution_x, sc.render.resolution_y = 1200, 1400
aim(float(args[3]) if len(args) > 3 else -28)
sc.render.filepath = f"{out}/render.png"
bpy.ops.render.render(write_still=True)

if frames:
    sc.render.resolution_x, sc.render.resolution_y = 720, 840
    sc.cycles.samples = 32
    for i in range(frames):
        aim(-28 + 360 * i / frames)
        sc.render.filepath = f"{out}/vuelta/{i:04d}.png"
        bpy.ops.render.render(write_still=True)
print("RENDER OK")
