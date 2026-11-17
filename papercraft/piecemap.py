"""Ilustraciones del cuadernillo (Blender sin interfaz):
  - mapa_frente.png / mapa_atras.png: cada pieza de un color + posiciones de sus etiquetas (mapa.json)
  - paso_XX.png: montaje acumulado (hecho = textura, actual = naranja, pendiente = oculto)

blender -b -P piecemap.py -- carpeta   (lee modelo.obj y kit_piezas.json, escribe en carpeta/kit/)
"""
import sys, json, math
import bpy
from mathutils import Vector
from bpy_extras.object_utils import world_to_camera_view

d = sys.argv[sys.argv.index("--") + 1]
spec = json.load(open(f"{d}/kit_piezas.json"))
face_piece = spec["face_piece"]          # índice de cara -> número de pieza
steps = spec["steps"]                    # [[piezas del paso 1], [paso 2], ...]
palette = [tuple(c) for c in spec["palette"]]

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.wm.obj_import(filepath=f"{d}/modelo.obj")
ob = bpy.context.selected_objects[0]
me = ob.data
# empareja polígonos con caras del OBJ por su centro (Blender puede descartar caras degeneradas al importar)
from mathutils import kdtree
cents = spec["centroids"]  # coords del OBJ; Blender las deja iguales en local (la rotación va en el objeto)
kd = kdtree.KDTree(len(cents))
for i, (x, y, z) in enumerate(cents):
    kd.insert((x, y, z), i)
kd.balance()
face_piece = [face_piece[kd.find(p.center)[1]] for p in me.polygons]
ob.scale = (0.001,) * 3
bpy.context.view_layer.update()
bb = [ob.matrix_world @ Vector(c) for c in ob.bound_box]
mn = Vector((min(v.x for v in bb), min(v.y for v in bb), min(v.z for v in bb)))
mx = Vector((max(v.x for v in bb), max(v.y for v in bb), max(v.z for v in bb)))
ob.location -= Vector(((mn.x + mx.x) / 2, (mn.y + mx.y) / 2, mn.z))
bpy.context.view_layer.update()
size = (mx - mn).length; h = mx.z - mn.z
tex_mat = me.materials[0]
for n in tex_mat.node_tree.nodes:
    if n.type == "TEX_IMAGE":
        n.interpolation, n.extension = "Cubic", "EXTEND"
tex_mat.node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value = 1.0


def flat(name, rgb):
    m = bpy.data.materials.new(name); m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*rgb, 1); b.inputs["Roughness"].default_value = 1.0
    return m


def invisible():
    m = bpy.data.materials.new("oculto"); m.use_nodes = True
    nt = m.node_tree; nt.nodes.clear()
    o = nt.nodes.new("ShaderNodeOutputMaterial"); t = nt.nodes.new("ShaderNodeBsdfTransparent")
    nt.links.new(t.outputs[0], o.inputs[0])
    m.surface_render_method = "DITHERED"
    return m


def srgb2lin(c):
    return tuple(((x / 255 + 0.055) / 1.055) ** 2.4 if x / 255 > 0.04045 else x / 255 / 12.92 for x in c)


npieces = max(face_piece)
me.materials.clear()
me.materials.append(tex_mat)                                      # 0 textura
me.materials.append(flat("actual", srgb2lin((214, 110, 52))))      # 1 naranja de acento
me.materials.append(invisible())                                  # 2 oculto
for i in range(npieces):                                          # 3.. colores del mapa
    me.materials.append(flat(f"p{i+1}", srgb2lin(palette[i % len(palette)])))

sc = bpy.context.scene
sc.render.engine = "BLENDER_EEVEE_NEXT"
sc.render.film_transparent = True
sc.view_settings.view_transform = "Standard"
w = bpy.data.worlds.new("w"); sc.world = w; w.use_nodes = True
w.node_tree.nodes["Background"].inputs[0].default_value = (1, 1, 1, 1)
w.node_tree.nodes["Background"].inputs[1].default_value = 0.75
sun = bpy.data.objects.new("sol", bpy.data.lights.new("sol", "SUN"))
sun.data.energy = 2.2; sun.data.angle = math.radians(20)
sc.collection.objects.link(sun)
cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam")); cam.data.lens = 70
sc.collection.objects.link(cam); sc.camera = cam
target = Vector((0, 0, h * 0.5))


def aim(angle):
    a = math.radians(angle)
    cam.location = target + Vector((2.0 * size * math.sin(a), -2.0 * size * math.cos(a), 0.45 * size))
    cam.rotation_euler = (target - cam.location).to_track_quat("-Z", "Y").to_euler()
    sun.rotation_euler = (math.radians(50), 0, math.radians(angle - 35))
    bpy.context.view_layer.update()


def set_mats(fn):
    for p in me.polygons:
        p.material_index = fn(face_piece[p.index])


def render(path, res):
    sc.render.resolution_x, sc.render.resolution_y = res
    sc.render.filepath = path
    bpy.ops.render.render(write_still=True)


def visible_faces(angle):
    """Caras visibles desde la cámara (rayos), con su área proyectada aproximada y su píxel."""
    aim(angle)
    dg = bpy.context.evaluated_depsgraph_get()
    out = {}
    mw = ob.matrix_world
    for p in me.polygons:
        c = mw @ p.center
        n = (mw.to_3x3() @ p.normal).normalized()
        v = (cam.location - c)
        if n.dot(v.normalized()) < 0.15:
            continue
        hit, loc, nn, idx, o, _ = sc.ray_cast(dg, cam.location, (c - cam.location).normalized())
        if not hit or idx != p.index:
            continue
        uv = world_to_camera_view(sc, cam, c)
        out[p.index] = (p.area * n.dot(v.normalized()), uv.x, 1 - uv.y)
    return out


import os
os.makedirs(f"{d}/kit", exist_ok=True)

# --- mapa de piezas: dos vistas
labels = {}
for nm, ang in (("frente", -32), ("atras", 148)):
    set_mats(lambda k: 2 + k)
    vis = visible_faces(ang)
    best = {}
    for fi, (area, u, v) in vis.items():
        k = face_piece[fi]
        best.setdefault(k, []).append((area, u, v))
    lab = {}
    for k, lst in best.items():
        tot = sum(a for a, _, _ in lst)
        if tot < 1e-9:
            continue
        # etiqueta en la cara visible más grande de la pieza
        a, u, v = max(lst)
        lab[k] = [u, v, tot]
    labels[nm] = lab
    render(f"{d}/kit/mapa_{nm}.png", (900, 1050))

# --- pasos
done = set()
step_views = []
for i, st in enumerate(steps, 1):
    cur = set(st)
    def facing(ang):  # área de la pieza vuelta hacia la cámara (sin oclusiones: nunca sale vacío)
        a = math.radians(ang)
        view = Vector((math.sin(a), -math.cos(a), 0.25)).normalized()
        mw3 = ob.matrix_world.to_3x3()
        return sum(p.area * max(0.0, (mw3 @ p.normal).normalized().dot(view)) for p in me.polygons if face_piece[p.index] in cur)
    ang = -32 if facing(-32) >= facing(148) else 148
    aim(ang)
    set_mats(lambda k: 1 if k in cur else (0 if k in done else 2))
    if i == 1:  # el primer render con materiales transparentes sale vacío en EEVEE: se calienta
        render(f"{d}/kit/_calentar.png", (520, 600))
    render(f"{d}/kit/paso_{i:02d}.png", (520, 600))
    step_views.append("frente" if ang == -32 else "atras")
    done |= cur
json.dump({"labels": labels, "step_views": step_views}, open(f"{d}/kit/mapa.json", "w"))
print("PIECEMAP OK", len(steps), "pasos")
