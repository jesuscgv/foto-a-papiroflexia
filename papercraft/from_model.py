"""Paso 1+2 alternativo (Blender sin interfaz): modelo 3D texturizado (GLB de Tripo, etc.) -> busto low-poly
con la textura horneada.

blender -b -P from_model.py -- modelo.glb carpeta CARAS CORTE [frente] [fondo]
  CORTE: altura del corte del busto como fracción de la altura total (0 = suelo, 1 = coronilla); 0 = sin cortar
  frente: eje hacia el que mira el modelo en Blender (+X, -X, +Y, -Y). Por defecto +X (Tripo).
Escribe low.obj (con UV), baked.png y meta.json.
"""
import sys, json, math
import bpy, bmesh
import numpy as np
from mathutils import Vector, Matrix

args = sys.argv[sys.argv.index("--") + 1:]
src, out, target, cut = args[0], args[1], int(args[2]), float(args[3])
front = args[4] if len(args) > 4 else "+X"
keep_front = float(args[5]) if len(args) > 5 else 1.0  # fracción de la profundidad (desde el hocico) que se conserva
TEX = 4096

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=src)
hi = [o for o in bpy.context.scene.objects if o.type == "MESH"][0]
for o in bpy.context.scene.objects:
    o.select_set(o == hi)
bpy.context.view_layer.objects.active = hi
bpy.ops.object.parent_clear(type="CLEAR_KEEP_TRANSFORM")
# que mire a -Y (convención del resto del programa)
rot = {"+X": -90, "-X": 90, "+Y": 180, "-Y": 0}[front]
hi.matrix_world = Matrix.Rotation(math.radians(rot), 4, "Z") @ hi.matrix_world
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

co = np.empty(len(hi.data.vertices) * 3); hi.data.vertices.foreach_get("co", co); co = co.reshape(-1, 3)
zmin, zmax = co[:, 2].min(), co[:, 2].max()
zcut = zmin + cut * (zmax - zmin) if cut > 0 else None

# --- copia para la low-poly: remallado por vóxeles (cerrado y limpio), corte, simplificación
lo = hi.copy(); lo.data = hi.data.copy(); lo.name = "low"
bpy.context.scene.collection.objects.link(lo)
for o in bpy.context.scene.objects:
    o.select_set(o == lo)
bpy.context.view_layer.objects.active = lo
size = (co.max(0) - co.min(0)).max()
m = lo.modifiers.new("remesh", "REMESH"); m.mode = "VOXEL"; m.voxel_size = size / 220; m.use_smooth_shade = False
bpy.ops.object.modifier_apply(modifier=m.name)
# suavizado ANTES de cortar: así las tapas del corte quedan perfectamente planas
m = lo.modifiers.new("smooth", "LAPLACIANSMOOTH"); m.iterations, m.lambda_factor = 4, 0.5
bpy.ops.object.modifier_apply(modifier=m.name)

if zcut is not None:
    bm = bmesh.new(); bm.from_mesh(lo.data)
    res = bmesh.ops.bisect_plane(bm, geom=bm.verts[:] + bm.edges[:] + bm.faces[:], plane_co=(0, 0, zcut),
                                 plane_no=(0, 0, 1), clear_inner=True)
    edges = [e for e in res["geom_cut"] if isinstance(e, bmesh.types.BMEdge)]
    bmesh.ops.holes_fill(bm, edges=edges, sides=0)
    if keep_front < 1.0:  # corte vertical por detrás: quita el lomo
        ys = [v.co.y for v in bm.verts]
        yback = min(ys) + keep_front * (max(ys) - min(ys))
        res = bmesh.ops.bisect_plane(bm, geom=bm.verts[:] + bm.edges[:] + bm.faces[:], plane_co=(0, yback, 0),
                                     plane_no=(0, -1, 0), clear_inner=True)
        edges = [e for e in res["geom_cut"] if isinstance(e, bmesh.types.BMEdge)]
        bmesh.ops.holes_fill(bm, edges=edges, sides=0)
    bmesh.ops.triangulate(bm, faces=bm.faces)
    bm.to_mesh(lo.data); bm.free()

dense = lo.data.copy(); n0 = sum(len(p.vertices) - 2 for p in dense.polygons)  # en triángulos


def attempt(t):
    me = dense.copy(); lo.data = me
    m = lo.modifiers.new("dec", "DECIMATE")
    m.decimate_type, m.ratio, m.use_collapse_triangulate = "COLLAPSE", t / n0, True
    bpy.ops.object.modifier_apply(modifier=m.name)
    bm = bmesh.new(); bm.from_mesh(me)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=size * 1e-5)
    bmesh.ops.dissolve_degenerate(bm, edges=bm.edges, dist=size * 5e-4)
    bmesh.ops.triangulate(bm, faces=bm.faces)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bad = sum(1 for e in bm.edges if not e.is_manifold) + sum(1 for v in bm.verts if not v.is_manifold)
    bm.to_mesh(me); nf = len(bm.faces); bm.free()
    return bad, nf


for k in range(25):
    t = target + (k + 1) // 2 * 4 * (1 if k % 2 else -1)
    bad, nf = attempt(t)
    if bad == 0:
        break
print(f"LOWPOLY {n0} -> {nf} caras, no-manifold {bad}")

# --- UV de la low-poly y horneado del color desde la alta
for p in lo.data.polygons:
    p.use_smooth = False
lo.data.materials.clear()
bpy.ops.object.mode_set(mode="EDIT"); bpy.ops.mesh.select_all(action="SELECT")
bpy.ops.uv.smart_project(angle_limit=math.radians(1), island_margin=0.004)
bpy.ops.object.mode_set(mode="OBJECT")
img = bpy.data.images.new("baked", TEX, TEX)
mat = bpy.data.materials.new("bake"); mat.use_nodes = True
tn = mat.node_tree.nodes.new("ShaderNodeTexImage"); tn.image = img
mat.node_tree.nodes.active = tn
lo.data.materials.append(mat)

sc = bpy.context.scene
sc.render.engine = "CYCLES"
try:
    p = bpy.context.preferences.addons["cycles"].preferences
    p.compute_device_type = "METAL"; p.get_devices()
    for d in p.devices: d.use = True
    sc.cycles.device = "GPU"
except Exception:
    pass
sc.cycles.samples = 4
bk = sc.render.bake
bk.use_selected_to_active = True
bk.cage_extrusion = size * 0.03
bk.max_ray_distance = size * 0.08
bk.margin = 16
for o in bpy.context.scene.objects:
    o.select_set(o in (hi, lo))
bpy.context.view_layer.objects.active = lo
bpy.ops.object.bake(type="DIFFUSE", pass_filter={"COLOR"})
img.filepath_raw = f"{out}/baked.png"; img.file_format = "PNG"; img.save()

for o in bpy.context.scene.objects:
    o.select_set(o == lo)
bpy.ops.wm.obj_export(filepath=f"{out}/low.obj", export_selected_objects=True, export_uv=True,
                      export_normals=False, export_materials=False)
json.dump({"W": TEX, "H": TEX, "baked": True, "textures": {"baked": "baked.png"}}, open(f"{out}/meta.json", "w"))
print("FROM_MODEL OK")
