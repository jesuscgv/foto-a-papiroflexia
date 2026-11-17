"""Paso 2 (Blender sin interfaz): malla densa -> low-poly para papercraft.

blender -b -P decimate.py -- dense.obj low.obj CARAS
Si la simplificación deja aristas no-manifold, reintenta con objetivos cercanos.
"""
import sys
import bpy
import bmesh

src, dst, target = sys.argv[sys.argv.index("--") + 1:][:3]
target = int(target)

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.wm.obj_import(filepath=src, forward_axis="Y", up_axis="Z")
ob = bpy.context.selected_objects[0]
bpy.context.view_layer.objects.active = ob

# suavizado para quitar el escalonado de los vóxeles
m = ob.modifiers.new("smooth", "LAPLACIANSMOOTH")
m.iterations, m.lambda_factor, m.use_volume_preserve = 8, 0.6, True
bpy.ops.object.modifier_apply(modifier=m.name)
dense = ob.data.copy()
n0 = len(dense.polygons)


def attempt(t):
    me = dense.copy()
    ob.data = me
    m = ob.modifiers.new("dec", "DECIMATE")
    m.decimate_type, m.ratio, m.use_collapse_triangulate = "COLLAPSE", t / n0, True
    bpy.ops.object.modifier_apply(modifier=m.name)
    bm = bmesh.new()
    bm.from_mesh(me)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-3)
    bmesh.ops.dissolve_degenerate(bm, edges=bm.edges, dist=0.5)
    bmesh.ops.triangulate(bm, faces=bm.faces)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bad = sum(1 for e in bm.edges if not e.is_manifold) + sum(1 for v in bm.verts if not v.is_manifold)
    bm.to_mesh(me)
    nf = len(bm.faces)
    bm.free()
    return bad, nf


for k in range(25):
    t = target + (k + 1) // 2 * 4 * (1 if k % 2 else -1)
    bad, nf = attempt(t)
    if bad == 0:
        break
print(f"DECIMATE {n0} -> {nf} caras, aristas no-manifold: {bad}")
bpy.ops.wm.obj_export(filepath=dst, export_selected_objects=True, export_uv=False,
                      export_normals=False, export_materials=False, forward_axis="Y", up_axis="Z")
