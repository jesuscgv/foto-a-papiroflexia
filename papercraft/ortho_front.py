"""Render frontal ortográfico de un GLB (mirando hacia -Y tras rotarlo como from_model.py).
Píxel (u, v) de una imagen RES x RES  <->  x = (u - RES/2) * ESC / RES,  z = (RES/2 - v) * ESC / RES + ZC
blender -b -P ortho_front.py -- modelo.glb salida.png [frente] [zc] [esc]"""
import sys, math, bpy
from mathutils import Matrix
a = sys.argv[sys.argv.index("--") + 1:]
src, out = a[0], a[1]
front = a[2] if len(a) > 2 else "+X"
zc = float(a[3]) if len(a) > 3 else 0.0
esc = float(a[4]) if len(a) > 4 else 1.0
RES = 1000
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=src)
hi = [o for o in bpy.context.scene.objects if o.type == "MESH"][0]
for o in bpy.context.scene.objects: o.select_set(o == hi)
bpy.context.view_layer.objects.active = hi
bpy.ops.object.parent_clear(type="CLEAR_KEEP_TRANSFORM")
rot = {"+X": -90, "-X": 90, "+Y": 180, "-Y": 0}[front]
hi.matrix_world = Matrix.Rotation(math.radians(rot), 4, "Z") @ hi.matrix_world
sc = bpy.context.scene; sc.render.engine = "BLENDER_EEVEE_NEXT"
sc.view_settings.view_transform = "Standard"
w = bpy.data.worlds.new("w"); sc.world = w; w.use_nodes = True
w.node_tree.nodes["Background"].inputs[0].default_value = (1, 1, 1, 1)
mat = hi.data.materials[0]
nt = mat.node_tree
tex = [n for n in nt.nodes if n.type == "TEX_IMAGE"][0]
emi = nt.nodes.new("ShaderNodeEmission"); outn = [n for n in nt.nodes if n.type == "OUTPUT_MATERIAL"][0]
nt.links.new(tex.outputs["Color"], emi.inputs["Color"]); nt.links.new(emi.outputs[0], outn.inputs["Surface"])
cam = bpy.data.objects.new("c", bpy.data.cameras.new("c")); sc.collection.objects.link(cam); sc.camera = cam
cam.data.type = "ORTHO"; cam.data.ortho_scale = esc
cam.location = (0, -5, zc); cam.rotation_euler = (math.radians(90), 0, 0)
sc.render.resolution_x = sc.render.resolution_y = RES
sc.render.filepath = out; bpy.ops.render.render(write_still=True)
print("ORTHO OK")
