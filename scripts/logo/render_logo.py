"""Render the VolFill logo with Cycles (transparent background).

  /Applications/Blender.app/Contents/MacOS/Blender -b -P scripts/logo/render_logo.py -- \
      --out figures/logo/volfill_logo.png --res 2048 --samples 512

Variants
  insert  : (default) 5x5x5 rounded-voxel cube in a purple->blue->teal screen-space ramp.
            Its top-front 2x2x2 corner is an empty, darkened slot; the matching yellow block
            hovers just above it (grid-aligned, slightly offset) and casts a soft shadow
            into the slot -- the hidden part being filled in.
  cutaway : earlier design (figures/logo/alt/cutaway_*): the corner is cut away to reveal a
            glowing yellow interior. Reproduce with
            --variant cutaway --block 3,3,3 --elev 31 --glow 0.2 --key 1.5
            --key-dir=0.37,-0.57,0.73 --ambient 0.32
"""
import argparse
import math
import sys
from pathlib import Path

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--out", default="figures/logo/volfill_logo.png")
ap.add_argument("--res", type=int, default=1024)
ap.add_argument("--samples", type=int, default=256)
ap.add_argument("--variant", default="insert", choices=["insert", "cutaway"])
ap.add_argument("--n", type=int, default=5, help="voxels per cube edge")
ap.add_argument("--block", default="2,2,2", help="corner block size x,y,z in voxels")
ap.add_argument("--gap", type=float, default=0.08)
ap.add_argument("--bevel", type=float, default=0.13)
ap.add_argument("--azim", type=float, default=38.0)
ap.add_argument("--elev", type=float, default=30.0)
ap.add_argument("--dist", type=float, default=6.0, help="camera distance in cube sizes")
ap.add_argument("--pad", type=float, default=0.04, help="frame padding (fraction of image)")
ap.add_argument("--glow", type=float, default=0.1, help="yellow emission strength")
ap.add_argument("--key", type=float, default=2.0, help="key sun strength")
ap.add_argument("--key-dir", default="0.3,-0.35,0.89", help="direction towards the key sun")
ap.add_argument("--ambient", type=float, default=0.14, help="flat world ambient strength")
ap.add_argument("--spec", type=float, default=0.4, help="studio HDRI strength (reflections)")
ap.add_argument("--lift", type=float, default=1.4, help="insert: block lift (z) in voxels")
ap.add_argument("--shift", default="-0.45,0.45", help="insert: block x,y offset in voxels")
ap.add_argument("--slot-dark", type=float, default=0.8, help="insert: slot wall brightness")
ap.add_argument("--floor-dark", type=float, default=0.7, help="insert: slot floor brightness")
args = ap.parse_args(argv)

OUT = Path(args.out).resolve()
N = args.n


def srgb(h):
    h = h.lstrip("#")
    c = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    return [x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4 for x in c]


RAMP = [srgb("#7E6BD9"), srgb("#58C4DD"), srgb("#5CD0B3")]
YELLOW = srgb("#F7D96F")


def ramp(t):
    t = min(max(t, 0.0), 1.0) * (len(RAMP) - 1)
    i = min(int(t), len(RAMP) - 2)
    f = t - i
    return [a + (b - a) * f for a, b in zip(RAMP[i], RAMP[i + 1])]


bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene


# ---------------------------------------------------------------- materials
def principled(name, **inputs):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    for k, v in inputs.items():
        bsdf.inputs[k].default_value = v
    return mat, bsdf


COAT = {"Coat Weight": 0.6, "Coat Roughness": 0.08}
vox_mat, vox_bsdf = principled("voxel", Roughness=0.38, **COAT)
info = vox_mat.node_tree.nodes.new("ShaderNodeObjectInfo")
vox_mat.node_tree.links.new(info.outputs["Color"], vox_bsdf.inputs["Base Color"])

fill_mat, _ = principled("fill", **{
    "Base Color": (*YELLOW, 1), "Roughness": 0.35, **COAT,
    "Emission Color": (*YELLOW, 1), "Emission Strength": args.glow})


# ---------------------------------------------------------------- geometry
def rounded_cube_mesh(size, bevel):
    bpy.ops.mesh.primitive_cube_add(size=size)
    ob = bpy.context.active_object
    ob.data.shade_smooth()
    mod = ob.modifiers.new("bevel", "BEVEL")
    mod.width = bevel
    mod.segments = 5
    mod.limit_method = "NONE"
    mod.harden_normals = True
    me = bpy.data.meshes.new_from_object(ob.evaluated_get(bpy.context.evaluated_depsgraph_get()))
    bpy.data.objects.remove(ob)
    me.materials.append(None)
    return me


VOX = rounded_cube_mesh(1 - args.gap, args.bevel)
voxels = []


def add_voxel(p, kind):
    ob = bpy.data.objects.new("vox", VOX)
    ob.location = p
    scene.collection.objects.link(ob)
    # All voxels share one mesh, so the material must live on the object slot.
    slot = ob.material_slots[0]
    slot.link = "OBJECT"
    slot.material = fill_mat if kind == "fill" else vox_mat
    voxels.append((ob, kind))


# The camera looks from +x, -y, +z; the corner block sits at that top-front corner.
bx, by, bz = (int(s) for s in args.block.split(","))
x0, z0 = N - bx, N - bz
sx, sy = (float(s) for s in args.shift.split(","))
lift = Vector((sx, sy, args.lift))
for x in range(N):
    for y in range(N):
        for z in range(N):
            p = Vector((x + 0.5, y + 0.5, z + 0.5))
            if x >= x0 and y < by and z >= z0:
                if args.variant == "insert":
                    add_voxel(p + lift, "fill")
                continue
            shell = min(x, y, z) == 0 or max(x, y, z) == N - 1
            add_voxel(p, "shell" if shell or args.variant == "insert" else "fill")

center = Vector((N / 2,) * 3)

if args.variant == "insert":
    # Darken faces lining the empty slot. d = min(x - x0, by - y, z - z0) is ~ -gap/2 on slot
    # faces and < -0.5 on every outer face, so a smoothstep on d isolates the slot. Walls get
    # --slot-dark, the up-facing floor --floor-dark.
    nt = vox_mat.node_tree
    geo = nt.nodes.new("ShaderNodeNewGeometry")
    pos_xyz = nt.nodes.new("ShaderNodeSeparateXYZ")
    nrm_xyz = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(geo.outputs["Position"], pos_xyz.inputs[0])
    nt.links.new(geo.outputs["Normal"], nrm_xyz.inputs[0])

    def math_node(op, *ins):
        m = nt.nodes.new("ShaderNodeMath")
        m.operation = op
        for i, v in enumerate(ins):
            if isinstance(v, (int, float)):
                m.inputs[i].default_value = v
            else:
                nt.links.new(v, m.inputs[i])
        return m.outputs[0]

    dx = math_node("SUBTRACT", pos_xyz.outputs["X"], x0)
    dy = math_node("SUBTRACT", by, pos_xyz.outputs["Y"])
    dz = math_node("SUBTRACT", pos_xyz.outputs["Z"], z0)
    d = math_node("MINIMUM", math_node("MINIMUM", dx, dy), dz)
    up = math_node("MAXIMUM", nrm_xyz.outputs["Z"], 0.0)
    target = math_node("MULTIPLY_ADD", up, args.floor_dark - args.slot_dark, args.slot_dark)
    mr = nt.nodes.new("ShaderNodeMapRange")
    mr.interpolation_type = "SMOOTHSTEP"
    mr.inputs["From Min"].default_value = -args.gap / 2 - args.bevel
    mr.inputs["From Max"].default_value = -args.gap / 2 + 0.01
    mr.inputs["To Min"].default_value = 1.0
    nt.links.new(d, mr.inputs["Value"])
    nt.links.new(target, mr.inputs["To Max"])
    scale = nt.nodes.new("ShaderNodeVectorMath")
    scale.operation = "SCALE"
    nt.links.new(info.outputs["Color"], scale.inputs[0])
    nt.links.new(mr.outputs["Result"], scale.inputs["Scale"])
    nt.links.new(scale.outputs["Vector"], vox_bsdf.inputs["Base Color"])

# ---------------------------------------------------------------- camera
az, el = math.radians(args.azim), math.radians(args.elev)
view_dir = Vector((math.cos(el) * math.cos(az), -math.cos(el) * math.sin(az), math.sin(el)))
cam_data = bpy.data.cameras.new("cam")
cam_data.lens = 50
cam_data.sensor_fit = "HORIZONTAL"
cam = bpy.data.objects.new("cam", cam_data)
cam.location = center + view_dir * (N * args.dist)
cam.rotation_euler = (-view_dir).to_track_quat("-Z", "Y").to_euler()
scene.collection.objects.link(cam)
scene.camera = cam
scene.render.resolution_x = scene.render.resolution_y = args.res
bpy.context.view_layer.update()

# Tight framing: fit the projected bounding box of every voxel vertex.
us, vs = [], []
for ob, _ in voxels:
    for v in VOX.vertices:
        q = world_to_camera_view(scene, cam, ob.matrix_world @ v.co)
        us.append(q.x)
        vs.append(q.y)
k = (1 - 2 * args.pad) / max(max(us) - min(us), max(vs) - min(vs))
cam_data.lens *= k
cam_data.shift_x = ((min(us) + max(us)) / 2 - 0.5) * k
cam_data.shift_y = ((min(vs) + max(vs)) / 2 - 0.5) * k
bpy.context.view_layer.update()

# Screen-space colour ramp over the cube body: purple on the left, teal on the right.
pos = [world_to_camera_view(scene, cam, ob.location) for ob, _ in voxels]
body = [q for (_, kind), q in zip(voxels, pos) if args.variant == "cutaway" or kind != "fill"]
u_lo, u_hi = min(q.x for q in body), max(q.x for q in body)
v_lo, v_hi = min(q.y for q in body), max(q.y for q in body)
for (ob, _), q in zip(voxels, pos):
    t = 0.85 * (q.x - u_lo) / (u_hi - u_lo) + 0.15 * (q.y - v_lo) / (v_hi - v_lo)
    ob.color = (*ramp(1.15 * t - 0.1), 1)

# ---------------------------------------------------------------- lights / world
# Steep key sun from the camera side: distinct brightness per face pins the 3D reading, and
# its soft shadow of the hovering block lands on the slot walls/floor facing the camera.
sun = bpy.data.lights.new("key", "SUN")
sun.energy = args.key
sun.angle = math.radians(12)
sun_ob = bpy.data.objects.new("key", sun)
key_dir = Vector([float(s) for s in args.key_dir.split(",")]).normalized()
sun_ob.rotation_euler = (-key_dir).to_track_quat("-Z", "Y").to_euler()
scene.collection.objects.link(sun_ob)

rim = bpy.data.lights.new("rim", "AREA")
rim.energy = 400
rim.size = N * 1.5
rim_ob = bpy.data.objects.new("rim", rim)
rim_ob.location = center + Vector((-1.0, 1.0, 0.6)).normalized() * N * 3
rim_ob.rotation_euler = (center - rim_ob.location).to_track_quat("-Z", "Y").to_euler()
scene.collection.objects.link(rim_ob)

# World: flat cool ambient for diffuse; the studio HDRI is seen by glossy rays only.
world = bpy.data.worlds.new("world")
world.use_nodes = True
wn, wl = world.node_tree.nodes, world.node_tree.links
amb = wn["Background"]
amb.inputs["Color"].default_value = (0.86, 0.9, 1.0, 1)
amb.inputs["Strength"].default_value = args.ambient
hdr = wn.new("ShaderNodeBackground")
hdr.inputs["Strength"].default_value = args.spec
studio = Path(bpy.utils.resource_path("LOCAL")) / "datafiles/studiolights/world/studio.exr"
if studio.exists():
    env = wn.new("ShaderNodeTexEnvironment")
    env.image = bpy.data.images.load(str(studio))
    wl.new(env.outputs["Color"], hdr.inputs["Color"])
lp = wn.new("ShaderNodeLightPath")
mix = wn.new("ShaderNodeMixShader")
wl.new(lp.outputs["Is Glossy Ray"], mix.inputs["Fac"])
wl.new(amb.outputs["Background"], mix.inputs[1])
wl.new(hdr.outputs["Background"], mix.inputs[2])
wl.new(mix.outputs["Shader"], wn["World Output"].inputs["Surface"])
scene.world = world

# ---------------------------------------------------------------- render
scene.render.engine = "CYCLES"
cy = scene.cycles
cy.samples = args.samples
cy.use_adaptive_sampling = True
cy.use_denoising = True
prefs = bpy.context.preferences.addons["cycles"].preferences
for backend in ("METAL", "OPTIX", "CUDA"):
    try:
        prefs.compute_device_type = backend
        prefs.get_devices()
        if any(d.type == backend for d in prefs.devices):
            for d in prefs.devices:
                d.use = True
            cy.device = "GPU"
            break
    except TypeError:
        continue
scene.render.film_transparent = True
scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_mode = "RGBA"
scene.render.image_settings.color_depth = "8"
# Standard (not AgX/Filmic) so the Manim palette stays saturated.
scene.view_settings.view_transform = "Standard"
scene.view_settings.look = "None"

OUT.parent.mkdir(parents=True, exist_ok=True)
scene.render.filepath = str(OUT)
bpy.ops.render.render(write_still=True)
print(f"wrote {OUT}  device={cy.device}  voxels={len(voxels)}")
