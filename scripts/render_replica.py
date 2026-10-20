"""Render camera trajectories from Replica vertex-colored geometry using Blender."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from open_atlas.geometry import intrinsics, trajectory


def render(args, mesh_path):
    import bpy
    from mathutils import Matrix

    mesh_data = np.load(mesh_path, allow_pickle=False)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 1
    scene.cycles.use_denoising = False
    scene.render.resolution_x = scene.render.resolution_y = args.size
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.film_transparent = False
    scene.view_settings.view_transform = "Standard"
    mesh = bpy.data.meshes.new("replica")
    mesh.from_pydata(mesh_data["vertices"].tolist(), [], mesh_data["faces"].tolist())
    mesh.update()
    obj = bpy.data.objects.new("replica", mesh)
    scene.collection.objects.link(obj)
    colors = mesh_data["colors"]
    # 顶点色是 sRGB；先转为线性值，避免 Blender 输出时重复应用伽马。
    linear = np.where(colors <= .04045, colors / 12.92, ((colors + .055) / 1.055)**2.4)
    rgba = np.concatenate([linear, np.ones((len(colors), 1))], axis=1).astype(np.float32)
    attr = mesh.color_attributes.new(name="replica_color", type="FLOAT_COLOR", domain="POINT")
    attr.data.foreach_set("color", rgba.ravel())
    material = bpy.data.materials.new("vertex_color_approximation")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    color_node = nodes.new("ShaderNodeVertexColor")
    color_node.layer_name = attr.name
    emission = nodes.new("ShaderNodeEmission")
    output = nodes.new("ShaderNodeOutputMaterial")
    material.node_tree.links.new(color_node.outputs["Color"], emission.inputs["Color"])
    material.node_tree.links.new(emission.outputs[0], output.inputs["Surface"])
    obj.data.materials.append(material)

    camera = bpy.data.cameras.new("camera")
    camera.sensor_fit = "HORIZONTAL"
    camera.sensor_width = 36
    camera.lens = 36 / (2 * np.tan(np.deg2rad(args.fov) / 2))
    camera.clip_start, camera.clip_end = .005, 10.
    camera_obj = bpy.data.objects.new("camera", camera)
    scene.collection.objects.link(camera_obj)
    scene.camera = camera_obj
    K = intrinsics(args.size, args.size, args.fov)
    transform = mesh_data["world_from_raw"]
    center = (mesh_data["vertices"].min(0) + mesh_data["vertices"].max(0)) / 2
    eye = (transform @ np.array([*args.position, 1]))[:3] if args.position else center
    focus = (transform @ np.array([*args.target, 1]))[:3] if args.target else eye + np.array([0, 1, 0])
    scene.view_layers[0].use_pass_z = True
    scene.use_nodes = True
    nodes = scene.node_tree.nodes
    nodes.clear()
    layers = nodes.new("CompositorNodeRLayers")
    depth_output = nodes.new("CompositorNodeOutputFile")
    depth_output.format.file_format = "OPEN_EXR"
    depth_output.format.color_depth = "32"
    depth_output.file_slots[0].path = "z_"
    scene.node_tree.links.new(layers.outputs["Depth"], depth_output.inputs[0])
    scene_id = args.scene.name if args.scene.is_dir() else args.scene.parent.name

    for kind in args.trajectories:
        for seed in range(args.seed, args.seed + args.seeds):
            directory = args.output / f"{scene_id}_{kind}_{seed:03d}"
            if directory.exists():
                raise ValueError(f"Output already exists: {directory}; choose a fresh output directory")
            (directory / "rgb").mkdir(parents=True)
            (directory / "depth").mkdir()
            temporary = directory / ".depth_exr"
            temporary.mkdir()
            depth_output.base_path = str(temporary.resolve())
            poses = trajectory(eye, focus, kind, args.frames, args.amplitude, seed)
            frames = []
            for index, pose in enumerate(poses):
                # OpenCV 的右/下/前转换为 Blender 相机的右/上/后。
                camera_obj.matrix_world = Matrix((pose @ np.diag([1, -1, -1, 1])).tolist())
                scene.frame_set(index)
                rgb_path = directory / "rgb" / f"{index:06d}.png"
                scene.render.filepath = str(rgb_path.resolve())
                bpy.ops.render.render(write_still=True)
                depth_file = next(temporary.glob("*.exr"))
                image = bpy.data.images.load(str(depth_file.resolve()), check_existing=False)
                # EXR 像素自下而上；翻转后与 PNG 对齐，Z pass 已是相机 Z 深度。
                z = np.array(image.pixels[:], dtype=np.float32).reshape(args.size, args.size, 4)[::-1, :, 0].copy()
                bpy.data.images.remove(image)
                depth_file.unlink()
                z[~np.isfinite(z) | (z <= 0) | (z > camera.clip_end)] = 0
                np.save(directory / "depth" / f"{index:06d}.npy", z)
                frames.append({"frame_id": index, "timestamp": index / args.fps,
                               "rgb": f"rgb/{index:06d}.png", "depth": f"depth/{index:06d}.npy",
                               "c2w": pose.tolist(), "w2c": np.linalg.inv(pose).tolist()})
            temporary.rmdir()
            metadata = {"scene_id": scene_id, "split": args.split, "trajectory": kind, "seed": seed,
                        "appearance": "vertex_color", "voxel_size": args.voxel_size,
                        "blender_version": bpy.app.version_string,
                        "camera_convention": "OpenCV: right, down, forward; pixel centers (x+.5,y+.5)",
                        "depth_convention": "camera_z; normalized world units; 0 means invalid",
                        "world_from_raw": transform.tolist(), "width": args.size, "height": args.size,
                        "fps": args.fps, "K": K.tolist(), "frames": frames}
            (directory / "transforms.json").write_text(json.dumps(metadata, indent=2) + "\n")
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", str(args.fps),
                            "-i", str(directory / "rgb/%06d.png"), "-c:v", "libx264",
                            "-pix_fmt", "yuv420p", str(directory / "preview.mp4")], check=True)
            print(f"RENDERED {directory}: {len(frames)} frames", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/rendered/replica"))
    parser.add_argument("--trajectories", nargs="+", choices=["orbit", "dolly", "truck", "crane", "pan", "compound"],
                        default=["orbit", "dolly", "truck", "crane", "pan", "compound"])
    parser.add_argument("--frames", type=int, default=48)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--fov", type=float, default=60)
    parser.add_argument("--amplitude", type=float, default=.08,
                        help="Translation in normalized scene units; orbit/pan rotation in radians")
    parser.add_argument("--position", type=float, nargs=3, help="Camera center in raw Replica coordinates")
    parser.add_argument("--target", type=float, nargs=3, help="Look-at point in raw Replica coordinates")
    parser.add_argument("--up-axis", choices=["z", "y"], default="z")
    parser.add_argument("--voxel-size", type=float, default=.025, help="Normalized-world voxel size; 0 keeps full mesh")
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--worker-mesh", type=Path, help=argparse.SUPPRESS)
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    args = parser.parse_args(argv)
    if args.frames < 2 or args.size < 8 or args.size % 2 or args.fps < 1 or args.seeds < 1:
        parser.error("Need frames>=2, even size>=8, fps>=1 and seeds>=1")
    if not shutil.which("ffmpeg"):
        parser.error("ffmpeg is required")
    if args.worker_mesh:
        render(args, args.worker_mesh)
        return
    if not shutil.which("blender"):
        parser.error("Blender is required; install it and put blender on PATH")
    from open_atlas.mesh import prepare_mesh
    args.output.mkdir(parents=True, exist_ok=True)
    source = args.scene / "mesh.ply" if args.scene.is_dir() else args.scene
    mesh = prepare_mesh(source, args.voxel_size, args.up_axis)
    prepared = args.output / f".{source.parent.name}_mesh.npz"
    np.savez(prepared, **mesh)
    print(f"Prepared {len(mesh['vertices']):,} vertices / {len(mesh['faces']):,} triangles", flush=True)
    # Blender 自带 Python；通过准备好的 NumPy 网格传递数据，避免依赖训练环境。
    subprocess.run(["blender", "--background", "--factory-startup", "--threads", "2", "--python-exit-code", "1",
                    "--python", str(Path(__file__).resolve()), "--", *argv, "--worker-mesh", str(prepared.resolve())], check=True)


if __name__ == "__main__":
    main()
