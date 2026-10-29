import os
import sys
import gc
import ctypes
import torch
import numpy as np
import trimesh
import cv2
from PIL import Image
from omegaconf import OmegaConf
import folder_paths
from huggingface_hub import hf_hub_download

current_dir = os.path.dirname(os.path.abspath(__file__))
ultrashape_path = os.path.join(current_dir, "UltraShape-1.0")
if ultrashape_path not in sys.path:
    sys.path.insert(0, ultrashape_path)

cellocut_dir = os.path.join(os.path.dirname(current_dir), "ComfyUI-CelloCut")
cellocut_build_dir = os.path.join(cellocut_dir, "build")
if os.path.isdir(cellocut_build_dir) and cellocut_build_dir not in sys.path:
    sys.path.insert(0, cellocut_build_dir)
if os.path.isdir(cellocut_dir) and cellocut_dir not in sys.path:
    sys.path.insert(0, cellocut_dir)

try:
    from ultrashape.rembg import BackgroundRemover
    from ultrashape.utils.misc import instantiate_from_config
    from ultrashape.surface_loaders import SharpEdgeSurfaceLoader
    from ultrashape.utils import voxelize_from_point
    from ultrashape.pipelines import UltraShapePipeline, export_to_trimesh
    from ultrashape.models.autoencoders import SurfaceExtractors
    from ultrashape import FloaterRemover
    ULTRASHAPE_IMPORT_ERROR = None
except ImportError as e:
    ULTRASHAPE_IMPORT_ERROR = e
    print(f"Error importing UltraShape modules: {e}")

try:
    from model import project_to_original_surface_gpu, _prepare_trimesh
except Exception:
    project_to_original_surface_gpu = None
    _prepare_trimesh = None

MC_ALGO_CHOICES = [
    "isoext_mc_lewiner",
    "isoext_mc_vega",
    "isoext_mc_nagae",
    "isoext_mc_lorensen",
    "isoext_dmc_vega",
    "isoext_dmc_lewiner",
    "isoext_dc_ju",
    "isoext_dc_carrera",
    "isoext_mt",
    "isoext_sn",
    "mc",
    "dmc",
]

ultrashape_models_dir = os.path.join(folder_paths.models_dir, "ultrashape")
if not os.path.exists(ultrashape_models_dir):
    os.makedirs(ultrashape_models_dir, exist_ok=True)
folder_paths.add_model_folder_path("ultrashape", ultrashape_models_dir)


try:
    _libc = ctypes.CDLL("libc.so.6")
except OSError:
    _libc = None


def _free_vram():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    # glibc keeps freed blocks (meshes, decoded grids, model weights moved back from the GPU) in its heap
    # instead of returning them to the OS, so the process RSS grew with every run; hand them back.
    if _libc is not None:
        _libc.malloc_trim(0)


class UltraShapeModelLoader:
    DESCRIPTION = "Loads the UltraShape model checkpoint and instantiates VAE, DiT, Conditioner, Scheduler, and Image Processor."
    _cache = {}

    @classmethod
    def INPUT_TYPES(s):
        raw_files = folder_paths.get_filename_list("ultrashape")
        files = [
            f for f in raw_files
            if f.endswith((".pt", ".pth", ".safetensors", ".ckpt", ".bin"))
            and not f.startswith(".")
            and "/." not in f
        ]
        if not files:
            files = ["ultrashape_v1.pt"]
        return {
            "required": {
                "ckpt_name": (files, {"tooltip": "Name of the UltraShape checkpoint file."}),
                "download_if_missing": ("BOOLEAN", {"default": True, "tooltip": "Automatically download checkpoint from Hugging Face if missing."}),
            }
        }

    RETURN_TYPES = ("ULTRASHAPE_MODEL",)
    RETURN_NAMES = ("ultrashape_model",)
    FUNCTION = "load_model"
    CATEGORY = "UltraShape"

    def load_model(self, ckpt_name, download_if_missing):
        if ULTRASHAPE_IMPORT_ERROR is not None:
            raise RuntimeError(f"UltraShape modules failed to import: {ULTRASHAPE_IMPORT_ERROR}. Please check requirements.")

        if not ckpt_name or not ckpt_name.endswith((".pt", ".pth", ".safetensors", ".ckpt", ".bin")):
            ckpt_name = "ultrashape_v1.pt"
        ckpt_path = folder_paths.get_full_path("ultrashape", ckpt_name)
        if not ckpt_path or not os.path.exists(ckpt_path) or not ckpt_path.endswith((".pt", ".pth", ".safetensors", ".ckpt", ".bin")):
            default_path = os.path.join(ultrashape_models_dir, "ultrashape_v1.pt")
            if os.path.exists(default_path):
                ckpt_path = default_path

        if not ckpt_path or not os.path.exists(ckpt_path):
            if download_if_missing:
                print(f"Downloading {ckpt_name} to {ultrashape_models_dir}...")
                try:
                    ckpt_path = hf_hub_download(
                        repo_id="infinith/UltraShape",
                        filename="ultrashape_v1.pt",
                        local_dir=ultrashape_models_dir,
                    )
                except Exception as e:
                    raise RuntimeError(f"Failed to download model: {e}")
            else:
                raise FileNotFoundError(f"Checkpoint {ckpt_name} not found and download disabled.")

        if ckpt_path in UltraShapeModelLoader._cache:
            return (UltraShapeModelLoader._cache[ckpt_path],)

        config_path = os.path.join(ultrashape_path, "configs", "infer_dit_refine.yaml")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config not found at {config_path}")

        print(f"Loading config from {config_path}...")
        config = OmegaConf.load(config_path)

        print("Instantiating VAE...")
        vae = instantiate_from_config(config.model.params.vae_config)

        print("Instantiating DiT...")
        dit = instantiate_from_config(config.model.params.dit_cfg)

        print("Instantiating Conditioner...")
        conditioner = instantiate_from_config(config.model.params.conditioner_config)

        print("Instantiating Scheduler & Processor...")
        scheduler = instantiate_from_config(config.model.params.scheduler_cfg)
        image_processor = instantiate_from_config(config.model.params.image_processor_cfg)

        # The checkpoint stores bfloat16 and every refine path runs these models in bfloat16. Float32
        # modules would only hold exact upcasts of the same values at twice the RAM (13.7 GB vs 6.9 GB).
        for module in (vae, dit, conditioner):
            module.to(torch.bfloat16)

        print(f"Loading weights from {ckpt_path}...")
        # memory-mapped: the weights are copied straight from the file, no second 7.4 GB copy in RAM
        weights = torch.load(ckpt_path, map_location="cpu", weights_only=False, mmap=True)

        vae.load_state_dict(weights["vae"], strict=True)
        dit.load_state_dict(weights["dit"], strict=True)
        conditioner.load_state_dict(weights["conditioner"], strict=True)
        del weights
        _free_vram()

        vae.eval()
        dit.eval()
        conditioner.eval()

        if hasattr(vae, "enable_flashvdm_decoder"):
            vae.enable_flashvdm_decoder()

        components = {
            "vae": vae,
            "dit": dit,
            "conditioner": conditioner,
            "scheduler": scheduler,
            "image_processor": image_processor,
        }

        result = {"components": components, "config": config, "ckpt_path": ckpt_path}
        UltraShapeModelLoader._cache[ckpt_path] = result
        return (result,)


class UltraShapeRefine:
    DESCRIPTION = "Closes holes and recreates a watertight surface (vae_hole_close / low_step_shell_refine) or refines a 3D mesh conditioned on a reference image using UltraShape."

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ultrashape_model": ("ULTRASHAPE_MODEL", {"tooltip": "Loaded UltraShape model components."}),
                "mesh": ("TRIMESH", {"tooltip": "Input coarse or open 3D mesh (Trimesh object) to close and refine."}),
                "refine_mode": (
                    ["vae_hole_close", "low_step_shell_refine", "full_image_refine"],
                    {
                        "default": "vae_hole_close",
                        "tooltip": "vae_hole_close: 0-step ShapeVAE hole closing & surface recreation; low_step_shell_refine: partial noise + low-step RefineDiT shell closure; full_image_refine: full image-conditioned refinement.",
                    },
                ),
                "mc_algo": (
                    MC_ALGO_CHOICES,
                    {
                        "default": "isoext_mc_lewiner",
                        "tooltip": "GPU surface extraction algorithm (isoext or diso/cubvh).",
                    },
                ),
                "backproject_mode": (
                    ["rizz", "bvh", "none"],
                    {
                        "default": "rizz",
                        "tooltip": "Post-extraction back-projection onto original input mesh to preserve exact cavity/surface detail.",
                    },
                ),
                "restore_world_scale": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Restore output mesh from normalized [-1, 1] space back to the original input mesh world bounding box.",
                    },
                ),
                "denoise_strength": (
                    "FLOAT",
                    {
                        "default": 0.25,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "Partial flow-matching noise strength for low_step_shell_refine (0.20 - 0.35 closes holes while preserving shape).",
                    },
                ),
                "remove_bg": ("BOOLEAN", {"default": True, "tooltip": "Automatically remove background from input image before conditioning."}),
                "steps": ("INT", {"default": 8, "min": 1, "max": 200, "tooltip": "Number of diffusion sampling steps (ignored in vae_hole_close mode)."}),
                "guidance_scale": ("FLOAT", {"default": 7.5, "min": 0.0, "max": 20.0, "step": 0.1, "tooltip": "Classifier-free guidance scale."}),
                "num_latents": ("INT", {"default": 8192, "min": 1024, "max": 65536, "step": 1024, "tooltip": "Number of surface point tokens to sample and voxelize (e.g. 4096, 8192, 16384, 32768)."}),
                "octree_res": ("INT", {"default": 384, "min": 64, "max": 2048, "step": 64, "tooltip": "Resolution for Octree surface extraction."}),
                "voxel_resolution": ("INT", {"default": 128, "min": 32, "max": 512, "step": 32, "tooltip": "Voxel resolution for conditioning query grid."}),
                "scale": ("FLOAT", {"default": 0.99, "min": 0.1, "max": 2.0, "step": 0.01, "tooltip": "Normalization scale for input surface point cloud."}),
                "chunk_size": ("INT", {"default": 12000, "min": 512, "max": 32768, "step": 512, "tooltip": "Chunk size for surface extraction decoding to save memory."}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff, "tooltip": "Random seed for reproducible sampling."}),
                "remove_floaters": ("BOOLEAN", {"default": False, "tooltip": "Remove disconnected small mesh floaters post-processing."}),
                "low_vram": ("BOOLEAN", {"default": True, "tooltip": "Enable CPU offloading during inference to save VRAM."}),
                "output_on_cpu": ("BOOLEAN", {"default": True, "tooltip": "Keep intermediate output mesh tensors on CPU to prevent OOM."}),
                "adaptive_cfg": ("BOOLEAN", {"default": False, "tooltip": "Enable adaptive CFG scheduler."}),
                "cfg_min": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 30.0, "step": 0.1, "tooltip": "Minimum guidance scale for adaptive CFG."}),
                "cfg_max": ("FLOAT", {"default": 7.5, "min": 0.0, "max": 30.0, "step": 0.1, "tooltip": "Maximum guidance scale for adaptive CFG."}),
                "cfg_start_ratio": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Start step ratio for adaptive CFG schedule."}),
                "cfg_end_ratio": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "End step ratio for adaptive CFG schedule."}),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "Optional reference image for full_image_refine (not required for vae_hole_close or low_step_shell_refine)."}),
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("refined_mesh",)
    FUNCTION = "refine"
    CATEGORY = "UltraShape"

    @classmethod
    def _preprocess_image(cls, image_tensor, mask_tensor=None, remove_bg=True, padding_ratio=0.15, target_size=(1024, 1024)):
        arr = np.clip(255.0 * image_tensor[0].cpu().numpy(), 0, 255).astype(np.uint8)

        if mask_tensor is not None:
            m = mask_tensor[0].cpu().numpy() if mask_tensor.ndim >= 3 else mask_tensor.cpu().numpy()
            alpha = np.clip(255.0 * m, 0, 255).astype(np.uint8)
            if alpha.shape[:2] != arr.shape[:2]:
                alpha = cv2.resize(alpha, (arr.shape[1], arr.shape[0]), interpolation=cv2.INTER_LINEAR)
            img_rgb = arr[:, :, :3] if arr.ndim == 3 else np.stack([arr] * 3, axis=-1)
            rgba = Image.fromarray(np.dstack([img_rgb, alpha]), mode="RGBA")
        elif arr.ndim == 3 and arr.shape[2] == 4:
            rgba = Image.fromarray(arr, mode="RGBA")
        else:
            img_rgb = arr[:, :, :3] if arr.ndim == 3 else np.stack([arr] * 3, axis=-1)
            pil_raw = Image.fromarray(img_rgb, mode="RGB")
            if remove_bg:
                try:
                    rembg_remover = BackgroundRemover()
                    rgba = rembg_remover(pil_raw)
                except Exception:
                    rgba = pil_raw.convert("RGBA")
            else:
                rgba = pil_raw.convert("RGBA")

        rgba_np = np.array(rgba)
        alpha = rgba_np[:, :, 3] if rgba_np.ndim == 3 and rgba_np.shape[2] == 4 else None

        if alpha is not None and np.any(alpha > 10):
            thresh = (alpha > 10).astype(np.uint8) * 255
        else:
            diff = np.max(np.abs(rgba_np[:, :, :3].astype(np.int32) - 255), axis=-1)
            thresh = (diff > 15).astype(np.uint8) * 255

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
            if w > 4 and h > 4:
                cropped = rgba.crop((x, y, x + w, y + h))
                side = max(w, h)
                pad = max(int(side * padding_ratio), 4)
                canvas_size = side + 2 * pad

                canvas = Image.new("RGB", (canvas_size, canvas_size), (255, 255, 255))
                paste_x = pad + (side - w) // 2
                paste_y = pad + (side - h) // 2

                if cropped.mode == "RGBA":
                    canvas.paste(cropped, (paste_x, paste_y), mask=cropped.split()[3])
                else:
                    canvas.paste(cropped, (paste_x, paste_y))

                return canvas.resize(target_size, Image.LANCZOS)

        bg = Image.new("RGB", rgba.size, (255, 255, 255))
        if rgba.mode == "RGBA":
            bg.paste(rgba, (0, 0), mask=rgba.split()[3])
        else:
            bg.paste(rgba, (0, 0))
        return bg.resize(target_size, Image.LANCZOS)

    def refine(
        self,
        ultrashape_model,
        mesh,
        refine_mode="vae_hole_close",
        mc_algo="isoext_mc_lewiner",
        backproject_mode="rizz",
        restore_world_scale=True,
        denoise_strength=0.25,
        remove_bg=True,
        steps=8,
        guidance_scale=7.5,
        num_latents=8192,
        octree_res=384,
        voxel_resolution=128,
        scale=0.99,
        chunk_size=12000,
        seed=42,
        remove_floaters=False,
        low_vram=True,
        output_on_cpu=True,
        adaptive_cfg=False,
        cfg_min=1.0,
        cfg_max=7.5,
        cfg_start_ratio=0.0,
        cfg_end_ratio=1.0,
        image=None,
        mask=None,
    ):
        _free_vram()
        components = ultrashape_model["components"]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        token_count = max(512, int(num_latents))
        if token_count % 64 != 0:
            token_count = ((token_count + 63) // 64) * 64

        if isinstance(mesh, trimesh.Scene):
            orig_mesh = trimesh.util.concatenate(
                [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
            )
        else:
            orig_mesh = trimesh.Trimesh(
                vertices=np.asarray(mesh.vertices, dtype=np.float32),
                faces=np.asarray(mesh.faces, dtype=np.int32),
                process=False,
            )

        v_orig = np.asarray(orig_mesh.vertices, dtype=np.float32)
        bbox_min = v_orig.min(0)
        bbox_max = v_orig.max(0)
        world_center = 0.5 * (bbox_min + bbox_max)
        world_scale_max = float((bbox_max - bbox_min).max())
        norm_scale = float(scale)

        vae = components["vae"]
        dit = components["dit"]
        conditioner = components["conditioner"]
        scheduler = components["scheduler"]

        algo = mc_algo if mc_algo in SurfaceExtractors else "isoext_mc_lewiner"
        vae.enable_flashvdm_decoder(
            enabled=True,
            adaptive_kv_selection=True,
            topk_mode="mean",
            mc_algo=algo,
        )

        eff_mode = refine_mode
        if eff_mode == "full_image_refine" and image is None:
            eff_mode = "low_step_shell_refine"

        if eff_mode in ("vae_hole_close", "low_step_shell_refine"):
            pts_per_pool = max(32768, token_count * 4)
            loader = SharpEdgeSurfaceLoader(
                num_sharp_points=pts_per_pool,
                num_uniform_points=pts_per_pool,
            )
            surface = loader(orig_mesh.copy(), normalize_scale=norm_scale).to(device, dtype=torch.bfloat16)
            del loader

            orig_num_latents = vae.encoder.num_latents
            orig_pc_size = vae.encoder.pc_size
            orig_downsample = vae.encoder.downsample_ratio
            orig_voxel_res = vae.encoder.voxel_query_res

            vae.encoder.num_latents = token_count
            vae.encoder.pc_size = pts_per_pool * 2
            vae.encoder.downsample_ratio = max(1, (pts_per_pool * 2) // token_count)
            vae.encoder.voxel_query_res = int(voxel_resolution)

            dit.to("cpu")
            conditioner.to("cpu")
            _free_vram()
            vae.to(device, dtype=torch.bfloat16)

            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                enc_out = vae.encode(surface, sample_posterior=False)
                kl_embed = enc_out[0] if isinstance(enc_out, tuple) else enc_out

            if eff_mode == "low_step_shell_refine":
                pc = surface[:, :, :3].float()
                _, voxel_idx = voxelize_from_point(pc, token_count, resolution=int(voxel_resolution))
                del pc, surface
                sf = float(getattr(vae, "scale_factor", 1.0039506158752403))
                x1 = (kl_embed * sf).detach()
                del kl_embed
                vae.to("cpu")
                _free_vram()

                if hasattr(dit, "voxel_query_res"):
                    dit.voxel_query_res = int(voxel_resolution)
                dit.to(device, dtype=torch.bfloat16)

                s_strength = min(0.95, max(0.05, float(denoise_strength)))
                sigma_start = 1.0 - s_strength
                n_steps = max(1, int(steps))
                sigmas = np.linspace(sigma_start, 1.0, n_steps, dtype=np.float32)
                scheduler.set_timesteps(n_steps, device=device, sigmas=sigmas)

                gen = torch.Generator(device=device).manual_seed(int(seed))
                noise = torch.randn(x1.shape, generator=gen, device=device, dtype=torch.bfloat16)
                latents = sigma_start * x1 + (1.0 - sigma_start) * noise
                del x1, noise

                if image is not None:
                    conditioner.to(device, dtype=torch.bfloat16)
                    image_pil = self._preprocess_image(
                        image,
                        mask_tensor=mask,
                        remove_bg=bool(remove_bg),
                        padding_ratio=0.15,
                        target_size=(1024, 1024),
                    )
                    proc_out = components["image_processor"](image_pil)
                    img_t = proc_out["image"].to(device, dtype=torch.bfloat16)
                    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        cond = conditioner(image=img_t)
                    del img_t, proc_out
                    conditioner.to("cpu")
                    _free_vram()
                else:
                    cond = {"main": torch.zeros((1, 1370, 1024), device=device, dtype=torch.bfloat16)}

                voxel_cond = voxel_idx.to(device)
                with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    for t in scheduler.timesteps:
                        t_in = (t.expand(1).to(latents.dtype) / float(scheduler.config.num_train_timesteps)).to(device)
                        v_pred = dit(latents, t_in, cond, guidance=None, voxel_cond=voxel_cond)
                        latents = scheduler.step(v_pred, t, latents).prev_sample

                del cond, voxel_cond, voxel_idx
                dit.to("cpu")
                _free_vram()

                vae.to(device, dtype=torch.bfloat16)
                with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    decoded_latents = vae.decode(latents / sf)
                    del latents
                    outputs, _ = vae.latents2mesh(
                        decoded_latents,
                        bounds=1.01,
                        mc_level=0.0,
                        num_chunks=int(chunk_size),
                        octree_resolution=int(octree_res),
                        mc_algo=algo,
                        enable_pbar=False,
                        output_on_cpu=bool(output_on_cpu),
                    )
                    del decoded_latents
            else:
                del surface
                with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    decoded_latents = vae.decode(kl_embed)
                    del kl_embed
                    outputs, _ = vae.latents2mesh(
                        decoded_latents,
                        bounds=1.01,
                        mc_level=0.0,
                        num_chunks=int(chunk_size),
                        octree_resolution=int(octree_res),
                        mc_algo=algo,
                        enable_pbar=False,
                        output_on_cpu=bool(output_on_cpu),
                    )
                    del decoded_latents

            vae.encoder.num_latents = orig_num_latents
            vae.encoder.pc_size = orig_pc_size
            vae.encoder.downsample_ratio = orig_downsample
            vae.encoder.voxel_query_res = orig_voxel_res
            if low_vram:
                vae.to("cpu")
            _free_vram()

            mesh_out_list = export_to_trimesh(outputs)
            refined_mesh = mesh_out_list[0]
        else:
            if hasattr(dit, "voxel_query_res"):
                dit.voxel_query_res = int(voxel_resolution)

            pipeline = UltraShapePipeline(
                vae=vae,
                model=dit,
                scheduler=scheduler,
                conditioner=conditioner,
                image_processor=components["image_processor"],
                device="cpu",
            )

            if low_vram:
                pipeline.enable_model_cpu_offload()
            else:
                pipeline.to(device)

            loader = SharpEdgeSurfaceLoader(
                num_sharp_points=204800,
                num_uniform_points=204800,
            )

            image_pil = self._preprocess_image(
                image,
                mask_tensor=mask,
                remove_bg=bool(remove_bg),
                padding_ratio=0.15,
                target_size=(1024, 1024),
            )

            surface = loader(orig_mesh.copy(), normalize_scale=norm_scale).to(device, dtype=torch.float16)
            pc = surface[:, :, :3]
            _, voxel_idx = voxelize_from_point(pc, token_count, resolution=int(voxel_resolution))
            del surface, pc

            gen_device = "cpu" if low_vram else device
            generator = torch.Generator(gen_device).manual_seed(int(seed))

            import comfy.utils
            import comfy.model_management

            pbar = comfy.utils.ProgressBar(int(steps))

            def callback_fn(step, timestep, latents_or_outputs):
                comfy.model_management.throw_exception_if_processing_interrupted()
                pbar.update(1)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                mesh_out_list = pipeline(
                    image=image_pil,
                    voxel_cond=voxel_idx,
                    generator=generator,
                    guidance_scale=float(guidance_scale),
                    box_v=1.01,
                    mc_level=0.0,
                    mc_algo=algo,
                    octree_resolution=int(octree_res),
                    num_chunks=int(chunk_size),
                    num_inference_steps=int(steps),
                    output_on_cpu=bool(output_on_cpu),
                    adaptive_cfg=bool(adaptive_cfg),
                    cfg_min=float(cfg_min),
                    cfg_max=float(cfg_max),
                    cfg_start_ratio=float(cfg_start_ratio),
                    cfg_end_ratio=float(cfg_end_ratio),
                    callback=callback_fn,
                    callback_steps=1,
                )

            if isinstance(mesh_out_list, tuple):
                # UltraShapePipeline returns (meshes, latents)
                mesh_out_list = mesh_out_list[0]
            refined_mesh = mesh_out_list[0] if isinstance(mesh_out_list, list) else mesh_out_list
            _free_vram()

        if refined_mesh is None:
            # the surface extractor prints its own traceback and returns None (most often out of GPU memory)
            raise RuntimeError(
                f"UltraShape surface extraction failed at octree_res={int(octree_res)} with {algo} "
                "(see the traceback above; usually out of GPU memory). Try a lower octree_res, "
                "an isoext_mc_* extractor, or enable low_vram."
            )

        if remove_floaters:
            try:
                floater_remover = FloaterRemover()
                refined_mesh = floater_remover(refined_mesh)
            except Exception as e:
                print(f"Warning: floater removal failed ({e})")

        if restore_world_scale and world_scale_max > 0:
            v_norm = np.asarray(refined_mesh.vertices, dtype=np.float32)
            v_world = v_norm * (world_scale_max / (2.0 * norm_scale)) + world_center
            refined_mesh = trimesh.Trimesh(
                vertices=v_world.astype(np.float32),
                faces=np.asarray(refined_mesh.faces, dtype=np.int32),
                process=False,
            )

        if backproject_mode != "none" and project_to_original_surface_gpu is not None:
            target_orig = orig_mesh
            if not restore_world_scale and world_scale_max > 0:
                v_orig_norm = (v_orig - world_center) * ((2.0 * norm_scale) / world_scale_max)
                target_orig = trimesh.Trimesh(
                    vertices=v_orig_norm.astype(np.float32),
                    faces=np.asarray(orig_mesh.faces, dtype=np.int32),
                    process=False,
                )
            refined_mesh = project_to_original_surface_gpu(
                refined_mesh,
                target_orig,
                eps=max(0.85 / float(octree_res), 0.85 / 384.0),
                backproject_mode=backproject_mode,
            )

        _ = refined_mesh.vertex_normals
        _free_vram()
        return (refined_mesh,)


class UltraShapeLoadMesh:
    DESCRIPTION = "Loads a 3D mesh file (OBJ, GLB, STL, PLY, etc.) using Trimesh."

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "path": ("STRING", {"default": "input.glb", "tooltip": "File path to the 3D mesh."}),
            }
        }

    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("mesh",)
    FUNCTION = "load"
    CATEGORY = "UltraShape"

    def load(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Mesh file not found: {path}")
        mesh = trimesh.load(path, force="mesh", merge_primitives=True)
        return (mesh,)


class UltraShapeSaveMesh:
    DESCRIPTION = "Saves a 3D mesh (Trimesh object) to disk."

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mesh": ("TRIMESH", {"tooltip": "Trimesh object to save."}),
                "filename": ("STRING", {"default": "refined_output.glb", "tooltip": "Output filename for saved mesh."}),
            }
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "save"
    CATEGORY = "UltraShape"

    def save(self, mesh, filename):
        output_dir = folder_paths.get_output_directory()
        path = os.path.join(output_dir, filename)
        mesh.export(path)
        return ()


NODE_CLASS_MAPPINGS = {
    "UltraShapeModelLoader": UltraShapeModelLoader,
    "UltraShapeRefine": UltraShapeRefine,
    "UltraShapeLoadMesh": UltraShapeLoadMesh,
    "UltraShapeSaveMesh": UltraShapeSaveMesh,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UltraShapeModelLoader": "Load UltraShape Model",
    "UltraShapeRefine": "Refine Mesh (UltraShape)",
    "UltraShapeLoadMesh": "Load Trimesh",
    "UltraShapeSaveMesh": "Save Trimesh",
}
