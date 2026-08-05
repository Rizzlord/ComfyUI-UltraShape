import os
import sys
import torch
import numpy as np
import trimesh
from PIL import Image
from omegaconf import OmegaConf
import folder_paths
from huggingface_hub import hf_hub_download

current_dir = os.path.dirname(os.path.abspath(__file__))
ultrashape_path = os.path.join(current_dir, "UltraShape-1.0")
if ultrashape_path not in sys.path:
    sys.path.insert(0, ultrashape_path)

try:
    from ultrashape.rembg import BackgroundRemover
    from ultrashape.utils.misc import instantiate_from_config
    from ultrashape.surface_loaders import SharpEdgeSurfaceLoader
    from ultrashape.utils import voxelize_from_point
    from ultrashape.pipelines import UltraShapePipeline
    from ultrashape import FloaterRemover
    ULTRASHAPE_IMPORT_ERROR = None
except ImportError as e:
    ULTRASHAPE_IMPORT_ERROR = e
    print(f"Error importing UltraShape modules: {e}")
    pass

ultrashape_models_dir = os.path.join(folder_paths.models_dir, "ultrashape")
if not os.path.exists(ultrashape_models_dir):
    os.makedirs(ultrashape_models_dir, exist_ok=True)
folder_paths.add_model_folder_path("ultrashape", ultrashape_models_dir)

class UltraShapeModelLoader:
    DESCRIPTION = "Loads the UltraShape model checkpoint and instantiates VAE, DiT, Conditioner, Scheduler, and Image Processor."

    @classmethod
    def INPUT_TYPES(s):
        files = folder_paths.get_filename_list("ultrashape")
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
            
        ckpt_path = folder_paths.get_full_path("ultrashape", ckpt_name)
        
        if not ckpt_path or not os.path.exists(ckpt_path):
            if download_if_missing:
                print(f"Downloading {ckpt_name} to {ultrashape_models_dir}...")
                try:
                    ckpt_path = hf_hub_download(
                        repo_id="infinith/UltraShape",
                        filename="ultrashape_v1.pt",
                        local_dir=ultrashape_models_dir
                    )
                except Exception as e:
                    raise RuntimeError(f"Failed to download model: {e}")
            else:
                raise FileNotFoundError(f"Checkpoint {ckpt_name} not found and download disabled.")
        
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
        
        print(f"Loading weights from {ckpt_path}...")
        weights = torch.load(ckpt_path, map_location='cpu')
        
        vae.load_state_dict(weights['vae'], strict=True)
        dit.load_state_dict(weights['dit'], strict=True)
        conditioner.load_state_dict(weights['conditioner'], strict=True)
        
        vae.eval()
        dit.eval()
        conditioner.eval()
        
        if hasattr(vae, 'enable_flashvdm_decoder'):
            vae.enable_flashvdm_decoder()
            
        components = {
            "vae": vae,
            "dit": dit,
            "conditioner": conditioner,
            "scheduler": scheduler,
            "image_processor": image_processor,
        }
        
        return ({"components": components, "config": config},)

class UltraShapeRefine:
    DESCRIPTION = "Refines an input 3D mesh conditioned on a reference image using UltraShape with optional adaptive CFG schedule."

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ultrashape_model": ("ULTRASHAPE_MODEL", {"tooltip": "Loaded UltraShape model components."}),
                "image": ("IMAGE", {"tooltip": "Reference image for 3D shape refinement."}),
                "mesh": ("TRIMESH", {"tooltip": "Input coarse 3D mesh (Trimesh object) to be refined."}),
                "steps": ("INT", {"default": 50, "min": 1, "max": 200, "tooltip": "Number of diffusion sampling steps."}),
                "guidance_scale": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 20.0, "step": 0.1, "tooltip": "Classifier-free guidance scale (default CFG max if adaptive CFG is enabled)."}),
                "scale": ("FLOAT", {"default": 0.99, "min": 0.1, "max": 2.0, "step": 0.01, "tooltip": "Normalization scale for input surface point cloud."}),
                "octree_res": ("INT", {"default": 1024, "min": 64, "max": 2048, "step": 64, "tooltip": "Resolution for Octree surface extraction during marching cubes."}),
                "voxel_resolution": ("INT", {"default": 128, "min": 32, "max": 1024, "step": 32, "tooltip": "Voxel resolution for conditioning query grid."}),
                "num_latents": ("INT", {"default": 8192, "min": 1024, "max": 32768, "step": 128, "tooltip": "Number of surface point tokens to sample and voxelize."}),
                "chunk_size": ("INT", {"default": 2048, "min": 512, "max": 10000, "step": 512, "tooltip": "Chunk size for surface extraction decoding to save memory."}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff, "tooltip": "Random seed for reproducible sampling."}),
                "remove_bg": ("BOOLEAN", {"default": False, "tooltip": "Automatically remove background from input image before conditioning."}),
                "remove_floaters": ("BOOLEAN", {"default": False, "tooltip": "Remove disconnected small mesh floaters post-processing."}),
                "low_vram": ("BOOLEAN", {"default": True, "tooltip": "Enable CPU offloading during inference to save VRAM."}),
                "output_on_cpu": ("BOOLEAN", {"default": True, "tooltip": "Keep intermediate output mesh tensors on CPU to prevent OOM."}),
                "adaptive_cfg": ("BOOLEAN", {"default": False, "tooltip": "Enable adaptive CFG scheduler (sinusoidal guidance scale schedule)."}),
                "cfg_min": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 30.0, "step": 0.1, "tooltip": "Minimum guidance scale for adaptive CFG."}),
                "cfg_max": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 30.0, "step": 0.1, "tooltip": "Maximum guidance scale for adaptive CFG."}),
                "cfg_start_ratio": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Start step ratio for adaptive CFG schedule."}),
                "cfg_end_ratio": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "End step ratio for adaptive CFG schedule."}),
            }
        }

    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("refined_mesh",)
    FUNCTION = "refine"
    CATEGORY = "UltraShape"

    def refine(self, ultrashape_model, image, mesh, steps, guidance_scale, scale, octree_res, voxel_resolution, num_latents, chunk_size, seed, remove_bg, remove_floaters, low_vram, output_on_cpu, adaptive_cfg=False, cfg_min=1.0, cfg_max=5.0, cfg_start_ratio=0.0, cfg_end_ratio=1.0):
        components = ultrashape_model["components"]
        config = ultrashape_model["config"]
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if hasattr(components['dit'], 'voxel_query_res'):
            components['dit'].voxel_query_res = voxel_resolution

        pipeline = UltraShapePipeline(
            vae=components['vae'],
            model=components['dit'],
            scheduler=components['scheduler'],
            conditioner=components['conditioner'],
            image_processor=components['image_processor'],
            device='cpu'
        )
        
        if low_vram:
            pipeline.enable_model_cpu_offload()
        else:
            pipeline.to(device)
            
        print(f"Initializing Surface Loader (Token Num: {num_latents})...")
        loader = SharpEdgeSurfaceLoader(
            num_sharp_points=204800,
            num_uniform_points=204800,
        )
        
        image_tensor = image[0]
        image_np = (image_tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        image_pil = Image.fromarray(image_np).convert("RGBA")
        
        if remove_bg:
            rembg = BackgroundRemover()
            image_pil = rembg(image_pil)
        
        surface = loader(mesh, normalize_scale=scale).to(device, dtype=torch.float16)
        pc = surface[:, :, :3]
        
        _, voxel_idx = voxelize_from_point(pc, num_latents, resolution=voxel_resolution)
        
        print("Running diffusion process...")
        gen_device = "cpu" if low_vram else device
        generator = torch.Generator(gen_device).manual_seed(seed)
        
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            mesh_out_list, _ = pipeline(
                image=image_pil,
                voxel_cond=voxel_idx,
                generator=generator,
                guidance_scale=guidance_scale,
                box_v=1.0,
                mc_level=0.0,
                octree_resolution=octree_res,
                num_chunks=chunk_size,
                num_inference_steps=steps,
                output_on_cpu=output_on_cpu,
                adaptive_cfg=adaptive_cfg,
                cfg_min=cfg_min,
                cfg_max=cfg_max,
                cfg_start_ratio=cfg_start_ratio,
                cfg_end_ratio=cfg_end_ratio,
            )
            
        refined_mesh = mesh_out_list[0]
        if remove_floaters:
            floater_remover = FloaterRemover()
            refined_mesh = floater_remover(refined_mesh)
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
    "UltraShapeSaveMesh": UltraShapeSaveMesh
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UltraShapeModelLoader": "Load UltraShape Model",
    "UltraShapeRefine": "Refine Mesh (UltraShape)",
    "UltraShapeLoadMesh": "Load Trimesh",
    "UltraShapeSaveMesh": "Save Trimesh"
}
