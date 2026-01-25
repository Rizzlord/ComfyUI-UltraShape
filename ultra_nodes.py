import os
import sys
import torch
import numpy as np
import trimesh
from PIL import Image
from omegaconf import OmegaConf
import folder_paths
from huggingface_hub import hf_hub_download

# Ensure UltraShape-1.0 is in sys.path
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
    ULTRASHAPE_IMPORT_ERROR = None
except ImportError as e:
    ULTRASHAPE_IMPORT_ERROR = e
    print(f"Error importing UltraShape modules: {e}")
    # We might want to handle this gracefully if dependencies are missing during startup
    pass

# Register model folder
ultrashape_models_dir = os.path.join(folder_paths.models_dir, "ultrashape")
if not os.path.exists(ultrashape_models_dir):
    os.makedirs(ultrashape_models_dir, exist_ok=True)
folder_paths.add_model_folder_path("ultrashape", ultrashape_models_dir)

class UltraShapeModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        files = folder_paths.get_filename_list("ultrashape")
        if not files:
            files = ["ultrashape_v1.pt"]
        return {
            "required": {
                "ckpt_name": (files,),
                "download_if_missing": ("BOOLEAN", {"default": True}),
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
                        filename="ultrashape_v1.pt", # Defaulting to known filename if generic name passed
                        local_dir=ultrashape_models_dir
                    )
                    # If the downloaded file is not what we asked for (e.g. user selected something else but we forced dl), handle it.
                    # Here we assume user selects ultrashape_v1.pt or we download it.
                except Exception as e:
                    raise RuntimeError(f"Failed to download model: {e}")
            else:
                raise FileNotFoundError(f"Checkpoint {ckpt_name} not found and download disabled.")
        
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load config
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
        
        # Keep on CPU to avoid immediate OOM.
        # Refine node will handle device placement.
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
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ultrashape_model": ("ULTRASHAPE_MODEL",),
                "image": ("IMAGE",),
                "mesh": ("TRIMESH",),
                "steps": ("INT", {"default": 50, "min": 1, "max": 200}),
                "guidance_scale": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "scale": ("FLOAT", {"default": 0.99, "min": 0.1, "max": 2.0, "step": 0.01}),
                "octree_res": ("INT", {"default": 1024, "min": 64, "max": 2048, "step": 64}),
                "voxel_resolution": ("INT", {"default": 128, "min": 32, "max": 1024, "step": 32}),
                "num_latents": ("INT", {"default": 8192, "min": 1024, "max": 32768, "step": 128}),
                "chunk_size": ("INT", {"default": 2048, "min": 512, "max": 10000, "step": 512}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
                "remove_bg": ("BOOLEAN", {"default": False}),
                "low_vram": ("BOOLEAN", {"default": True}),
                "output_on_cpu": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("refined_mesh",)
    FUNCTION = "refine"
    CATEGORY = "UltraShape"

    def refine(self, ultrashape_model, image, mesh, steps, guidance_scale, scale, octree_res, voxel_resolution, num_latents, chunk_size, seed, remove_bg, low_vram, output_on_cpu):
        components = ultrashape_model["components"]
        config = ultrashape_model["config"]
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Override voxel_query_res in DiT if specified
        if hasattr(components['dit'], 'voxel_query_res'):
            components['dit'].voxel_query_res = voxel_resolution

        # Create Pipeline - start on CPU
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
        
        # Prepare Image
        image_tensor = image[0]
        image_np = (image_tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        image_pil = Image.fromarray(image_np).convert("RGBA")
        
        if remove_bg:
            rembg = BackgroundRemover()
            image_pil = rembg(image_pil)
        
        # Prepare Mesh
        # We can use GPU for voxelization for speed, then move result if needed.
        # But if VRAM is super tight, maybe voxelize on CPU?
        # Let's try voxelizing on GPU if available (it's small memory footprint compared to models)
        surface = loader(mesh, normalize_scale=scale).to(device, dtype=torch.float16)
        pc = surface[:, :, :3] # [B, N, 3]
        
        # Voxelize
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
            )
            
        refined_mesh = mesh_out_list[0]
        return (refined_mesh,)


class UltraShapeLoadMesh:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "path": ("STRING", {"default": "input.glb"}),
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
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mesh": ("TRIMESH",),
                "filename": ("STRING", {"default": "refined_output.glb"}),
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
