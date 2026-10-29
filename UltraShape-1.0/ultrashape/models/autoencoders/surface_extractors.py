# ==============================================================================
# Original work Copyright (c) 2025 Tencent.
# Modified work Copyright (c) 2025 UltraShape Team.
# 
# Modified by UltraShape on 2025.12.25
# ==============================================================================

# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

from typing import Union, Tuple, List

import numpy as np
import torch
from skimage import measure

has_cubvh = False
try:
    import cubvh
    has_cubvh = True
except (ImportError, RuntimeError) as e:
    print(f"Warning: cubvh import failed ({e}), falling back to skimage.")


class Latent2MeshOutput:
    def __init__(self, mesh_v=None, mesh_f=None):
        self.mesh_v = mesh_v
        self.mesh_f = mesh_f


def center_vertices(vertices):
    """Translate the vertices so that bounding box is centered at zero."""
    vert_min = vertices.min(dim=0)[0]
    vert_max = vertices.max(dim=0)[0]
    vert_center = 0.5 * (vert_min + vert_max)
    return vertices - vert_center


class SurfaceExtractor:
    def _compute_box_stat(self, bounds: Union[Tuple[float], List[float], float], octree_resolution: int):
        """
        Compute grid size, bounding box minimum coordinates, and bounding box size based on input 
        bounds and resolution.

        Args:
            bounds (Union[Tuple[float], List[float], float]): Bounding box coordinates or a single 
            float representing half side length.
                If float, bounds are assumed symmetric around zero in all axes.
                Expected format if list/tuple: [xmin, ymin, zmin, xmax, ymax, zmax].
            octree_resolution (int): Resolution of the octree grid.

        Returns:
            grid_size (List[int]): Grid size along each axis (x, y, z), each equal to octree_resolution + 1.
            bbox_min (np.ndarray): Minimum coordinates of the bounding box (xmin, ymin, zmin).
            bbox_size (np.ndarray): Size of the bounding box along each axis (xmax - xmin, etc.).
        """
        if isinstance(bounds, float):
            bounds = [-bounds, -bounds, -bounds, bounds, bounds, bounds]

        bbox_min, bbox_max = np.array(bounds[0:3]), np.array(bounds[3:6])
        bbox_size = bbox_max - bbox_min
        grid_size = [int(octree_resolution) + 1, int(octree_resolution) + 1, int(octree_resolution) + 1]
        return grid_size, bbox_min, bbox_size

    def run(self, *args, **kwargs):
        """
        Abstract method to extract surface mesh from grid logits.

        This method should be implemented by subclasses.

        Raises:
            NotImplementedError: Always, since this is an abstract method.
        """
        return NotImplementedError

    def __call__(self, grid_logits, **kwargs):
        """
        Process a batch of grid logits to extract surface meshes.

        Args:
            grid_logits (torch.Tensor): Batch of grid logits with shape (batch_size, ...).
            **kwargs: Additional keyword arguments passed to the `run` method.

        Returns:
            List[Optional[Latent2MeshOutput]]: List of mesh outputs for each grid in the batch.
                If extraction fails for a grid, None is appended at that position.
        """
        outputs = []
        for i in range(grid_logits.shape[0]):
            try:
                vertices, faces = self.run(grid_logits[i], **kwargs)
                vertices = vertices.astype(np.float32)
                faces = np.ascontiguousarray(faces)
                outputs.append(Latent2MeshOutput(mesh_v=vertices, mesh_f=faces))

            except Exception:
                import traceback
                traceback.print_exc()
                outputs.append(None)

        return outputs


def get_sparse_valid_voxels(grid_logit: torch.Tensor):

    if not isinstance(grid_logit, torch.Tensor):
        raise TypeError("Input must be a PyTorch tensor.")
    if grid_logit.dim() != 3 or grid_logit.shape[0] != grid_logit.shape[1] or grid_logit.shape[0] != grid_logit.shape[2]:
        raise ValueError("Input tensor must have shape (N, N, N)")

    N = grid_logit.shape[0]
    device = grid_logit.device

    # Chunk processing to save memory
    chunk_size = 128

    all_sparse_coords = []
    all_sparse_logits = []

    # Process in chunks along x-axis
    for start_x in range(0, N - 1, chunk_size):
        end_x = min(start_x + chunk_size, N - 1)

        # Determine slice range including +1 for neighbor checks
        # slice_end needs to be end_x + 1 to include the neighbors for the last voxel in chunk
        slice_end = end_x + 1

        chunk = grid_logit[start_x:slice_end, :, :]
        nan_mask = torch.isnan(chunk)

        # Compute mask for this chunk (valid voxels are 0 to end_x - start_x)
        # Note: chunk shape is [D_chunk, N, N].
        # We want to check validity for [0..D_chunk-1, :-1, :-1]

        sub_nan_mask = nan_mask

        # Validity check requires looking at i and i+1
        # Invalid if ANY corner is NaN
        invalid_voxel_mask = (
            sub_nan_mask[:-1, :-1, :-1] |
            sub_nan_mask[1:, :-1, :-1]  |
            sub_nan_mask[:-1, 1:, :-1]  |
            sub_nan_mask[:-1, :-1, 1:]  |
            sub_nan_mask[:-1, 1:, 1:]   |
            sub_nan_mask[1:, :-1, 1:]   |
            sub_nan_mask[1:, 1:, :-1]   |
            sub_nan_mask[1:, 1:, 1:]
        )

        valid_voxel_mask = ~invalid_voxel_mask

        # Get local coordinates
        local_coords = valid_voxel_mask.nonzero(as_tuple=False)

        if local_coords.shape[0] > 0:
            lx, ly, lz = local_coords[:, 0], local_coords[:, 1], local_coords[:, 2]

            # Extract logits using local indices on the chunk
            # v0 is at lx, v1 is at lx+1, etc.
            sparse_vertex_logits = torch.stack([
                chunk[lx,     ly,     lz],     # v0
                chunk[lx + 1, ly,     lz],     # v1
                chunk[lx + 1, ly + 1, lz],     # v2
                chunk[lx,     ly + 1, lz],     # v3
                chunk[lx,     ly,     lz + 1], # v4
                chunk[lx + 1, ly,     lz + 1], # v5
                chunk[lx + 1, ly + 1, lz + 1], # v6
                chunk[lx,     ly + 1, lz + 1]  # v7
            ], dim=1)

            # Convert local coords to global coords
            # x coordinate needs offset added
            global_coords = local_coords.clone()
            global_coords[:, 0] += start_x

            all_sparse_coords.append(global_coords)
            all_sparse_logits.append(sparse_vertex_logits)

        # Free memory
        del chunk, nan_mask, invalid_voxel_mask, valid_voxel_mask, local_coords

    if not all_sparse_coords:
        return torch.empty((0, 3), dtype=torch.long, device=device), torch.empty((0, 8), dtype=grid_logit.dtype, device=device)

    sparse_coords = torch.cat(all_sparse_coords, dim=0)
    sparse_vertex_logits = torch.cat(all_sparse_logits, dim=0)

    return sparse_coords, sparse_vertex_logits


class MCSurfaceExtractor(SurfaceExtractor):
    def run(self, grid_logit, *, mc_level, bounds, octree_resolution, **kwargs):
        """
        Extract surface mesh using the Marching Cubes algorithm.

        Args:
            grid_logit (torch.Tensor): 3D grid logits tensor representing the scalar field.
            mc_level (float): The level (iso-value) at which to extract the surface.
            bounds (Union[Tuple[float], List[float], float]): Bounding box coordinates or half side length.
            octree_resolution (int): Resolution of the octree grid.
            **kwargs: Additional keyword arguments (ignored).

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple containing:
                - vertices (np.ndarray): Extracted mesh vertices, scaled and translated to bounding 
                  box coordinates.
                - faces (np.ndarray): Extracted mesh faces (triangles).
        """

        grid_logit = grid_logit.detach()
        
        use_cubvh_local = False
        if has_cubvh:
            try:
                sparse_coords, sparse_logits = get_sparse_valid_voxels(grid_logit)
                # Convert to float32 only for the sparse set
                vertices, faces = cubvh.sparse_marching_cubes(sparse_coords, sparse_logits.float(), mc_level)
                vertices, faces = vertices.cpu().numpy(), faces.cpu().numpy()
                use_cubvh_local = True
            except Exception as e:
                print(f"Warning: cubvh execution failed ({e}), falling back to skimage.")
        
        if not use_cubvh_local:
            # Fallback to skimage.measure.marching_cubes
            grid_logit_np = grid_logit.cpu().float().numpy()
            vertices, faces, normals, _ = measure.marching_cubes(
                 grid_logit_np,
                 mc_level, 
                 method="lewiner", 
                 mask=(~np.isnan(grid_logit_np))
            )
        
        grid_size, bbox_min, bbox_size = self._compute_box_stat(bounds, octree_resolution)
        vertices = vertices / grid_size * bbox_size + bbox_min
        return vertices, faces


class DMCSurfaceExtractor(SurfaceExtractor):
    def run(self, grid_logit, *, octree_resolution, **kwargs):
        """
        Extract surface mesh using Differentiable Marching Cubes (DMC) algorithm.

        Args:
            grid_logit (torch.Tensor): 3D grid logits tensor representing the scalar field.
            octree_resolution (int): Resolution of the octree grid.
            **kwargs: Additional keyword arguments (ignored).

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple containing:
                - vertices (np.ndarray): Extracted mesh vertices, centered and converted to numpy.
                - faces (np.ndarray): Extracted mesh faces (triangles), with reversed vertex order.
        
        Raises:
            ImportError: If the 'diso' package is not installed.
        """
        device = grid_logit.device
        if not hasattr(self, 'dmc'):
            try:
                from diso import DiffDMC
                self.dmc = DiffDMC(dtype=torch.float32).to(device)
            except:
                raise ImportError("Please install diso via `pip install diso`, or set mc_algo to 'mc'")
        sdf = -grid_logit / octree_resolution
        sdf = sdf.to(torch.float32).contiguous()
        verts, faces = self.dmc(sdf, deform=None, return_quads=False, normalize=True)
        verts = center_vertices(verts)
        vertices = verts.detach().cpu().numpy()
        faces = faces.detach().cpu().numpy()[:, ::-1]
        return vertices, faces


def _sparse_marching_cubes(sdf, bmin, bmax, method, level, chunk=1 << 22):
    """Marching cubes on only the cells that cross the level (isoext SparseGrid).

    Same mesh as dense marching cubes, but no float32 copy of the whole grid and no dense per-cell
    buffers, which is what lets octree 1024 fit in 16 GB.
    """
    import isoext

    n = [int(x) for x in sdf.shape]
    m = [x - 1 for x in n]
    ins = sdf < level
    any_in = ins[:m[0], :m[1], :m[2]].clone()
    all_in = any_in.clone()
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                s = ins[dx:m[0] + dx, dy:m[1] + dy, dz:m[2] + dz]
                any_in |= s
                all_in &= s
    del ins
    any_in &= all_in.logical_not_()
    del all_in
    cells = torch.nonzero(any_in)
    del any_in
    off = torch.tensor([[a, b, c] for a in (0, 1) for b in (0, 1) for c in (0, 1)], device=sdf.device)
    vals = torch.empty((len(cells), 8), dtype=torch.float32, device=sdf.device)
    for s0 in range(0, len(cells), chunk):
        cn = cells[s0:s0 + chunk].unsqueeze(1) + off.unsqueeze(0)
        vals[s0:s0 + chunk] = sdf[cn[..., 0], cn[..., 1], cn[..., 2]].float()
        del cn
    lin = ((cells[:, 0] * m[1] + cells[:, 1]) * m[2] + cells[:, 2]).to(torch.int32).contiguous()
    del cells
    # isoext allocates with cudaMalloc directly, so hand PyTorch's cached blocks back first
    torch.cuda.empty_cache()
    grid = isoext.SparseGrid(n, bmin, bmax)
    grid.add_cells(lin)
    grid.set_values(vals.contiguous())
    del lin, vals
    v, f = isoext.marching_cubes(grid, level=level, method=method)
    del grid
    return v, f


class IsoextSurfaceExtractor(SurfaceExtractor):
    # above this many grid points (octree > 640) the marching-cubes variants run on a sparse grid
    LEAN_GRID_POINTS = 641 ** 3
    SPARSE_METHODS = {
        "isoext_mc_lewiner": "lewiner",
        "isoext_mc_vega": "vega",
        "isoext_mc_nagae": "nagae",
        "isoext_mc_lorensen": "lorensen",
    }

    def __init__(self, algo="isoext_mc_lewiner"):
        self.algo = algo

    def run(self, grid_logit, *, mc_level=0.0, bounds=1.01, octree_resolution=256, **kwargs):
        _, bbox_min, bbox_size = self._compute_box_stat(bounds, octree_resolution)
        bmin = [float(bbox_min[0]), float(bbox_min[1]), float(bbox_min[2])]
        bmax = [
            float(bbox_min[0] + bbox_size[0]),
            float(bbox_min[1] + bbox_size[1]),
            float(bbox_min[2] + bbox_size[2]),
        ]

        # One owned copy of the grid, prepared in place: at octree 1024 every full-size temporary is
        # 4 GB in float32, and the old out-of-place version needed four of them at once. isoext only
        # takes CUDA tensors, and with output_on_cpu the decoder hands the grid over on the CPU.
        lean = grid_logit.numel() > self.LEAN_GRID_POINTS and self.algo in self.SPARSE_METHODS
        sdf = grid_logit.detach().to("cuda", torch.float16 if lean else torch.float32, copy=True).contiguous()
        sdf.nan_to_num_(nan=-10.0).neg_()
        sdf[0, :, :] = 10.0
        sdf[-1, :, :] = 10.0
        sdf[:, 0, :] = 10.0
        sdf[:, -1, :] = 10.0
        sdf[:, :, 0] = 10.0
        sdf[:, :, -1] = 10.0

        level = float(-mc_level)
        if abs(level) < 1e-6:
            level = 1.234567e-5
        # push values lying exactly on the level off it, one slab at a time
        for s0 in range(0, sdf.shape[0], 64):
            slab = sdf[s0:s0 + 64]
            slab[(slab.float() - level).abs_() < 1e-6] += 1e-6

        if lean:
            v, f = _sparse_marching_cubes(sdf, bmin, bmax, self.SPARSE_METHODS[self.algo], level)
        else:
            v, f = self._dense(sdf, bmin, bmax, level)
        del sdf
        v_np = v.detach().cpu().numpy().astype(np.float32)
        f_np = f.detach().cpu().numpy().astype(np.int32)
        del v, f
        return self._keep_outer_components(v_np, f_np)

    def _dense(self, sdf, bmin, bmax, level):
        import isoext

        Rx, Ry, Rz = int(sdf.shape[0]), int(sdf.shape[1]), int(sdf.shape[2])
        ugrid = isoext.UniformGrid([Rx, Ry, Rz], bmin, bmax)
        ugrid.set_values(sdf)

        if self.algo == "isoext_mc_vega":
            v, f = isoext.marching_cubes(ugrid, level=level, method="vega")
        elif self.algo == "isoext_mc_nagae":
            v, f = isoext.marching_cubes(ugrid, level=level, method="nagae")
        elif self.algo == "isoext_mc_lorensen":
            v, f = isoext.marching_cubes(ugrid, level=level, method="lorensen")
        elif self.algo == "isoext_dmc_vega":
            v, f = isoext.dual_marching_cubes(ugrid, level=level, method="vega")
        elif self.algo == "isoext_dmc_lewiner":
            v, f = isoext.dual_marching_cubes(ugrid, level=level, method="lewiner")
        elif self.algo == "isoext_dc_ju":
            v, f = isoext.dual_contouring(ugrid, level=level, method="ju")
        elif self.algo == "isoext_dc_carrera":
            v, f = isoext.dual_contouring(ugrid, level=level, method="carrera")
        elif self.algo == "isoext_mt":
            v, f = isoext.marching_tetrahedra(ugrid, level=level)
        elif self.algo == "isoext_sn":
            v, f = isoext.surface_nets(ugrid, level=level)
        else:
            v, f = isoext.marching_cubes(ugrid, level=level, method="lewiner")
        del ugrid
        return v, f

    @staticmethod
    def _keep_outer_components(v_np, f_np):
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components

        if len(f_np) > 0:
            rows = np.concatenate([f_np[:, 0], f_np[:, 1], f_np[:, 2]])
            cols = np.concatenate([f_np[:, 1], f_np[:, 2], f_np[:, 0]])
            adj = coo_matrix((np.ones_like(rows), (rows, cols)), shape=(len(v_np), len(v_np)))
            n_c, c_ids = connected_components(adj, directed=False)
            if n_c > 1:
                u_c, counts = np.unique(c_ids[f_np[:, 0]], return_counts=True)
                order = np.argsort(-counts)
                max_faces = counts[order[0]]
                kept_v = []
                kept_f = []
                v_offset = 0
                for idx in order:
                    if counts[idx] < max(1000, int(0.02 * max_faces)):
                        break
                    cid = u_c[idx]
                    face_mask = c_ids[f_np[:, 0]] == cid
                    f_sub = f_np[face_mask]
                    used = np.zeros(len(v_np), dtype=bool)
                    used[f_sub.reshape(-1)] = True
                    remap = np.full(len(v_np), -1, dtype=np.int32)
                    remap[used] = np.arange(used.sum(), dtype=np.int32)
                    cv = v_np[used]
                    cf = remap[f_sub]
                    vt = torch.as_tensor(cv, dtype=torch.float64, device="cuda")
                    ft = torch.as_tensor(cf, dtype=torch.int64, device="cuda")
                    svol = float((torch.sum(vt[ft[:, 0]] * torch.cross(vt[ft[:, 1]], vt[ft[:, 2]], dim=-1)) / 6.0).item())
                    del vt, ft
                    if svol > 0.0:
                        kept_v.append(cv)
                        kept_f.append(cf + v_offset)
                        v_offset += len(cv)
                if len(kept_v) > 0:
                    v_np = np.vstack(kept_v).astype(np.float32)
                    f_np = np.vstack(kept_f).astype(np.int32)

        return v_np, f_np[:, ::-1]


SurfaceExtractors = {
    'mc': MCSurfaceExtractor,
    'dmc': DMCSurfaceExtractor,
    'isoext_mc_lewiner': lambda: IsoextSurfaceExtractor('isoext_mc_lewiner'),
    'isoext_mc_vega': lambda: IsoextSurfaceExtractor('isoext_mc_vega'),
    'isoext_mc_nagae': lambda: IsoextSurfaceExtractor('isoext_mc_nagae'),
    'isoext_mc_lorensen': lambda: IsoextSurfaceExtractor('isoext_mc_lorensen'),
    'isoext_dmc_vega': lambda: IsoextSurfaceExtractor('isoext_dmc_vega'),
    'isoext_dmc_lewiner': lambda: IsoextSurfaceExtractor('isoext_dmc_lewiner'),
    'isoext_dc_ju': lambda: IsoextSurfaceExtractor('isoext_dc_ju'),
    'isoext_dc_carrera': lambda: IsoextSurfaceExtractor('isoext_dc_carrera'),
    'isoext_mt': lambda: IsoextSurfaceExtractor('isoext_mt'),
    'isoext_sn': lambda: IsoextSurfaceExtractor('isoext_sn'),
}

