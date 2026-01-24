from typing import Dict, Optional, Callable, TypeVar, List, Union, Tuple
import torch
from torch import nn, Tensor, scatter_add
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Data, Batch
import lightning.pytorch as pl
from torchdiffeq import odeint
from torch_geometric.utils import scatter

from flow_matching.utils import get_shortest_path_fast_batched_x_1, rmsd_error, rmsd_loss, \
    spring_loss, get_substruct_matches, get_min_dmae_match_torch_batch, \
        kabsch_align_batched, align_x0_to_reference_batched, compute_neb_path_metrics
import numpy as np

from gotennet.models.components.outputs import Atomwise3DOut
from gotennet.models.representation.gotennet import GotenNet
import numpy as np

from gotennet.models.components.outputs import Atomwise3DOut
from gotennet.models.representation.gotennet import GotenNet


class FlowModule(pl.LightningModule):
    def __init__(
            self,
            representation: GotenNet,
            lr: float = 5e-4,
            lr_decay: float = 0.5,
            lr_patience: int = 100,
            lr_minlr: float = 1e-6,
            lr_monitor: str = "validation/ema_val_loss",
            chain_scheduler: Optional[Callable] = None,
            weight_decay: float = 0.01,
            num_steps: int = 10,
            num_samples: int = 1,
            seed: int = 1,
            ema_decay: float = 0.9,
            dataset_meta: Optional[Dict[str, Dict[int, Tensor]]] = None,
            output: Optional[Dict] = None,
            scheduler: Optional[Callable] = None,
            save_predictions: Optional[bool] = None,
            input_contribution: float = 1,
            task_config: Optional[Dict] = None,
            lr_warmup_steps: int = 0,
            schedule_free: bool = False,
            use_ema: bool = False,
            neb_img_subset: Optional[list] = None,
            sample_method: str = "gaussian",
            save_all_samples: bool = True,
            lambda_spring: float = 0.0,
            spring_loss_warmup_steps: int = 0,
            spring_loss_schedule: str = "linear",
            preserve_path_coherence: bool = False,
            rmse_extended_metrics: bool = False,
            **kwargs
    ):
        super().__init__()
        self.representation = representation
        self.atomwise_3D_out_layer = Atomwise3DOut(n_in=representation.hidden_dim, n_hidden=output['n_hidden'],
                                                   activation=F.silu)

        self.lr = lr
        self.lr_decay = lr_decay
        self.lr_patience = lr_patience
        self.lr_monitor = lr_monitor
        self.weight_decay = weight_decay
        self.dataset_meta = dataset_meta

        self.num_steps = num_steps
        self.num_samples = num_samples

        self.use_ema = use_ema
        self.schedule_free = schedule_free
        self.lr_warmup_steps = lr_warmup_steps
        self.lr_minlr = lr_minlr
        self.chain_scheduler = chain_scheduler
        self.input_contribution = input_contribution
        self.save_predictions = save_predictions
        self.save_all_samples = save_all_samples

        # NEB spring force parameters
        self.lambda_spring = lambda_spring
        self.spring_loss_warmup_steps = spring_loss_warmup_steps
        self.spring_loss_schedule = spring_loss_schedule
        self.preserve_path_coherence = preserve_path_coherence
        self.rmse_extended_metrics = rmse_extended_metrics

        self.lambda_cf = 0.0
        self.seed = seed
        print("FM has seed", self.seed, "-----------------------------")

        self.scheduler = scheduler

        # Note: torch.compile() disabled due to incompatibility with torch_cluster custom ops
        # if hasattr(torch, 'compile'):
        #     self.representation = torch.compile(representation)

        self.save_hyperparameters(ignore=['representation'])

        # save results in test_step
        self.results_R = []
        self.neb_img_subset = neb_img_subset
    
    def get_spring_loss_weight(self) -> float:
        """
        Compute annealed weight for spring loss based on training progress.
        Implements warmup followed by optional scheduling.
        
        Returns:
            Weight multiplier in [0, 1]
        """
        global_step = self.global_step
        
        if global_step < self.spring_loss_warmup_steps:
            # Linear warmup phase
            weight = global_step / max(self.spring_loss_warmup_steps, 1)
        elif self.spring_loss_schedule == "linear":
            # Already at full weight after warmup
            weight = 1.0
        elif self.spring_loss_schedule == "cosine":
            # Cosine annealing (optional: for gentle increase)
            progress = (global_step - self.spring_loss_warmup_steps) / max(
                self.trainer.max_steps - self.spring_loss_warmup_steps, 1
            )
            progress = min(progress, 1.0)
            weight = 0.5 * (1 + torch.cos(torch.tensor(torch.pi * progress))).item()
        else:
            weight = 1.0
        
        return weight
    
    ### New methods for NEB image conditioning
    
    def get_perturbed_flow_point_and_time_with_neb(self, batch: Data):

        num_atoms, dim, neb_img = batch.pos_N_3_I.shape
        neb_img_subset = self.neb_img_subset # Determine which NEB images to use

        # Accept string sentinel 'all' or 'auto'
        if isinstance(neb_img_subset, str):
            if neb_img_subset.lower() in ['all', 'auto']:
                neb_img_subset = list(range(0, neb_img))
            else:
                raise ValueError(f"Invalid neb_img_subset string: {neb_img_subset}")
            
        # Case 1: Empty subset or None means use only TS (one image per reaction)
        if neb_img_subset is None or len(neb_img_subset) == 0:
            # Get TS indices for each reaction in batch
            if hasattr(batch, 'ts_idx'):
                ts_indices = batch.ts_idx
            else:
                # Fallback to middle image
                ts_indices = [neb_img // 2] * batch.num_graphs
                print("No ts_idx provided; using middle image for TS images:", ts_indices)

            # Verify we have exactly one TS index per reaction
            assert len(ts_indices) == batch.num_graphs, \
                f"Expected {batch.num_graphs} TS indices, got {len(ts_indices)}"

            # Extract positions for each reaction's TS
            x_1_N_3_list = []
            start_idx = 0
            for graph_idx, ts_idx in enumerate(ts_indices):
                # Find atoms belonging to this reaction
                mask = batch.batch == graph_idx
                num_atoms_in_graph = mask.sum().item()
                end_idx = start_idx + num_atoms_in_graph

                # Extract TS positions for this reaction
                x_1_N_3_list.append(batch.pos_N_3_I[start_idx:end_idx, :, ts_idx])
                start_idx = end_idx

            x_1_N_3 = torch.cat(x_1_N_3_list, dim=0)
            x_0_N_3 = torch.randn_like(x_1_N_3, device=self.device)

            t_G = torch.rand(batch.num_graphs, 1, device=self.device)
            t_N = t_G[batch.batch]

            x_1_aligned_N_3 = get_shortest_path_fast_batched_x_1(x_0_N_3, x_1_N_3, batch)
            x_t_N_3 = (1 - t_N) * x_0_N_3 + t_N * x_1_aligned_N_3
            dx_dt_N_3 = x_1_aligned_N_3 - x_0_N_3

            # Return with single image dimension (I=1)
            x_t_I_N_3 = x_t_N_3.unsqueeze(0)  # [1, N, 3]
            dx_dt_I_N_3 = dx_dt_N_3.unsqueeze(0)  # [1, N, 3]
            t_I_G = t_G.unsqueeze(0)  # [1, G, 1]
            neb_imgs_I_G = torch.tensor(ts_indices, device=self.device).unsqueeze(0)  # [1, G]

        else:
            # Collect all flow data for all NEB images
            x_t_I_N_3 = []
            dx_dt_I_N_3 = []
            t_G_I = []
            neb_imgs_I_G = []
            # Sample x_0 for first image (reference frame)
            x_0_ref_N_3 = None
            for neb_idx in neb_img_subset:
                x_1_N_3 = batch.pos_N_3_I[:, :, neb_idx]
                
                if x_0_ref_N_3 is None:
                    # First image: sample x_0 normally
                    x_0_N_3 = torch.randn_like(x_1_N_3, device=self.device)
                    x_0_ref_N_3 = x_0_N_3  # Save as reference
                else:
                    # Subsequent images: sample and align to reference
                    x_0_new_N_3 = torch.randn_like(x_1_N_3, device=self.device)

                    # Align x_0_new to x_0_ref using Kabsch per molecule
                    x_0_N_3 = align_x0_to_reference_batched(
                        x_0_new_N_3, x_0_ref_N_3, batch.batch, batch.num_graphs
                    )

                t_G = torch.rand(batch.num_graphs, 1, device=self.device)
                t_N = t_G[batch.batch]

                x_1_aligned_N_3 = get_shortest_path_fast_batched_x_1(x_0_N_3, x_1_N_3, batch)
                x_t_N_3 = (1 - t_N) * x_0_N_3 + t_N * x_1_aligned_N_3
                dx_dt_N_3 = x_1_aligned_N_3 - x_0_N_3

                x_t_I_N_3.append(x_t_N_3)
                dx_dt_I_N_3.append(dx_dt_N_3)
                t_G_I.append(t_G)
                neb_imgs_I_G.append(torch.full((batch.num_graphs,), neb_idx, dtype=torch.long, device=self.device))

            # Stack all results
            x_t_I_N_3 = torch.stack(x_t_I_N_3)
            dx_dt_I_N_3 = torch.stack(dx_dt_I_N_3)
            t_I_G = torch.stack(t_G_I)
            neb_imgs_I_G = torch.stack(neb_imgs_I_G)

        return x_t_I_N_3, dx_dt_I_N_3, t_I_G, neb_imgs_I_G

    def train_val_step_with_neb(self, batch: Data) -> Tensor:
        x_t_I_N_3, dx_dt_I_N_3, t_I_G, neb_img_I_G = self.get_perturbed_flow_point_and_time_with_neb(batch)
        # print("NEB images used in this batch (train_val_step):", torch.unique(neb_img_I_G))
        atom_I_N_3 = self.model_output_with_neb(x_t_I_N_3, batch, t_I_G, neb_img_I_G)

        # ------------------------------------------- DEBUGGING ------------------------------------------- 
        # Add this debug to train_val_step_with_neb:
        # print(f"x_t_I_N_3 shape: {x_t_I_N_3.shape}")
        # print(f"atom_I_N_3 shape: {atom_I_N_3.shape}")
        # print(f"neb_img_I_G shape: {neb_img_I_G.shape}")
        # print(f"neb_img_I_G unique values: {torch.unique(neb_img_I_G)}")
        # print(f"batch.num_graphs: {batch.num_graphs}")
        # print(f"batch.pos_N_3_I shape: {batch.pos_N_3_I.shape}")
        # Check atoms per graph
        # atoms_per_graph = scatter(torch.ones(batch.pos_N_3_I.shape[0], device=batch.batch.device), 
        #                  batch.batch, 
        #                  dim=0, 
        #                  reduce='sum')
        # print(f"Atoms per graph: {atoms_per_graph}")
        # print(f"Sum: {atoms_per_graph.sum()}")  
        # --------------------------------------------------------------------------------------------------

        rmsd_loss_I = []
        rmsd_error_I = [] 
        for i in range(neb_img_I_G.shape[0]):
            rmsd_loss_value = rmsd_loss(atom_I_N_3[i], dx_dt_I_N_3[i])
            rmsd_error_value = rmsd_error(atom_I_N_3[i], dx_dt_I_N_3[i])
            # print(f"NEB image {neb_img_I_G[i]}: RMSD Loss = {rmsd_loss_value}, RMSD Error = {rmsd_error_value}")
            rmsd_loss_I.append(rmsd_loss_value)
            rmsd_error_I.append(rmsd_error_value)
                
        # Add TS weighting
        # if hasattr(batch, 'ts_idx') and neb_img_I_G.shape[0] > 1:
        #     # Convert ts_indices to tensor if it's a list
        #     if isinstance(batch.ts_idx, list):
        #         ts_indices = torch.tensor(batch.ts_idx, device=self.device, dtype=torch.float32)
        #     else:
        #         ts_indices = batch.ts_idx.float().to(self.device)
            
        #     all_indices = torch.unique(neb_img_I_G).float()
        #     weights = []
        #     for neb_idx in all_indices:
        #         # Exponential decay from TS: average distance to all TS in batch
        #         dist = torch.abs(neb_idx - ts_indices).mean()
        #         weight = torch.exp(-dist / 2.0)
        #         weights.append(weight)
        #     weights = torch.stack(weights)
        #     weights = weights / weights.sum()
            
        #     # Apply weights to losses (ensure alignment with rmsd_loss_I indexing)
        #     weighted_losses = [w * l for w, l in zip(weights, rmsd_loss_I)]
        #     loss = sum(weighted_losses)
        # else:
        loss = torch.stack(rmsd_loss_I).mean()

        spring_loss_value = torch.tensor(0.0, device=self.device)
        if neb_img_I_G.shape[0] > 1 and self.lambda_spring > 0:
            # Expand t from per-graph [I, G, 1] to per-atom [I, N, 1]
            t_I_N = t_I_G[:, batch.batch, :]  # Shape: [I, N, 1]
            pos_pred_I_N_3 = x_t_I_N_3 + (1-t_I_N) * atom_I_N_3
            spring_loss_value = spring_loss(pos_pred_I_N_3, neb_img_I_G, 
                                           batch_indices_N=batch.batch)
            
            # Apply annealing schedule
            spring_weight = self.get_spring_loss_weight()
            spring_loss_value = spring_weight * spring_loss_value
            # print(f"Spring Loss (step {self.global_step}, weight={spring_weight:.3f}): {spring_loss_value}")
        
        # Add spring loss to total loss, lambda is the weighting factor
        if self.lambda_spring > 0:
            loss = loss + self.lambda_spring * spring_loss_value

        metrics = {}
        metrics['train/val_loss'] = loss.detach()
        metrics['train/loss_per_batch_mean'] = torch.stack(rmsd_loss_I).mean().detach()
        if self.lambda_spring > 0:
           metrics['train/spring_loss'] = spring_loss_value.detach()
        
        for i, (l, e) in enumerate(zip(rmsd_loss_I, rmsd_error_I)):
            # Per-molecule: per-graph mean then averaged across graphs
            counts = torch.bincount(batch.batch, minlength=batch.num_graphs).to(e.dtype).unsqueeze(1)  # (G,1)
            sum_per_graph = torch.zeros(batch.num_graphs, device=e.device).index_add(0, batch.batch, e)
            mean_per_graph = sum_per_graph / counts.squeeze(1)  # (G,)
            neb_idx = neb_img_I_G[i, 0].item()
            metrics[f'train/error_image_{neb_idx}_mean_per_graph'] = mean_per_graph.mean().detach()
            # metrics[f'train/loss_image_{neb_idx}'] = l.detach()

        return loss, metrics

    def model_output_with_neb(self, x_t_I_N_3, batch: Data, t_I_G: Tensor, neb_img_I_G: Tensor) -> Tensor:

        num_images = neb_img_I_G.shape[0]  # Number of NEB images
        atom_I_N_3 = []
        
        # Handle single-image case separately for detecting effects of NEB embedding
        if num_images == 1:
            # x_t_I_N_3 has shape [1, N, 3] in the single-image case; representation expects [N, 3]
            if x_t_I_N_3.dim() == 3 and x_t_I_N_3.shape[0] == 1:
                x_t_N_3 = x_t_I_N_3[0]
            else:
                x_t_N_3 = x_t_I_N_3.squeeze(0)

            # t_I_G has shape [1, G, 1] -> use the first image's t_G and flatten to 1D
            if t_I_G.dim() >= 2:
                t_G_flat = t_I_G[0].flatten()
            else:
                t_G_flat = t_I_G.flatten()

            h_N_D, X_N_L_D = self.representation(x_t_N_3, t_G_flat, batch)
            atom_N_3 = self.atomwise_3D_out_layer(h_N_D, X_N_L_D[:, :3, :])
            # Return with an image dimension to match the multi-image return shape: [I=1, N, 3]
            return atom_N_3.unsqueeze(0)
        else:
            for i in range(num_images):
                # Extract data for this NEB image
                x_t_N_3 = x_t_I_N_3[i] 
                t_G = t_I_G[i]  # Shape: [num_graphs, 1]
                neb_idx_G = neb_img_I_G[i]
                
                # Flatten t_G for time embedding (expects 1D tensor)
                t_G_flat = t_G.flatten()
                
                h_N_D, X_N_L_D = self.representation(x_t_N_3, t_G_flat, batch, neb_idx_G)
                atom_N_3 = self.atomwise_3D_out_layer(h_N_D, X_N_L_D[:, :3, :])

                atom_I_N_3.append(atom_N_3)

            # Stack results from all NEB images to match input structure
            return torch.stack(atom_I_N_3, dim=0)

    def training_step(self, batch: Data, batch_idx: int) -> Tensor:
        loss, metrics = self.train_val_step_with_neb(batch)
        self.log_dict(metrics, prog_bar=True, on_step=True, on_epoch=False, batch_size=batch.num_graphs)
        return loss

    def validation_step(self, batch: Data, batch_idx: int) -> Tensor:
        loss, metrics = self.train_val_step_with_neb(batch)
        self.log_dict({k.replace('train/', 'validation/'): v for k, v in metrics.items()},
                      prog_bar=True, on_step=False, on_epoch=True, batch_size=batch.num_graphs)
        return loss

      # S... number of samples
    def test_step(self, batch: Batch, batch_idx: int):
        """Main test step dispatcher."""
        self.seed += 1
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        
        num_atoms, dim, num_neb_images = batch.pos_N_3_I.shape
        neb_imgs_subset = self.neb_img_subset # Determine which NEB images to use

        # Accept string sentinel 'all' or 'auto'
        if isinstance(neb_imgs_subset, str):
            if neb_imgs_subset.lower() in ['all', 'auto']:
                neb_imgs_subset = list(range(0, num_neb_images))
            else:
                raise ValueError(f"Invalid neb_img_subset string: {neb_imgs_subset}")
        
        # Dispatch to appropriate handler
        if neb_imgs_subset is None or len(neb_imgs_subset) == 0:
            self._test_step_ts_only(batch, num_neb_images)
        else:
            self._test_step_neb_images(batch, neb_imgs_subset, num_neb_images)

    def _test_step_ts_only(self, batch: Batch, num_neb_images: int):
        """Test step for TS-only mode (single image per reaction)."""
        # Get TS indices for each reaction in batch
        if hasattr(batch, 'ts_idx'):
            ts_indices = batch.ts_idx
        else:
            # Fallback to middle image
            ts_indices = [num_neb_images // 2] * batch.num_graphs
            print("No ts_idx provided; using middle image for TS images:", ts_indices)

        # Verify we have exactly one TS index per reaction
        assert len(ts_indices) == batch.num_graphs, \
            f"Expected {batch.num_graphs} TS indices, got {len(ts_indices)}"
        
        # Generate one image per reaction (no stacking across images)
        t_T = torch.linspace(0, 1, steps=self.num_steps, device=self.device)

        def ode_func(t, x_t_N_3):
            # x_t_N_3 shape: [N, 3] for single image
            # Return shape should match: [N, 3]
            t_G = torch.full((batch.num_graphs, 1), t, device=self.device)
            # Create neb_img_G for each reaction
            neb_idx_G = torch.tensor(ts_indices, dtype=torch.long, device=self.device)
            model_forces_N_3 = self.model_output_with_neb(x_t_N_3.unsqueeze(0), batch, t_G.unsqueeze(0), neb_idx_G.unsqueeze(0))
            return model_forces_N_3.squeeze(0)  # Remove image dimension

        # Generate #num_samples trajectories for batch
        pos_gen_traj_S_T_N_3 = torch.zeros((self.num_samples, self.num_steps, batch.num_nodes, 3), device=self.device)
        
        for i in range(self.num_samples):
            if self.seed is not None:
                torch.manual_seed(self.seed + i)
            pos_init_N_3 = torch.randn(batch.num_nodes, 3, device=self.device)
            pos_gen_traj_S_T_N_3[i, ...] = odeint(ode_func, pos_init_N_3, t_T, method='euler')

        for j, data in enumerate(batch.to_data_list()):
            # Get single molecule positions from sampled trajectories
            mask = (batch.batch == j).cpu()
            ts_idx = ts_indices[j]
            
            pos_gen_traj_S_T_Nm_3 = pos_gen_traj_S_T_N_3[:, :, mask]
            pos_gt_Nm_3 = data.pos_N_3_I[:, :, ts_idx]
            pos_gen_traj_Sv_T_Nm_3, rmsd_S, _ = self.align_and_rotate_samples(data, pos_gen_traj_S_T_Nm_3, pos_gt_Nm_3, return_transform=True)

            # Aggregate the S samples
            pos_aggr_Nm_3 = torch.median(pos_gen_traj_Sv_T_Nm_3[:, -1, :, :], dim=0).values
            distances_Sv = torch.linalg.vector_norm(pos_gen_traj_Sv_T_Nm_3[:, -1, :, :] - pos_aggr_Nm_3, dim=(1, 2))
            pos_best_T_Nm_3 = pos_gen_traj_Sv_T_Nm_3[torch.argmin(distances_Sv)]

            # Store single NEB image result
            data.pos_gen_neb = [pos_best_T_Nm_3]
            data.pos_gen_neb_I_T_Nm_3 = torch.stack(data.pos_gen_neb, dim=0)  # [1, T, N, 3]
            data.neb_indices_used = [ts_idx]  # Store which NEB images were evaluated
            
            data.pos_gen_ = pos_best_T_Nm_3[-1, :, :]  # Last time step
            data.pos_gen = [data.pos_gen_]  # For compatibility

            if self.save_all_samples:
                data.pos_gen_all_samples_S_N_3 = pos_gen_traj_Sv_T_Nm_3[:, -1, :, :]
            self.results_R.append(data.to("cpu"))

    def _test_step_neb_images(self, batch: Batch, neb_imgs_subset: list, num_neb_images: int):
        """Test step for multiple NEB images."""
        t_T = torch.linspace(0, 1, steps=self.num_steps, device=self.device)

        def ode_func(t, x_t_N_3):
            t_I_G = torch.full((len(neb_imgs_subset), batch.num_graphs, 1), t, device=self.device)
            neb_imgs_I_G = torch.tensor(neb_imgs_subset, device=self.device).unsqueeze(1).expand(-1, batch.num_graphs)
            model_forces_I_N_3 = self.model_output_with_neb(x_t_N_3, batch, t_I_G, neb_imgs_I_G)
            return model_forces_I_N_3

        # Generate #num_samples trajectories for batch
        pos_gen_traj_S_T_I_N_3 = torch.zeros((self.num_samples, self.num_steps, len(neb_imgs_subset), batch.num_nodes, 3), device=self.device)
        
        for i in range(self.num_samples):
            if self.seed is not None:
                torch.manual_seed(self.seed + i)

            if self.preserve_path_coherence:
                # Match training: share aligned x_0 across all NEB images
                pos_init_ref_N_3 = torch.randn(batch.num_nodes, 3, device=self.device)
                pos_init_I_N_3 = torch.zeros(len(neb_imgs_subset), batch.num_nodes, 3, device=self.device)
                pos_init_I_N_3[0] = pos_init_ref_N_3
                
                for img_idx in range(1, len(neb_imgs_subset)):
                    pos_init_new_N_3 = torch.randn(batch.num_nodes, 3, device=self.device)
                    pos_init_I_N_3[img_idx] = align_x0_to_reference_batched(
                        pos_init_new_N_3, pos_init_ref_N_3, batch.batch, batch.num_graphs
                    )
            else:
                pos_init_I_N_3 = torch.randn(len(neb_imgs_subset), batch.num_nodes, 3, device=self.device)
            
            pos_gen_traj_S_T_I_N_3[i, ...] = odeint(ode_func, pos_init_I_N_3, t_T, method='euler')

        for j, data in enumerate(batch.to_data_list()):
            self._process_neb_molecule(data, j, batch, pos_gen_traj_S_T_I_N_3, neb_imgs_subset, num_neb_images)

    def _process_neb_molecule(self, data: Data, molecule_idx: int, batch: Batch, 
                               pos_gen_traj_S_T_I_N_3: Tensor, neb_imgs_subset: list, num_neb_images: int):
        """Process a single molecule through all NEB images."""
        mask = (batch.batch == molecule_idx).cpu()
        data.pos_gen_neb = []
        
        # For path coherence: store first image's transformation
        saved_transform = None
        best_sample_idx = None

        for neb_idx in range(len(neb_imgs_subset)):
            pos_gen_traj_S_T_Nm_3 = pos_gen_traj_S_T_I_N_3[:, :, neb_idx, mask]
            actual_neb_idx = neb_imgs_subset[neb_idx]

            pos_gt_Nm_3 = data.pos_N_3_I[:, :, actual_neb_idx]
            
            if self.preserve_path_coherence and saved_transform is not None:
                # Apply the same transformation from first image to subsequent images
                pos_gen_traj_Sv_T_Nm_3, rmsd_S = self.apply_saved_transform(
                    pos_gen_traj_S_T_Nm_3, pos_gt_Nm_3, saved_transform
                )
            else:
                # Standard alignment (or first image when preserving coherence)
                pos_gen_traj_Sv_T_Nm_3, rmsd_S, transform = self.align_and_rotate_samples(
                    data, pos_gen_traj_S_T_Nm_3, pos_gt_Nm_3, return_transform=True
                )
                if self.preserve_path_coherence and saved_transform is None:
                    saved_transform = transform

            # Aggregate the S samples (or use first sample for path coherence)
            if self.preserve_path_coherence:
                if best_sample_idx is None:
                    # For first image, select best sample and remember the index
                    pos_aggr_Nm_3 = torch.median(pos_gen_traj_Sv_T_Nm_3[:, -1, :, :], dim=0).values
                    distances_Sv = torch.linalg.vector_norm(pos_gen_traj_Sv_T_Nm_3[:, -1, :, :] - pos_aggr_Nm_3, dim=(1, 2))
                    best_sample_idx = torch.argmin(distances_Sv).item()
                # Use the same sample index for all images
                pos_best_T_Nm_3 = pos_gen_traj_Sv_T_Nm_3[best_sample_idx]
            else:
                pos_aggr_Nm_3 = torch.median(pos_gen_traj_Sv_T_Nm_3[:, -1, :, :], dim=0).values
                distances_Sv = torch.linalg.vector_norm(pos_gen_traj_Sv_T_Nm_3[:, -1, :, :] - pos_aggr_Nm_3, dim=(1, 2))
                pos_best_T_Nm_3 = pos_gen_traj_Sv_T_Nm_3[torch.argmin(distances_Sv)]

            setattr(data, f'pos_gen_neb_image_{neb_idx}', pos_best_T_Nm_3)
            setattr(data, f'rmsd_neb_image_{neb_idx}', rmsd_S)
            data.pos_gen_neb.append(pos_best_T_Nm_3)

        # Convert to tensor: [Images, Time, Nm_atoms, 3]
        data.pos_gen_neb_I_T_Nm_3 = torch.stack(data.pos_gen_neb, dim=0)
        data.neb_indices_used = neb_imgs_subset  # Store which NEB images were evaluated

        # Determine TS index
        if hasattr(batch, 'ts_idx'):
            if isinstance(batch.ts_idx, (list, tuple)):
                ts_idx = batch.ts_idx[molecule_idx]
            else:
                ts_idx = batch.ts_idx[molecule_idx] if len(batch.ts_idx.shape) > 0 else batch.ts_idx.item()
        else:
            ts_idx = num_neb_images // 2  # Fallback to middle

        # Find the position of ts_idx in neb_imgs_subset
        if ts_idx in neb_imgs_subset:
            subset_idx = neb_imgs_subset.index(ts_idx)
        elif num_neb_images == 3:
            subset_idx = 1
        else:
            # Fallback: use first image in subset if TS not in subset
            subset_idx = 0

        data.pos_gen_ = data.pos_gen_neb_I_T_Nm_3[subset_idx, -1, :, :]
        data.pos_gen = [data.pos_gen_neb_I_T_Nm_3[subset_idx, -1, :, :]]

        if self.save_all_samples:
            data.pos_gen_all_samples_S_N_3 = pos_gen_traj_Sv_T_Nm_3[:, -1, :, :]
        
        # Compute extended NEB path metrics if enabled
        if self.rmse_extended_metrics:
            # Prepare ground truth positions for all NEB images
            pos_gt_I_Nm_3 = torch.stack([data.pos_N_3_I[:, :, neb_imgs_subset[i]] for i in range(len(neb_imgs_subset))], dim=0)
            
            metrics = compute_neb_path_metrics(data.pos_gen_neb_I_T_Nm_3, pos_gt_I_Nm_3, neb_imgs_subset)
            
            # Attach metrics to data object
            data.rmsd_btw_images = metrics['rmsd_btw_images']
            data.rmsd_mean_across_images = metrics['rmsd_mean_across_images']
            data.rmsd_std_across_images = metrics['rmsd_std_across_images']
            data.rmsd_cv_across_images = metrics['rmsd_cv_across_images']
            data.rmsd_max_deviation = metrics['rmsd_max_deviation']
            data.gt_rmsd_btw_images = metrics['gt_rmsd_btw_images']
            data.gt_rmsd_mean_across_images = metrics['gt_rmsd_mean_across_images']
            data.spacing_ratio = metrics['spacing_ratio']
        
        self.results_R.append(data.to("cpu"))

    def align_and_rotate_samples(self, data, pos_gen_S_T_Nm_3: torch.Tensor, ref_pos_Nm_3: torch.Tensor, return_transform: bool = False):
        """Align and rotate trajectory samples using the last time step
        
        Args:
            data: Input data object
            pos_gen_S_T_Nm_3: Generated trajectory [Samples, Time, Nm_atoms, 3]
            ref_pos_Nm_3: Reference structure [Nm_atoms, 3] to align to
            return_transform: If True, return the transformation for reuse
        """
        # Use only the last time step for alignment
        pos_gen_S_Nm_3 = pos_gen_S_T_Nm_3[:, -1, :, :]  # [Samples, Nm_atoms, 3]
    
        # ===== ADD SUBSTRUCTURE MATCHING =====
        matches = get_substruct_matches(data.smiles)
        match_S_Nm = get_min_dmae_match_torch_batch(matches, ref_pos_Nm_3, pos_gen_S_Nm_3)
        pos_gen_S_Nm_3 = torch.gather(pos_gen_S_Nm_3, 1, match_S_Nm.unsqueeze(-1).expand(-1, -1, 3))
        # =====================================

        aligned_S_Nm_3, rmsd_S = kabsch_align_batched(ref_pos_Nm_3, pos_gen_S_Nm_3)
        
        # Copy aligned positions back to the trajectory
        pos_gen_S_T_Nm_3[:, -1, :, :] = aligned_S_Nm_3
        
        if return_transform:
            # Store transformation for path coherence: atom permutation indices
            transform = {'match_S_Nm': match_S_Nm}
            return pos_gen_S_T_Nm_3, rmsd_S.cpu().numpy(), transform
        
        return pos_gen_S_T_Nm_3, rmsd_S.cpu().numpy()
    
    def apply_saved_transform(self, pos_gen_S_T_Nm_3: torch.Tensor, ref_pos_Nm_3: torch.Tensor, transform: dict):
        """Apply a saved transformation from the first image to subsequent images.
        
        This ensures all NEB images use the same atom permutation for path coherence.
        
        Args:
            pos_gen_S_T_Nm_3: Generated trajectory [Samples, Time, Nm_atoms, 3]
            ref_pos_Nm_3: Reference structure [Nm_atoms, 3] for Kabsch alignment
            transform: Saved transformation dict with 'match_S_Nm' atom permutation
        """
        # Use only the last time step for alignment
        pos_gen_S_Nm_3 = pos_gen_S_T_Nm_3[:, -1, :, :]  # [Samples, Nm_atoms, 3]
        
        # Apply saved atom permutation (same as first image)
        match_S_Nm = transform['match_S_Nm']
        pos_gen_S_Nm_3 = torch.gather(pos_gen_S_Nm_3, 1, match_S_Nm.unsqueeze(-1).expand(-1, -1, 3))
        
        # Kabsch alignment to this image's ground truth (rotation only)
        aligned_S_Nm_3, rmsd_S = kabsch_align_batched(ref_pos_Nm_3, pos_gen_S_Nm_3)
        
        # Copy aligned positions back to the trajectory
        pos_gen_S_T_Nm_3[:, -1, :, :] = aligned_S_Nm_3
        
        return pos_gen_S_T_Nm_3, rmsd_S.cpu().numpy()

    def configure_optimizers(self) -> Tuple[List[torch.optim.Optimizer], List[Dict]]:
        """Configure optimizers and learning rate schedulers."""
        print("self.weight_decay", self.weight_decay)
        if self.schedule_free:
            import schedulefree
            optimizer = schedulefree.AdamWScheduleFreeClosure(
                self.trainer.model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
                eps=1e-8,
                warmup_steps=self.lr_warmup_steps
            )
            return [optimizer], []

        optimizer = torch.optim.AdamW(
            self.trainer.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            eps=1e-7,
        )

        if self.scheduler and callable(self.scheduler):
            scheduler, _ = self.scheduler(optimizer=optimizer)
        else:
            scheduler = ReduceLROnPlateau(
                optimizer,
                factor=self.lr_decay,
                patience=self.lr_patience,
                min_lr=self.lr_minlr,
            )

        schedule = {
            "scheduler": scheduler,
            "monitor": self.lr_monitor,
            "interval": "epoch",
            "frequency": 1,
            "strict": True,
        }

        return [optimizer], [schedule]