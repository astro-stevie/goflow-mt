from typing import Dict, Optional, Callable, TypeVar, List, Union, Tuple
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Data, Batch
import lightning.pytorch as pl
from torchdiffeq import odeint

from flow_matching.utils import get_shortest_path_fast_batched_x_1, rmsd_loss, get_substruct_matches, get_min_dmae_match_torch_batch, kabsch_align_batched
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

        self.lambda_cf = 0.0
        self.seed = seed
        print("FM has seed", self.seed, "-----------------------------")

        self.scheduler = scheduler

        self.save_hyperparameters(ignore=['representation'])

        # save results in test_step
        self.results_R = []
        self.neb_img_subset = neb_img_subset
    
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
    
            
        # Case 1: Empty subset means use only TS (one image per reaction)
        if neb_img_subset is None or len(neb_img_subset) == 0:
            # Get TS indices for each reaction in batch
            if hasattr(batch, 'ts_idx'):
                # ts_idx can be a tensor or list with one value per reaction
                if isinstance(batch.ts_idx, (list, tuple)):
                    ts_indices = batch.ts_idx
                elif torch.is_tensor(batch.ts_idx):
                    if batch.ts_idx.numel() == 1:
                        # Single value for all reactions
                        ts_indices = [batch.ts_idx.item()] * batch.num_graphs
                    else:
                        # One value per reaction
                        ts_indices = batch.ts_idx.tolist()
                        print("ts_indices:", ts_indices)
                else:
                    ts_indices = [batch.ts_idx] * batch.num_graphs
            else:
                # Fallback to middle image
                ts_indices = [neb_img // 2] * batch.num_graphs

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
            for neb_idx in neb_img_subset:
                x_1_N_3 = batch.pos_N_3_I[:, :, neb_idx]
                x_0_N_3 = torch.randn_like(x_1_N_3, device=self.device)

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

        atom_I_N_3 = self.model_output_with_neb(x_t_I_N_3, batch, t_I_G, neb_img_I_G)

        # Optional: Weight loss by proximity to transition state if ts_idx is available
        if hasattr(batch, 'ts_idx'):
            # TODO: Implement TS-weighted loss when rmsd_loss supports per-sample losses
            pass
        
        return rmsd_loss(atom_I_N_3, dx_dt_I_N_3)

    def model_output_with_neb(self, x_t_I_N_3, batch: Data, t_I_G: Tensor, neb_img_I_G: Tensor) -> Tensor:

        num_images = neb_img_I_G.shape[0]  # Number of NEB images
        atom_I_N_3 = []
        
        for i in range(num_images):
            # Extract data for this NEB image
            x_t_N_3 = x_t_I_N_3[i] 
            t_G = t_I_G[i]  # Shape: [num_graphs, 1]
            neb_idx_G = neb_img_I_G[i]  # Shape: [num_graphs]
            
            # Flatten t_G for time embedding (expects 1D tensor)
            t_G_flat = t_G.flatten()  # Shape: [num_graphs]
            
            h_N_D, X_N_L_D = self.representation(x_t_N_3, t_G_flat, batch, neb_idx_G)
            atom_N_3 = self.atomwise_3D_out_layer(h_N_D, X_N_L_D[:, :3, :])

            atom_I_N_3.append(atom_N_3)

        # Stack results from all NEB images to match input structure
        return torch.stack(atom_I_N_3, dim=0)

    def training_step(self, batch: Data, batch_idx: int) -> Tensor:
        loss = self.train_val_step_with_neb(batch)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=False, batch_size=batch.num_graphs)
        return loss

    def validation_step(self, batch: Data, batch_idx: int) -> Tensor:
        loss = self.train_val_step_with_neb(batch)
        self.log("validation/val_loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch.num_graphs)
        return loss

    def test_step(self, batch: Batch, batch_idx: int):
        self.seed += 1
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        
        # Get NEB dimensions from data
        num_atoms, dim, num_neb_images = batch.pos_N_3_I.shape

        t_T = torch.linspace(0, 1, steps=self.num_steps, device=self.device)

        def ode_func(t, x_t_N_3):
            t_I_G = torch.full((num_neb_images, batch.num_graphs, 1), t, device=self.device)
            neb_imgs_I_G = torch.arange(num_neb_images, device=self.device).unsqueeze(1).expand(-1, batch.num_graphs)
            model_forces_I_N_3 = self.model_output_with_neb(x_t_N_3, batch, t_I_G, neb_imgs_I_G)
            return model_forces_I_N_3

        # Generate #num_samples trajectories for batch
        pos_gen_traj_S_T_I_N_3 = torch.zeros((self.num_samples, self.num_steps, num_neb_images, batch.num_nodes, 3), device=self.device)
        for i in range(self.num_samples):
            if self.seed is not None:
                torch.manual_seed(self.seed + i)
            pos_init_I_N_3 = torch.randn((num_neb_images, batch.num_nodes, 3), device=self.device)
            pos_gen_traj_S_T_I_N_3[i, ...] = odeint(ode_func, pos_init_I_N_3, t_T, method='euler')

        for j, data in enumerate(batch.to_data_list()):
            # Get single molecule positions from sampled trajectories
            mask = (batch.batch == j).cpu()
            
            # Store results for each NEB image
            data.pos_gen_neb = []
            
            for neb_idx in range(num_neb_images):
                # Extract trajectory for this NEB image: [Samples, Time, Nm_atoms, 3]
                pos_gen_traj_S_T_Nm_3 = pos_gen_traj_S_T_I_N_3[:, :, neb_idx, mask, :]
                
                # Use ground truth from corresponding NEB image for alignment
                pos_gt_Nm_3 = data.pos_N_3_I[:, :, neb_idx].T  # Convert to [Nm_atoms, 3]
                
                # Align and rotate samples for this NEB image
                pos_gen_traj_S_T_Nm_3, _ = self.align_and_rotate_samples(data, pos_gen_traj_S_T_Nm_3)

                # -------------------------- START: Aggregate the S samples --------------------------
                pos_aggr_Nm_3 = torch.median(pos_gen_traj_S_T_Nm_3[:, -1, :, :], dim=0).values
                distances_Sv = torch.linalg.vector_norm(pos_gen_traj_S_T_Nm_3[:, -1, :, :] - pos_aggr_Nm_3, dim=(1, 2))
                pos_best_T_Nm_3 = pos_gen_traj_S_T_Nm_3[torch.argmin(distances_Sv)]
                # -------------------------- END: Aggregate the S samples --------------------------

                data.pos_gen_neb.append(pos_best_T_Nm_3)
            
            # Convert to tensor: [Images, Time, Nm_atoms, 3]
            data.pos_gen_neb_I_T_Nm_3 = torch.stack(data.pos_gen_neb, dim=0)
            
            if hasattr(batch, 'ts_idx'):
                if isinstance(batch.ts_idx, (list, tuple)):
                    ts_idx = batch.ts_idx[j]
                else:
                    ts_idx = batch.ts_idx[j] if len(batch.ts_idx.shape) > 0 else batch.ts_idx.item()
            else:
                ts_idx = num_neb_images // 2  # Fallback to middle
                
            data.pos_gen_ = data.pos_gen_neb_I_T_Nm_3[ts_idx, -1, :, :]  # TS image, last time step
            
            # Also set the format expected by the callback for compatibility
            data.pos_gen = [data.pos_gen_neb_I_T_Nm_3[ts_idx, -1, :, :]]  # Wrap in list so [-1] works
            
            # Store the actual TS index used for analysis
            data.ts_idx_used = ts_idx
            
            self.results_R.append(data.to("cpu"))

    def align_and_rotate_samples(self, data, pos_gen_S_T_Nm_3: torch.Tensor):
        """Align and rotate trajectory samples using the last time step"""
        # Use only the last time step for alignment
        pos_gen_S_Nm_3 = pos_gen_S_T_Nm_3[:, -1, :, :]  # [Samples, Nm_atoms, 3]
        
        aligned_S_Nm_3, rmsd_S = kabsch_align_batched(data.pos, pos_gen_S_Nm_3)
        
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
