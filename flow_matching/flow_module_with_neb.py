from typing import Dict, Optional, Callable, TypeVar, List, Union, Tuple
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Data, Batch
import lightning.pytorch as pl
from torchdiffeq import odeint

from flow_matching.utils import get_shortest_path_fast_batched_x_1, rmsd_loss, get_substruct_matches, get_min_dmae_match_torch_batch
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
    
    ### New methods for NEB image conditioning
    
    def get_perturbed_flow_point_and_time_with_neb(self, batch: Data):

        num_atoms, dim, neb_img = batch.pos_N_3_I.shape
        # Collect all flow data for all NEB images
        all_x_t = []
        all_dx_dt = []
        all_t_G = []
        all_neb_imgs = []

        for neb_idx in range(0, neb_img):
            x_1_N_3 = batch.pos_N_3_I[:, :, neb_idx]
            x_0_N_3 = torch.randn_like(x_1_N_3, device=self.device)

            t_G = torch.rand(batch.num_graphs, 1, device=self.device)
            t_N = t_G[batch.batch]

            x_1_aligned_N_3 = get_shortest_path_fast_batched_x_1(x_0_N_3, x_1_N_3, batch)
            x_t_N_3 = (1 - t_N) * x_0_N_3 + t_N * x_1_aligned_N_3
            dx_dt_N_3 = x_1_aligned_N_3 - x_0_N_3

            all_x_t.append(x_t_N_3)
            all_dx_dt.append(dx_dt_N_3)
            all_t_G.append(t_G)
            all_neb_imgs.append(torch.full((batch.num_graphs,), neb_idx, dtype=torch.long, device=self.device))

        # Concatenate all results
        x_t_all = torch.cat(all_x_t, dim=0)
        dx_dt_all = torch.cat(all_dx_dt, dim=0)
        t_G_all = torch.cat(all_t_G, dim=0)
        neb_imgs_all = torch.cat(all_neb_imgs, dim=0)

        return x_t_all, dx_dt_all, t_G_all, neb_imgs_all

    def train_val_step_with_neb(self, batch: Data) -> Tensor:
        x_t_N_3, dx_dt_N_3, t_G, neb_img_idx = self.get_perturbed_flow_point_and_time_with_neb(batch)

        atom_N_3 = self.model_output_with_neb(x_t_N_3, batch, t_G, neb_img_idx)

        return rmsd_loss(atom_N_3, dx_dt_N_3)

    def model_output_with_neb(self, x_t_N_3, batch: Data, t_G: Tensor, neb_img_idx: Tensor) -> Tensor:
        # Process each NEB image separately to avoid batch concatenation issues
        N = batch.num_nodes
        neb_img = x_t_N_3.shape[0] // N  # number of NEB images
        
        all_atom_outputs = []
        
        for i in range(neb_img):
            # Extract atomic positions for this NEB image
            start_idx = i * N
            end_idx = (i + 1) * N
            x_t_single = x_t_N_3[start_idx:end_idx]
            
            # Extract corresponding time and NEB index info  
            t_G_single = t_G[i * batch.num_graphs:(i + 1) * batch.num_graphs]
            neb_idx_single = neb_img_idx[i * batch.num_graphs:(i + 1) * batch.num_graphs]
            
            # Process this NEB image with original batch structure
            # Pass raw time values and NEB indices, let the representation handle embedding
            h_N_D, X_N_L_D = self.representation(x_t_single, t_G_single, batch, neb_idx_single)
            atom_N_3 = self.atomwise_3D_out_layer(h_N_D, X_N_L_D[:, :3, :])
            
            all_atom_outputs.append(atom_N_3)
        
        # Concatenate results from all NEB images
        return torch.cat(all_atom_outputs, dim=0)

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
        
        t_T = torch.linspace(0, 1, steps=self.num_steps, device=self.device)

        def ode_func(t, x_t_N_3):
            t_G = torch.tensor([t] * batch.num_graphs, device=self.device)
            model_forces_N_3 = self.model_output_with_neb(x_t_N_3, batch, t_G)
            return model_forces_N_3

        # Generate #num_samples trajectories for batch
        pos_gen_traj_S_T_N_3 = torch.zeros((self.num_samples, self.num_steps, batch.num_nodes, 3), device=self.device)
        for i in range(self.num_samples):
            if self.seed is not None:
                torch.manual_seed(self.seed + i)
            pos_init_N_3 = torch.randn_like(batch.pos[:, 1, :], device=self.device)
            pos_gen_traj_S_T_N_3[i, ...] = odeint(ode_func, pos_init_N_3, t_T, method='euler')

        for j, data in enumerate(batch.to_data_list()):
            # Get single molecule positions from sampled trajectories
            mask = (batch.batch == j).cpu()
            pos_gen_traj_S_T_Nm_3 = pos_gen_traj_S_T_N_3[:, :, mask]
            
            pos_gen_traj_S_T_Nm_3 = self.align_and_rotate_samples(data, pos_gen_traj_S_T_Nm_3)

            # -------------------------- START: Aggregate the S samples --------------------------
            pos_aggr_Nm_3 = torch.median(pos_gen_traj_S_T_Nm_3[:, -1, :, :], dim=0).values
            distances_Sv = torch.linalg.vector_norm(pos_gen_traj_S_T_Nm_3[:, -1, :, :] - pos_aggr_Nm_3, dim=(1, 2))
            pos_best_T_Nm_3 = pos_gen_traj_S_T_Nm_3[torch.argmin(distances_Sv)]
            # -------------------------- END: Aggregate the S samples --------------------------

            data.pos_gen = pos_best_T_Nm_3
            self.results_R.append(data.to("cpu"))

    def align_and_rotate_samples(self, data, pos_gen_traj_S_T_Nm_3):
        """
        For each molecule we sample the TS S times. 
        Do substructure matching, and align (w. Kabsch) to the GT.
        """
        pos_gt_Nm_3 = data.pos[:, 1, :]
        
        # Substructure matching (batched for S)
        matches = get_substruct_matches(data.smiles)
        match_Sv_Nm = get_min_dmae_match_torch_batch(matches, pos_gt_Nm_3, pos_gen_traj_S_T_Nm_3[:, -1, :, :])
        pos_gen_traj_S_T_Nm_3[:,-1,:,:] = torch.gather(pos_gen_traj_S_T_Nm_3[:,-1,:,:], 1, match_Sv_Nm.unsqueeze(-1).expand(-1,-1,3))
        
        # Kabsch rotation
        S = pos_gen_traj_S_T_Nm_3.shape[0]
        Nm = pos_gen_traj_S_T_Nm_3.shape[2]
        # This is a trick to make the batched rotation to the GT molecule easy
        data.batch = torch.arange(S, device=self.device).repeat_interleave(Nm)
        # Repeat GT pos S times (have to rotate each sample to it)
        pos_gt_SNm_3 = pos_gt_Nm_3.repeat(S, 1)
        # Reshape [S, Nm] nodes to single S*Nm dimension. Makes batched rotation possible.
        pos_gen_SNm_3 = pos_gen_traj_S_T_Nm_3[:,-1,:,:].reshape(S*Nm, 3)
        pos_gen_aligned_SNm_3 = get_shortest_path_fast_batched_x_1(pos_gt_SNm_3, pos_gen_SNm_3, data)
        pos_gen_aligned_S_Nm_3 = pos_gen_aligned_SNm_3.reshape(S, Nm, 3)
        pos_gen_traj_S_T_Nm_3[:,-1,:,:] = pos_gen_aligned_S_Nm_3
        
        return pos_gen_traj_S_T_Nm_3


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
