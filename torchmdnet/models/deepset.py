from typing import Callable, Dict, Optional, Union, List
import typing
import wandb
import rich
from rich import pretty

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as tg
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, zeros_
from torch.nn import init
# from torch_geometric.nn import GCNConv, SAGEConv, GATConv

from torchmdnet.models.utils import (
    NeighborEmbedding,
    CosineCutoff,
    OptimizedDistance,
    rbf_class_mapping,
    act_class_mapping,
    scatter,
)

# from src import utils
# from src.properties import properties

__all__ = ["DeepSet"]


class DeepSet(nn.Module):

    def __init__(
            self,
            base_cutoff: float,
            outer_cutoff: float,
            num_gates: int = 10,
            k: int = 2,
            max_num_neighbors: int = 400,
            embedding_size: int = 128,
            num_rbf=32,
            expert_out_features: int = 128,
            rbf_type: str = "gauss",
            trainable_rbf: bool = False,
            dtype: torch.dtype = torch.float32,
            skip_duplicates: bool = False,
            # use_sageconv: bool = False, # ! false for now
            # gnn_hidden_dim: int = 128
            # gnn_layers: int = 2
            # in_channels, hidden_channels, out_channels, aggr="mean" #Added for GNN
    ):

        super(DeepSet, self).__init__()

        self.outer_cutoff = outer_cutoff
        self.base_cutoff = base_cutoff
        self.embedding_size = embedding_size
        self.dtype = dtype
        self.skip_duplicates = skip_duplicates

        # self.sage1 = SAGEConv(in_channels, hidden_channels, aggr=aggr)
        # self.sage2 = SAGEConv(hidden_channels, hidden_channels, aggr==aggr)
        # self.lin = Linear(hidden_channels, out_channels)

        # self.reset_parameters

        self.distance = OptimizedDistance(
            base_cutoff,
            outer_cutoff,
            max_num_pairs=max_num_neighbors,
            return_vecs=True,
            loop=True,
            box=None,
            long_edge_index=True,
            check_errors=True,
            # Set False if there are more than 10k neighbors and it throw an error. Check this thread: https://github.com/torchmd/torchmd-net/issues/203
        )

        self.distance_expansion = rbf_class_mapping[rbf_type](
            base_cutoff, outer_cutoff, num_rbf, trainable_rbf
        )
        # Creates connected linear layer
        self.distance_proj = nn.Linear(num_rbf, embedding_size,
                                       dtype=dtype)  # transforms size num_rbf into embedding_size
        self.cutoff = CosineCutoff(base_cutoff, outer_cutoff)  # Restrict interactions beyond a certain distance
        self.embedding = nn.Embedding(100, embedding_size, dtype=dtype)  # Embedding layer maps ints

        self.neighbor_embedding = NeighborEmbedding(embedding_size, 20, base_cutoff, outer_cutoff, 100, dtype)
        # Part of a message passing NN/GNN
        # ! Need to update size for this as passing in new size later on
        expanded_feature_dim = embedding_size + 3
        self.d_ij_transform = nn.Linear(expanded_feature_dim, embedding_size,
                                        dtype=dtype)  # Refine how interatomic distances contribute
        self.a_i_transform = nn.Linear(embedding_size, embedding_size,
                                       dtype=dtype)  # Project atom features into a common space before interactions
        self.a_j_transform = nn.Linear(embedding_size, embedding_size,
                                       dtype=dtype)  # When node i receives messages from j (neighbors) those messages are transformed properly.

        # Gated message passing system
        self.gamma_transform = nn.Linear(3 * embedding_size, embedding_size,
                                         dtype=dtype)  # 3 suggests processing diff inputs i.e. central, neighbour and edge
        # Shape: (num_features, num_gates)
        self.W_g = nn.Parameter(torch.randn(embedding_size,
                                            num_gates) * 0.01)  # Small random initialization // gating mechanism controls info flow and selects important messages

        # Noise weight matrix W_noise
        # Shape: (num_features, num_gates)
        self.W_noise = nn.Parameter(torch.randn(embedding_size, num_gates) * 0.01)  # Small random initialization

        # Distance Learnable Parameters (Steven's method)
        self.W_distance = nn.Parameter(torch.randn(embedding_size, num_gates) * 0.01)

        self.t_parameters = nn.Parameter(torch.randn(num_gates) * 0.01)

        # Hyperparameter k for top-k selection
        # self.k = k
        self.num_gates = num_gates

        self.experts = nn.ModuleList([  # List of independent layers with each acting as an expert
            nn.Linear(embedding_size, expert_out_features, dtype=dtype)
            for _ in range(num_gates)  # One expert per gate (to handle all possible gates)
        ])

    # Gating mechanism chooses which experts contribute to output.

    def reset_parameters(self):
        # self.sage1.reset_parameters()
        # self.sage2.reset_parameters()
        # self.lin.reset_parameters()
        pass

    # Numpy style docstring
    def forward(self,
                z: torch.Tensor,  # Type hints, here for readability and clarity
                pos: torch.Tensor,
                batch: torch.Tensor,
                box: Optional[torch.Tensor] = None,
                q: Optional[torch.Tensor] = None,
                s: Optional[torch.Tensor] = None) -> typing.Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        #Google style doc string
        Args:
            z:                                  # Size is like (n_atoms, 1)
            pos:                                # Size is like (n_atoms, 3)
            batch:  						    # Size is like (n_atoms, 1)
            box:            		            # Size is like (3, 3)
            q:                                  # Size is like (n_atoms, 1)
            s:      					        # Size is like (n_atoms, 1)

        Returns:

        """
        # Beginning the forward pass
        x = self.embedding(z)  # Maps atomic numbers to learnable vectors, continuous representation

        edge_index, edge_weight, edge_vec = self.distance(pos, batch,
                                                          box)  # Finds pairs of atoms close and returns which atoms are connected, their distance and direction
        # Edge weight distance, take as x

        edge_attr = self.distance_expansion(
            edge_weight)  # Converts raw distances between atoms into a better format, NN struggle with raw values

        # edge_index = edge_index[:, edge_weight != 0]
        # edge_weight = edge_weight[edge_weight != 0] # * Prevent self loops (No distance between same an atom so it should be 0)
        # edge_vec = edge_vec[edge_vec != 0]
        mask = edge_index[0] != edge_index[
            1]  # Mask used to handle anomalous data points , filters out self loops (source and target)
        if not mask.all():  # Checks for any false values in the mask, if so then following code is executed
            edge_index = edge_index[:, mask]
            edge_weight = edge_weight[mask]
            edge_attr = edge_attr[mask]
            edge_vec = edge_vec[mask]  # Mask edge_vec as well

        if self.skip_duplicates:  # this remove repeated edges in calculation (it means upper triangle matrix)
            edge_index = edge_index[:, ::2]  # Slicing, selects every second edge so skips over dupes
            edge_weight = edge_weight[::2]  # Slice with 2 to skip duplicate edges and keep only one direction
            edge_attr = edge_attr[::2]
            edge_vec = edge_vec[::2]

        # ! Begin project here, add square, cube, tet, sqr root etc
        # edge_weight_sq = edge_weight ** 2
        edge_weight_cube = edge_weight ** 3
        # edge_weight_tet = edge_weight ** 4
        edge_weight_sqrt = torch.sqrt(edge_weight)
        # edge_weight_log = torch.log(edge_weight)

        edge_weight: torch.Tensor  # Type hint, edge_weight is expected to be a tensor

        # Normalize edge_vec for masked edges (similar to TorchMD_ET)
        edge_vec = edge_vec / torch.norm(edge_vec, dim=1, keepdim=True)

        # Compute the cutoff and distance projection
        C = self.cutoff(
            edge_weight)  # Applies cutoff function to edge weight tensor, limits interactions to certain threshold.
        d_ij_projection = self.distance_proj(edge_attr) * C.view(-1,
                                                                 1)  # Applies cutoff values to projections. If edge is zero then it is reflected here.

        # !Concat new weights to the projection, since dim = 1 need to ensure concat along a column ??
        edge_features = torch.cat(
            [
                d_ij_projection,
                # edge_weight_sq.view(-1,1),
                edge_weight_cube.view(-1, 1),
                # edge_weight_tet.view(-1,1),
                edge_weight_sqrt.view(-1, 1),
                # edge_weight_log.view(-1,1),
                edge_weight.view(-1, 1)
            ], dim=1
        )

        # Transform the distance projection, nuclear charges and atom embeddings
        d_ij_t_projection = self.d_ij_transform(edge_features)  # Fix size of this afterwards
        a_i_projection = self.a_i_transform(x[edge_index[0, :]])  # Prepares for calculations
        a_j_projection = self.a_j_transform(x[edge_index[1, :]])  # Enables message passing

        # ! Need to append GNN SAGEConv to gamma_projection

        gamma_projection = self.gamma_transform(
            torch.concat([a_i_projection.squeeze(), a_j_projection.squeeze(), d_ij_t_projection], dim=1))
        # Combine the source and target atom features then apply transformation to better learn them

        # Computing Z:
        d_ij_expanded = torch.repeat_interleave(edge_weight, self.num_gates, dim=0)

        d_ij_expanded = 1 / torch.clamp(
            torch.abs(d_ij_expanded.view(
                -1,
                self.t_parameters.size(0)
            ) - self.t_parameters),
            min=1e-8  # Avoid division by zero
        )
        softmax_d_ij_expanded = F.softmax(d_ij_expanded, dim=1)

        experts_output = [self.experts[i](gamma_projection) for i in range(self.num_gates)]

        experts_contributions = [e_o * e_w for e_o, e_w in zip(experts_output, softmax_d_ij_expanded.split(1, dim=1))]
        # do the sum of the experts_contributions
        edge_level_output = torch.sum(torch.stack(experts_contributions, dim=0), dim=0)

        # Doing aggregation over the atoms
        atom_level_output_x = scatter(edge_level_output, edge_index[0], dim=0, reduce="sum")

        # Doing the equivariant operation
        n_atoms = z.size(0)  # Number of atoms from z
        atom_level_output = torch.zeros(n_atoms, self.experts[0].out_features, dtype=self.dtype, device=z.device)
        vec = torch.zeros(n_atoms, 3, self.experts[0].out_features, dtype=self.dtype,
                          device=z.device)  # Atom-level vector features

        # Aggregate edge-level scalar output to atom-level using scatter_reduce (sum)
        atom_level_output.index_add_(0, edge_index[0], edge_level_output)  # Sum edge outputs to source atoms

        # Aggregate edge-level vector features to atom-level using scatter_reduce (sum)
        # Map edge_vec (shape: (num_edges, 3)) to atom-level vec
        # First, expand edge_vec to match expert_out_features
        edge_vec_expanded = edge_vec.unsqueeze(-1).repeat(1, 1, self.experts[
            0].out_features)  # Shape: (num_edges, 3, expert_out_features)
        # Weight edge_vec_expanded by edge_level_output (broadcasting)
        weighted_edge_vec = edge_vec_expanded * edge_level_output.unsqueeze(
            1)  # Shape: (num_edges, 3, expert_out_features)
        # Aggregate to atom-level using scatter_reduce (sum) for source atoms
        for dim in range(3):  # Iterate over spatial dimensions (0, 1, 2)
            # Extract the dim-th spatial component of weighted_edge_vec
            weighted_edge_vec_dim = weighted_edge_vec[:, dim, :]  # Shape: (num_edges, expert_out_features)
            # Aggregate to atom-level using index_add_ for source atoms
            vec[:, dim, :].index_add_(0, edge_index[0], weighted_edge_vec_dim)

        return atom_level_output_x, vec, z, pos, batch


if __name__ == "__main__":

    import torch
    import numpy as np
    import random
    import time


    def set_seed(seed: int):
        torch.manual_seed(seed)  # Ensures random number generator is deterministic on the cpu
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():  # Makes sure CUDA is set with the same seed, GPU results now reproducable
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True  # GPU ops deterministic
        torch.backends.cudnn.benchmark = False  # Avoid non deterministic behaviour when input size not fixed


    # Set the seed
    set_seed(2000)
    # Test the model
    model = DeepSet(
        base_cutoff=0.0,
        outer_cutoff=3.0,
        # radial_basis=None,
        # use_vector_representation=True,
        # forces_based_on_energy=False,
        # close_far_split=True,
        # using_triplet_module=True
    )

    # Generate random input
    z = torch.randint(1, 100, (18, 1))
    # print(z.size())
    pos = torch.randint(0, 5, (18, 3), dtype=torch.float32)
    batch = torch.zeros(100, dtype=torch.long)
    # box = torch.rand(3, 3)
    x, vec, z, pos, batch = model(z, pos, batch)
