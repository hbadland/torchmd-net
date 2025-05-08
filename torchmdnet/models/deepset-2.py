from typing import Callable, Dict, Optional, Union, List
import typing
import wandb
import rich
from networkx.classes.filters import hide_edges
from rich import pretty

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as tg
import torch.nn.functional as F
from torch.nn.functional import relu_
from torch.nn.init import xavier_uniform_, zeros_
from torch.nn import init, Sequential, ReLU, SiLU
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GATv2Conv, EdgeConv

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
			in_channels: int = 128,
			hidden_channels: int = 256,
			out_channels: int = 32,
			num_gates: int = 10,
			skip_duplicates: bool = True,
			base_cutoff: float = 0.0,
			outer_cutoff: float = 5.0,
			k: int = 2,
			max_num_neighbors: int = 400,
			embedding_size: int = 256,
			num_rbf: int = 50,
			expert_out_features: int = 128,
			rbf_type: str = "gauss",
			trainable_rbf: bool = False,
			dtype: torch.dtype = torch.float32
	):

		super(DeepSet, self).__init__()

		self.outer_cutoff = outer_cutoff
		self.base_cutoff = base_cutoff
		self.embedding_size = embedding_size
		self.dtype = dtype
		self.skip_duplicates = skip_duplicates
		self.in_channels = in_channels
		self.hidden_channels = hidden_channels
		self.out_channels = out_channels
		self.num_gates = num_gates

		# Atom embedding
		self.atom_embedding = nn.Linear(1, 64)
		# self.projection_layer = nn.Linear(1152, 128)
		# self.projection_layer_x = nn.Linear(128, 256)
		self.gamma_transform = nn.Linear(256, 384)  # ! Had to set to 384 to work with experts
		# self.gnn = nn.Sequential(SAGEConv(in_channels, hidden_channels), nn.SiLU(), SAGEConv(hidden_channels, hidden_channels))

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
		self.distance_proj = nn.Linear(64, hidden_channels, dtype=dtype)  # transforms size num_rbf into embedding_size
		self.cutoff = CosineCutoff(base_cutoff, outer_cutoff)  # Restrict interactions beyond a certain distance
		self.embedding = nn.Embedding(100, embedding_size, dtype=dtype)  # Embedding layer maps ints

		self.neighbor_embedding = NeighborEmbedding(embedding_size, 20, base_cutoff, outer_cutoff, 100, dtype)
		# Part of a message passing NN/GNN
		expanded_feature_dim = embedding_size + 3
		self.d_ij_transform = nn.Linear(259, 128, dtype=dtype)  # Refine how inter atomic distances contribute
		self.a_i_transform = nn.Linear(128, 128,
									   dtype=dtype)  # Project atom features into a common space before interactions
		self.a_j_transform = nn.Linear(128, 256,
									   dtype=dtype)  # When node i receives messages from j (neighbors) those messages are transformed properly.

		# ? New code 26/03
		# self.pairwise_transform = nn.Linear(embedding_size + 3, embedding_size, dtype=dtype)

		# self.edge = EdgeConv(128, 256)
		# self.norm1 = torch.nn.LayerNorm(256) Batch norm NOT for MD
		# self.edge = EdgeConv(256, hidden_channels, aggr='mean')
		# nn_edge = Sequential(
		# nn.Linear(512, hidden_channels),
		# SiLU(),
		# nn.Linear(hidden_channels, hidden_channels)
		# )

		# self.edge1 = EdgeConv(nn_edge, aggr=' mean')

		# self.gat = GATConv(
		# in_channels=128,  # Correct clearly the input dimension as numeric (256 here explicitly given)
		# out_channels=hidden_channels,  # Correct clearly specifying numeric output dimensions
		# heads=4,  # Recommended explicitly: clearly defined head number
		# concat=True,  # Recommended explicitly: typically True for GAT
		# dropout=0.1  # Optional explicitly: often helpful in practice
		# )

		# self.sage2 = tg.nn.SAGEConv(256, 256, aggr='mean6')

		# self.sage3 = tg.nn.SAGEConv(256, 256, aggr='max')

		# self.lin = torch.nn.Linear(hidden_channels, out_channels, dtype=dtype)
		# self.lin = torch.nn.Linear(256, 64)

		self.concat_projection = nn.Linear(5 * hidden_channels, 256)

		# nn.init.xavier_uniform_(self.embedding.weight)

		# Gated message passing system
		self.gamma_transform = nn.Linear(512, 256,
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
			nn.Linear(256, 128, dtype=dtype)
			for _ in range(num_gates)  # One expert per gate (to handle all possible gates)
		])

	# Gating mechanism chooses which experts contribute to output.

	def reset_parameters(self):
		...

	# Numpy style docstring
	def forward(self,
				z: torch.Tensor,  # Type hints, here for readability and clarity
				pos: torch.Tensor,
				batch: torch.Tensor,
				# based on batch can say how many indexes belong to one molecule, print nd compare with z + pos
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
		# Beginning the forward pass, ! had to squeeze to fix dimension errors
		# z = z.view(-1, 1)
		x = self.embedding(z)
		# x = nn.Dropout(0.1)(x)
		# print("z shape before SAGEConv:", z.shape)
		# print(f"x shape before SAGEConv: {x.shape}") # should be numnodes, 64

		edge_index, edge_weight, edge_vec = self.distance(pos, batch,
														  box)  # Finds pairs of atoms close and returns which atoms are connected, their distance and direction
		# Edge weight distance, take as x

		# num_nodes = z.shape[0]
		# if edge_index.max() >= num_nodes:
		# print(f'Edge index contains out of bounds value')
		# edge_index = edge_index.clamp(0, num_nodes - 1)

		# edge_index = edge_index.long()

		edge_attr = self.distance_expansion(
			edge_weight)  # Converts raw distances between atoms into a better format, NN struggle with raw values
		# ! New code 26/03
		# edge_proj = self.distance_proj(edge_attr)
		# edge_vec = edge_vec / (torch.norm(edge_vec, dim=1, keepdim=True) + 1e-8)

		# print("edge_weight shape (mask):", (edge_weight != 0).shape) # 142
		# print("edge_attr shape:", edge_attr.shape) # 142, 50

		# edge_index = edge_index[:, edge_weight != 0]
		# edge_weight = edge_weight[edge_weight != 0] # * Prevent self loops (No distance between same an atom so it should be 0)
		# edge_vec = edge_vec[edge_vec != 0]

		# print("After filtering zeros:")
		# print("edge_weight.shape:", edge_weight.shape)
		# print("edge_attr.shape:", edge_attr.shape)

		mask = edge_index[0] != edge_index[
			1]  # Mask used to handle anomalous data points , filters out self loops (source and target)
		if not mask.all():  # Checks for any false values in the mask, if so then following code is executed
			edge_index = edge_index[:, mask]
			edge_weight = edge_weight[mask]
			edge_attr = edge_attr[mask]
			edge_vec = edge_vec[mask]
		# edge_proj = edge_proj[mask]# Mask edge_vec as well

		# print(f"Mask shape:", mask.shape)

		if self.skip_duplicates:  # this remove repeated edges in calculation (it means upper triangle matrix)
			edge_index = edge_index[:, ::2]  # Slicing, selects every second edge so skips over dupes
			edge_weight = edge_weight[::2]  # Slice with 2 to skip duplicate edges and keep only one direction
			edge_attr = edge_attr[::2]
			edge_vec = edge_vec[::2]
		# edge_proj = edge_proj[::2]

		# edge_index = edge_index.float()

		# print(f"Edge index shape: {edge_index.shape}") # [2, 62]
		# print(f"Max index in edge_index: {edge_index.max().item()}") # 17 , none out of bounds
		# print(f"Number of nodes (atoms): {z.shape[0]}")  # 18

		# print(f"Edge weight sample: {edge_weight[:5]}")
		# print(f"Edge vec sample: {edge_vec[:5]}") Aligned properly

		# print(f"Min edge index: {edge_index.min().item()}") # 1
		# print(f"Max edge index: {edge_index.max().item()}") # 17

		# print(f"Node feature shape: {x.shape}")  # ([18, 1, 256)]) but needs to be 2D so is causing an issue

		# Added for project
		edge_weight_sq = edge_weight ** 2
		edge_weight_cube = edge_weight ** 3
		edge_weight_tes = edge_weight ** 4
		edge_weight_sqrt = torch.sqrt(edge_weight)
		edge_weight_log = torch.log(edge_weight)

		edge_weight: torch.Tensor  # Type hint, edge_weight is expected to be a tensor

		# Normalize edge_vec for masked edges (similar to TorchMD_ET)
		edge_vec = edge_vec / torch.norm(edge_vec, dim=1, keepdim=True)
		# edge_vec.squeeze(3)

		# ? New code for RBF here 02/04
		edge_attr_dist = self.distance_expansion(edge_weight)
		#edge_attr_sq = self.distance_expansion(edge_weight_sq)
		edge_attr_cub = self.distance_expansion(edge_weight_cube)
		#edge_attr_tes = self.distance_expansion(edge_weight_tes)
		# edge_attr_sqrt = self.distance_expansion(edge_weight_sqrt)
		#edge_attr_log = self.distance_expansion(edge_weight_log)

		edge_attr = torch.cat([edge_attr_dist, edge_attr_cub], dim=1)

		# print(f"Edge weight shape: {edge_weight.shape}")
		# print(f"Edge vec shape: {edge_vec.shape}")

		# Compute the cutoff and distance projection
		C = self.cutoff(
			edge_weight)  # Applies cutoff function to edge weight tensor, limits interactions to certain threshold.
		d_ij_projection = self.distance_proj(edge_attr) * C.view(-1,
																 1)  # Applies cutoff values to projections. If edge is zero then it is reflected here.

		# print(f"edge_proj shape: {edge_proj.shape}") # 168, 256 clearly the issue
		# print(f"edge_weight_cube shape: {edge_weight_cube.shape}") # 75
		# print(f"edge_weight_sqrt shape: {edge_weight_sqrt.shape}") # 75
		# print(f"edge_weight shape: {edge_weight.shape}") # 75
		# print(f"edge_vec shape: {edge_vec.shape}") # 75,3 thus should flatten

		# edge_vec_flat = edge_vec.mean(dim=1)

		# !Concat new weights to the projection, since dim = 1 need to ensure concat along a column ??
		edge_features = torch.cat(
			[
				d_ij_projection,
				# edge_weight_sq.view(-1,1),
				edge_weight_cube.view(-1, 1),
				# edge_weight_tet.view(-1,1),
				edge_weight_sqrt.view(-1, 1),
				# edge_weight_log.view(-1,1),
				edge_weight.view(-1, 1),
				# edge_vec_flat.view(-1,1)
			], dim=1
		)

		# print(f"edge_proj shape : {edge_proj.shape}")  #
		# print(f"edge_weight_cube : {edge_weight_cube.shape}")  # 75
		# print(f"edge_weight_sqrt shape : {edge_weight_sqrt.shape}")  # 75
		# print(f"edge_weight shape : {edge_weight.shape}")  # 75
		# print(f"edge_vec shape : {edge_vec_flat.shape}")

		# ? I have to transform 260 input to 259 output linearly, here accuracy will be lost

		# print(f"x shape: {x.shape}")  # [18, 256]
		# print(f'edge_features.shape BEFORE Linear: {edge_features.shape}')  # [77, 260]
		# assert edge_features.shape == (77, 260), f"Unexpected shape: {edge_features.shape}"

		# print("Edge features shape:", edge_features.shape)  # (77, 260)
		# print("Expected input dim:", self.d_ij_transform.in_features)  #  260
		# print("Expected output dim:", self.d_ij_transform.out_features)  # 256

		# try:
		# d_ij_t_projection = self.d_ij_transform(edge_features)
		# except RuntimeError as e:
		# print(f'error at d_ij_transform: {e}')
		# raise

		# Transform the distance projection, nuclear charges and atom embeddings
		d_ij_t_projection = self.d_ij_transform(edge_features)
		# print(f'dijt shape: {d_ij_t_projection.shape}') # 77, 256
		a_i_projection = self.a_i_transform(x[edge_index[0, :]])
		a_j_projection = self.a_j_transform(x[edge_index[1, :]])

		# print('Z shape:'z.shape) # 18, 1
		# print('Edge index:'edge_index.shape) # 2, 63

		# print('z.type:',z.dtype) # torch.int64
		# print('edge type:',edge_index.dtype) # torch.int64

		# z = z.float()

		# print(f"x shape: {x.shape}") , # 18 256

		# print(f"x shape before sage1: {x.shape}")
		# print(f"Before GNN: mean {x.mean().item()}, std {x.std().item()}, min {x.min().item()}, max {x.max().item()}")

		# z =self.sage1(x, edge_index)
		# z=F.silu(z)
		# print(f"After SAGE1: mean {z.mean().item()}, std {z.std().item()}, min {z.min().item()}, max {z.max().item()}")

		# print(f"z shape after sage1: {z.shape}")

		# z = self.sage2(z, edge_index)
		# z=F.silu(z)
		# print(f"After SAGE2: mean {z.mean().item()}, std {z.std().item()}, min {z.min().item()}, max {z.max().item()}")

		# print('z shape after 2: ', z.shape)

		# z = self.sage3(z, edge_index)
		# z=F.silu(z)
		# print(f"After SAGE3: mean {z.mean().item()}, std {z.std().item()}, min {z.min().item()}, max {z.max().item()}")

		# print(f"SAGE1 output shape: {x.shape}")
		# print(f"SAGE2 output shape: {x.shape}")
		# print(f"SAGE3 output shape: {x.shape}")

		# Need to specify the gnn output i.e. what it all is then put it here
		# x1 = self.sage1(z, edge_index)
		# x2 = self.sage2(x1, edge_index)  # explicitly giving required edge_index explicitly clearly explicitly explicitly explicitly
		# gnn_output = self.sage3(x2, edge_index)  # explicitly every layer explicitly clearly exactly matched exactly explicitly explicitly

		# ? Atomwise???
		# gnn_output = F.silu(self.edge1(x, edge_index))
		# gnn_output = F.silu(self.edge1(x, edge_index))

		# ? Pairwise gnn output maybe input too? added non linearity
		# gnn_output = torch.cat([x, edge_index], dim=1)

		# print(f"GNN Output Shape: {gnn_output.shape}")
		# print(f"Edge Index Shape: {edge_index.shape}")

		# Extract edge-level features by indexing, 0 = source nodes, 1 = target nodes
		# gnn_edge_features = torch.cat([gnn_output[edge_index[0]], gnn_output[edge_index[1]]], dim=1)

		# Sageconv handles edge_index and edge_weight, also takes z , need to put these into a parameter and pytorch will handlet this

		# print(f"a_i_projection shape: {a_i_projection.shape}")
		# print(f"a_j_projection shape: {a_j_projection.shape}")
		# print(f"d_ij_t_projection shape: {d_ij_t_projection.shape}")
		# print(f"gnn_edge_features shape: {gnn_edge_features.shape}")

		# Ensure all tensors have the same first dimension
		# min_edges = min(a_i_projection.shape[0], a_j_projection.shape[0], d_ij_t_projection.shape[0])
		# a_i_projection = a_i_projection[:min_edges]
		# a_j_projection = a_j_projection[:min_edges]
		# d_ij_t_projection = d_ij_t_projection[:min_edges]
		# gnn_edge_features = gnn_edge_features[:min_edges]

		# concat_gnn = torch.cat([a_i_projection, a_j_projection, d_ij_t_projection, gnn_edge_features], dim=1)
		# concat_gnn = self.projection_layer(concat_gnn)
		# concat_gnn = self.projection_layer_x(concat_gnn)  # New projection to reduce dimensions

		# print(a_i_projection.shape)
		# print(a_j_projection.shape)
		# print(d_ij_t_projection.shape)
		# print(gnn_edge_features.shape)

		# ! New code 26/03
		# concat_gnn = torch.cat([a_i_projection, a_j_projection, d_ij_t_projection, gnn_edge_features], dim=1)
		# print(f"concat_gnn shape before projection: {concat_gnn.shape}")
		# concat_gnn = self.concat_projection(concat_gnn)
		# concat_gnn = concat_gnn.view(-1, 384)
		# print(f"concat_gnn shape: {concat_gnn.shape}")

		# ? Did you want a linear layer here to transform after concat ? self gamma transform is now linear
		# gamma_projection = self.gamma_transform(torch.cat([a_i_projection.squeeze(), a_j_projection.squeeze(), d_ij_t_projection], dim=1))

		gamma_projection = self.gamma_transform(torch.cat([a_i_projection, a_j_projection, d_ij_t_projection], dim=1))

		# gamma_projection = self.gamma_transform(torch.cat([a_i_projection, a_j_projection, d_ij_t_projection], dim=1)) # Pairwise here, try to do this if append below is not accurate

		# Combine the source and target atom features then apply transformation to better learn them

		# Process each feature projection separately
		#gamma_i = self.gamma_transform_i(a_i_projection)
		#gamma_j = self.gamma_transform_j(a_j_projection)
		#gamma_projection = gamma_i + gamma_j  # or any other combination method

		# Computing Z:
		d_ij_expanded = torch.repeat_interleave(edge_weight, self.num_gates, dim=0)

		d_ij_expanded = 1 / torch.clamp(
			torch.abs(d_ij_expanded.view(
				-1,
				self.t_parameters.size(0)
			) - self.t_parameters),
			min=1e-8 # Avoid division by zero
		)
		softmax_d_ij_expanded = F.softmax(d_ij_expanded, dim=1)


		experts_output = [self.experts[i](gamma_projection) for i in range(self.num_gates)]

		experts_contributions = [e_o * e_w for e_o, e_w in zip(experts_output, softmax_d_ij_expanded.split(1, dim=1))]
		# do the sum of the experts_contributions
		edge_level_output = torch.sum(torch.stack(experts_contributions, dim=0), dim=0)

		# Doing aggregation over the atoms
		atom_level_output_x = scatter(edge_level_output, edge_index[0], dim=0, reduce="sum") # Concat gnn_output here

		#Then add another MLP layer here to transfer so I can concat.

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
		torch.manual_seed(seed) # Ensures random number generator is deterministic on the cpu
		np.random.seed(seed)
		random.seed(seed)
		if torch.cuda.is_available(): # Makes sure CUDA is set with the same seed, GPU results now reproducable
			torch.cuda.manual_seed(seed)
			torch.cuda.manual_seed_all(seed)
		torch.backends.cudnn.deterministic = True # GPU ops deterministic
		torch.backends.cudnn.benchmark = False # Avoid non deterministic behaviour when input size not fixed


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
