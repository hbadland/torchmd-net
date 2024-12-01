# Copyright Universitat Pompeu Fabra 2020-2023  https://www.compscience.org
# Distributed under the MIT License.
# (See accompanying file README.md file or copy at http://opensource.org/licenses/MIT)
import time
import typing
import rich
from os.path import join
from tqdm import tqdm
import torch
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader as GeometricDataLoader
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from lightning import LightningDataModule
from lightning_utilities.core.rank_zero import rank_zero_warn
from torchmdnet import datasets
from torchmdnet.utils import make_splits, MissingEnergyException
from torchmdnet.models.utils import scatter, OptimizedDistance
import warnings


class CustomDataLoader(GeometricDataLoader):
    def __init__(self, dataset, batch_size, collate_fn, **kwargs):
        super().__init__(dataset, batch_size=batch_size, **kwargs)
        self.collate_fn = collate_fn


def generate_triplets_gpu_spatial(connections, pos, cutoff):
    """
    GPU-accelerated spatial indexing for triplet generation.

    Args:
        connections (torch.Tensor): 2D tensor of atom connections (edges).
        pos (torch.Tensor): Positions of atoms (3D coordinates).
        cutoff (float): Distance threshold for filtering triplets.
    Returns:
        torch.Tensor: Tensor of valid triplets (i, j, k).
    """
    # Ensure tensors are on the same device
    connections = connections.to(pos.device)

    # Precompute pairwise distances (can be optimized further)
    pairwise_distances = torch.cdist(pos, pos)

    # Create adjacency matrix
    num_atoms = pos.size(0)
    adjacency = torch.zeros((num_atoms, num_atoms), dtype=torch.bool, device=pos.device)
    adjacency[connections[:, 0], connections[:, 1]] = True
    adjacency[connections[:, 1], connections[:, 0]] = True

    # Preallocate triplets with a generous upper bound
    max_triplets = connections.size(0) * num_atoms
    triplets = torch.zeros((max_triplets, 3), dtype=torch.long, device=pos.device)
    triplet_count = torch.tensor(0, dtype=torch.long, device=pos.device)

    # CUDA kernel-style generation
    @torch.no_grad()
    def generate_triplets_cuda_kernel():
        for idx in range(connections.size(0)):
            i, j = connections[idx]

            # Find potential k atoms (neighbors within cutoff)
            potential_k = torch.where(
                (pairwise_distances[j] < cutoff) &
                (pairwise_distances[i] < cutoff) &
                adjacency[j]
            )[0]

            # Filter out i from potential k
            potential_k = potential_k[potential_k != i]

            # Add valid triplets
            for k in potential_k:
                current_count = triplet_count.item()
                triplets[current_count] = torch.tensor([i, j, k], device=pos.device)
                triplet_count.add_(1)

    # Run the generation
    generate_triplets_cuda_kernel()

    # Trim to actual number of triplets
    return triplets[:triplet_count.item()]


def generate_triplets_optimized(connections, pos, cutoff):
    """
    Highly optimized triplet generation with distance cutoff.

    Args:
        connections (torch.Tensor): 2D tensor of atom connections (edges).
        pos (torch.Tensor): Positions of atoms (3D coordinates).
        cutoff (float): Distance threshold for filtering triplets.
    Returns:
        torch.Tensor: Tensor of valid triplets (i, j, k).
    """
    # Number of atoms
    num_atoms = pos.size(0)

    # Create an efficient adjacency list representation
    max_neighbors = torch.max(torch.bincount(connections[:, 0]))
    neighbor_indices = torch.full((num_atoms, max_neighbors.item()), -1, dtype=torch.long, device=pos.device)
    neighbor_counts = torch.zeros(num_atoms, dtype=torch.long, device=pos.device)

    # Populate the adjacency list
    for u, v in connections:
        idx = neighbor_counts[u]
        if idx < max_neighbors:
            neighbor_indices[u, idx] = v
            neighbor_counts[u] += 1

    # Preallocate memory for triplets (with a generous upper bound)
    max_possible_triplets = connections.size(0) * max_neighbors
    triplets = torch.zeros((max_possible_triplets, 3), dtype=torch.long, device=pos.device)
    triplet_count = torch.tensor(0, dtype=torch.long, device=pos.device)

    # Kernel to generate triplets
    @torch.no_grad()
    def generate_triplets_kernel():
        # Use torch.cuda.jit if using CUDA, otherwise this is a standard function
        for idx in range(connections.size(0)):
            i, j = connections[idx]

            # Iterate through neighbors of j, excluding i
            for k_idx in range(max_neighbors):
                k = neighbor_indices[j, k_idx]

                # Break if no more neighbors or invalid neighbor
                if k == -1 or k == i:
                    break

                # Check distance cutoff
                if torch.norm(pos[i] - pos[k]) < cutoff:
                    # Atomic add to avoid race conditions
                    current_count = triplet_count.item()
                    triplets[current_count] = torch.tensor([i, j, k], device=pos.device)
                    triplet_count.add_(1)

    # Run the kernel
    generate_triplets_kernel()

    # Trim to actual number of triplets
    final_triplets = triplets[:triplet_count.item()]

    return final_triplets


def generate_triplets_optimized_extended(connections, pos, cutoff):
    """
    Highly optimized triplet generation with distance cutoff.
    Considers neighbors of both i and j as potential k atoms.

    Args:
        connections (torch.Tensor): 2D tensor of atom connections (edges).
        pos (torch.Tensor): Positions of atoms (3D coordinates).
        cutoff (float): Distance threshold for filtering triplets.
    Returns:
        torch.Tensor: Tensor of valid triplets (i, j, k).
    """
    # Number of atoms
    num_atoms = pos.size(0)

    # Create an efficient adjacency list representation
    max_neighbors = torch.max(torch.bincount(connections[:, 0]))
    neighbor_indices = torch.full((num_atoms, max_neighbors.item() * 2), -1, dtype=torch.long, device=pos.device)
    neighbor_counts = torch.zeros(num_atoms, dtype=torch.long, device=pos.device)

    # Populate the adjacency list for both i and j
    for u, v in connections:
        # Add neighbors for both u and v
        for atom in [u, v]:
            for neighbor in [u, v]:
                if atom != neighbor:
                    idx = neighbor_counts[atom]
                    if idx < max_neighbors * 2:
                        neighbor_indices[atom, idx] = neighbor
                        neighbor_counts[atom] += 1

    # Preallocate memory for triplets (with a generous upper bound)
    max_possible_triplets = connections.size(0) * max_neighbors * 2
    triplets = torch.zeros((max_possible_triplets, 3), dtype=torch.long, device=pos.device)
    triplet_count = torch.tensor(0, dtype=torch.long, device=pos.device)

    # Kernel to generate triplets
    @torch.no_grad()
    def generate_triplets_kernel():
        for idx in range(connections.size(0)):
            i, j = connections[idx]

            # Iterate through neighbors of both i and j
            for k_idx in range(max_neighbors * 2):
                k = neighbor_indices[j, k_idx]

                # Break if no more neighbors or invalid neighbor
                if k == -1 or k == i or k == j:
                    continue

                # Check distance cutoffs for all three atoms
                if (torch.norm(pos[i] - pos[k]) < cutoff and
                    torch.norm(pos[j] - pos[k]) < cutoff):
                    # Atomic add to avoid race conditions
                    current_count = triplet_count.item()
                    triplets[current_count] = torch.tensor([i, j, k], device=pos.device)
                    triplet_count.add_(1)

    # Run the kernel
    generate_triplets_kernel()

    # Trim to actual number of triplets
    final_triplets = triplets[:triplet_count.item()]

    return final_triplets[:, 0], final_triplets[:, 1], final_triplets[:, 2]

class CollectAtomTriples(torch.nn.Module):

    def forward(
        self,
        idx_i: torch.Tensor,
    ) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Using the neighbors contained within the cutoff shell, generate all unique pairs
        of neighbors and convert them to index arrays. Applied to the neighbor arrays,
        these arrays generate the indices involved in the atom triples.

        Example:
            idx_j[idx_j_triples] -> j atom in triple
            idx_j[idx_k_triples] -> k atom in triple
            Rij[idx_j_triples] -> Rij vector in triple
            Rij[idx_k_triples] -> Rik vector in triple
        """

        _, n_neighbors = torch.unique_consecutive(idx_i, return_counts=True)

        offset = 0
        idx_i_triples = ()
        idx_jk_triples = ()
        for idx in range(n_neighbors.shape[0]):
            triples = torch.combinations(
                torch.arange(offset, offset + n_neighbors[idx]), r=2
            )
            idx_i_triples += (torch.ones(triples.shape[0], dtype=torch.long) * idx,)
            idx_jk_triples += (triples,)
            offset += n_neighbors[idx]

        idx_i_triples = torch.cat(idx_i_triples)

        idx_jk_triples = torch.cat(idx_jk_triples)
        idx_j_triples, idx_k_triples = idx_jk_triples.split(1, dim=-1)

        return idx_i_triples, idx_j_triples.squeeze(-1), idx_k_triples.squeeze(-1)


def custom_collate_fn(batch, distance, collect_triples=None, cutoff=15):
    # Convert the list of data objects to a batch
    batch = Batch.from_data_list(batch)

    # Perform the distance and triples computations
    pos = batch.pos
    batch_idx = batch.batch

    edge_index, edge_weight, edge_vec = distance(pos, batch_idx)

    if collect_triples is not None:
        idx_i_triples, idx_j_triples, idx_k_triples = collect_triples(edge_index.T, pos, cutoff)


        # Store the results in a dictionary
        computed_values = {
            'edge_index': edge_index,
            'edge_weight': edge_weight,
            'edge_vec': edge_vec,
            'idx_i_triples': idx_i_triples,
            'idx_j_triples': idx_j_triples,
            'idx_k_triples': idx_k_triples
        }
    else:
        computed_values = {
            'edge_index': edge_index,
            'edge_weight': edge_weight,
            'edge_vec': edge_vec,
        }

    # Add the dictionary to the batch
    batch.computed_values = computed_values
    return batch


class DataModule(LightningDataModule):
    """A LightningDataModule for loading datasets from the torchmdnet.datasets module.

    Args:
        hparams (dict): A dictionary containing the hyperparameters of the
            dataset. See the documentation of the torchmdnet.datasets module
            for details.
        dataset (torch_geometric.data.Dataset): A dataset to use instead of
            loading a new one from the torchmdnet.datasets module.
    """

    def __init__(self, hparams, dataset=None):
        super(DataModule, self).__init__()
        self.save_hyperparameters(hparams)
        self._mean, self._std = None, None
        self._saved_dataloaders = dict()
        self.dataset = dataset
        
        if self.hparams["pairwise_thread"] == True:
            self.distance = OptimizedDistance(
                cutoff_lower=self.hparams["base_cutoff"],
                cutoff_upper=self.hparams["outer_cutoff"],
                max_num_pairs=self.hparams["max_num_neighbors"],
                return_vecs=self.hparams["return_vecs"],
                loop=self.hparams["loop"],
                box=self.hparams["box"] if self.hparams.get("box") else None,
                long_edge_index=self.hparams["long_edge_index"],
                check_errors=self.hparams["check_errors"],
                strategy=self.hparams["strategy"],
                include_transpose=self.hparams["include_transpose"],
                resize_to_fit=self.hparams["resize_to_fit"],
            )

        self.collect_triples = generate_triplets_optimized_extended if self.hparams["triples_thread"] == True else None

            
            
    def setup(self, stage):
        if self.dataset is None:
            if self.hparams["dataset"] == "Custom":
                self.dataset = datasets.Custom(
                    self.hparams["coord_files"],
                    self.hparams["embed_files"],
                    self.hparams["energy_files"],
                    self.hparams["force_files"],
                    self.hparams["dataset_preload_limit"],
                )
            else:
                dataset_arg = {}
                if self.hparams["dataset_arg"] is not None:
                    dataset_arg = self.hparams["dataset_arg"]
                if self.hparams["dataset"] == "HDF5":
                    dataset_arg["dataset_preload_limit"] = self.hparams[
                        "dataset_preload_limit"
                    ]
                self.dataset = getattr(datasets, self.hparams["dataset"])(
                    self.hparams["dataset_root"], **dataset_arg
                )

        self.idx_train, self.idx_val, self.idx_test = make_splits(
            len(self.dataset),
            self.hparams["train_size"],
            self.hparams["val_size"],
            self.hparams["test_size"],
            self.hparams["seed"],
            join(self.hparams["log_dir"], "splits.npz"),
            self.hparams["splits"],
        )
        rich.print(
            f"train {len(self.idx_train)}, val {len(self.idx_val)}, test {len(self.idx_test)}"
        )

        self.train_dataset = Subset(self.dataset, self.idx_train)
        self.val_dataset = Subset(self.dataset, self.idx_val)
        self.test_dataset = Subset(self.dataset, self.idx_test)

        if self.hparams["standardize"]:
            # Mark as deprecated
            warnings.warn(
                "The standardize option is deprecated and will be removed in the future. ",
                DeprecationWarning,
            )
            self._standardize()



    def train_dataloader(self):
        return self._get_dataloader(self.train_dataset, "train")

    def val_dataloader(self):
        loaders = [self._get_dataloader(self.val_dataset, "val")]
        # To allow to report the performance on the testing dataset during training
        # we send the trainer two dataloaders every few steps and modify the
        # validation step to understand the second dataloader as test data.
        if self._is_test_during_training_epoch():
            loaders.append(self._get_dataloader(self.test_dataset, "test"))
        return loaders

    def test_dataloader(self):
        return self._get_dataloader(self.test_dataset, "test")

    @property
    def atomref(self):
        """Returns the atomref of the dataset if it has one, otherwise None."""
        if hasattr(self.dataset, "get_atomref"):
            return self.dataset.get_atomref()
        return None

    @property
    def mean(self):
        """Returns the mean of the dataset if it has one, otherwise None."""
        return self._mean

    @property
    def std(self):
        """Returns the standard deviation of the dataset if it has one, otherwise None."""
        return self._std

    def _is_test_during_training_epoch(self):
        return (
            len(self.test_dataset) > 0
            and self.hparams["test_interval"] > 0
            and self.trainer.current_epoch > 0
            and self.trainer.current_epoch % self.hparams["test_interval"] == 0
        )

    def _get_dataloader(self, dataset, stage, store_dataloader=True):
        if stage in self._saved_dataloaders and store_dataloader:
            return self._saved_dataloaders[stage]

        if stage == "train":
            batch_size = self.hparams["batch_size"]
        elif stage in ["val", "test"]:
            batch_size = self.hparams["inference_batch_size"]

        shuffle = stage == "train"
        collect_triples = None
        distance = None
        if self.hparams["pairwise_thread"]:
            distance = self.distance
            if self.hparams["triples_thread"]:
                collect_triples = self.collect_triples

            def collate_fn(batch):
                return custom_collate_fn(batch, distance, collect_triples, self.hparams["triplets_cutoff"]) \
                    if collect_triples is not None else custom_collate_fn(batch, distance, None)
            
            dl = CustomDataLoader(
                    dataset=dataset,
                    batch_size=batch_size,
                    num_workers=self.hparams["num_workers"],
                    persistent_workers=False,
                    pin_memory=True,
                    shuffle=shuffle,
                    collate_fn=collate_fn,
                )
        else:
            dl = DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                num_workers=self.hparams["num_workers"],
                persistent_workers=False,
                pin_memory=True,
                shuffle=shuffle,
            )

        if store_dataloader:
            self._saved_dataloaders[stage] = dl
        return dl

    def _standardize(self):
        def get_energy(batch, atomref):
            if "y" not in batch or batch.y is None:
                raise MissingEnergyException()

            if atomref is None:
                return batch.y.clone()

            # remove atomref energies from the target energy
            atomref_energy = scatter(atomref[batch.z], batch.batch, dim=0)
            return (batch.y.squeeze() - atomref_energy.squeeze()).clone()

        data = tqdm(
            self._get_dataloader(self.train_dataset, "val", store_dataloader=False),
            desc="computing mean and std",
        )
        try:
            # only remove atomref energies if the atomref prior is used
            atomref = self.atomref if self.hparams["prior_model"] == "Atomref" else None
            # extract energies from the data
            ys = torch.cat([get_energy(batch, atomref) for batch in data])
        except MissingEnergyException:
            rank_zero_warn(
                "Standardize is true but failed to compute dataset mean and "
                "standard deviation. Maybe the dataset only contains forces."
            )
            return

        # compute mean and standard deviation
        self._mean = ys.mean(dim=0)
        self._std = ys.std(dim=0)



if __name__ == "__main__":
    # Define hyperparameters
    hparams = {
        "dataset": "MD17",
        "dataset_root": "~/data",
        "train_size": 1000,
        "val_size": 100,
        "test_size": 100,
        "seed": 42,
        "log_dir": "logs",
        "splits": None,
        "batch_size": 8,
        "inference_batch_size": 64,
        "num_workers": 6,
        "standardize": True,
        "coord_files": None,
        "embed_files": None,
        "energy_files": None,
        "force_files": None,
        "dataset_preload_limit": 1000,
        "prior_model": None,
        "base_cutoff": 0.0,
        "inner_cutoff": 5.0,
        "outer_cutoff": 15.0,
        "max_num_neighbors": 50000,
        "return_vecs": True,
        "loop": True,
        "long_edge_index": True,
        "check_errors": False,
        "strategy": "auto",
        "include_transpose": False,
        "resize_to_fit": False,
        "pairwise_thread": True,
        "triples_thread": True,
        "dataset_arg": {"molecules": "ethanol"},  # Add this line to include the dataset_arg key
    }

    # Create a DataModule
    dm = DataModule(hparams)
    dm.setup("fit")

    # Get the train DataLoader
    train_loader = dm.train_dataloader()

    # Iterate over the DataLoader and print some information
    for batch in train_loader:
        print("Batch:")
        # print(f"pos: {batch.pos}")
        # print(f"edge_index: {batch.edge_index}")
        # print(f"edge_weight: {batch.edge_weight}")
        print(f"computed_values: {batch.computed_values}")
        break  # Only process the first batch for testing