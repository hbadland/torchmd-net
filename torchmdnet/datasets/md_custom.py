import os
import re
import torch
from torch_geometric.data import Data, Dataset


class MDCustom(Dataset):
    def __init__(self, root, start=None, end=None, transform=None, pre_transform=None):
        """
        Args:
            root (str): Root directory where the dataset is stored.
            start (int, optional): Start index for sampling files. Defaults to None.
            end (int, optional): End index for sampling files. Defaults to None.
            transform (callable, optional): A function/transform applied to each data object.
            pre_transform (callable, optional): A function/transform applied before saving data objects.
        """
        self.start = start
        self.end = end
        super().__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        # Apply sampling if start and/or end are provided
        if self.start is not None:
            end = self.end if self.end is not None else float('inf')
            sampled_files = self.sample_files(self.raw_dir, self.start, end)
            return [pos for pos, forces in sampled_files]
        else:
            # Default: return all .pos files in the directory
            return [f for f in os.listdir(self.raw_dir) if f.endswith('.pos')]

    @property
    def processed_file_names(self):
        return [f'data_{i}.pt' for i in range(len(self.raw_file_names))]

    def sample_files(self, directory, start, end):
        """
        Filters and samples files within a numeric range in their filenames.
        """
        pos_files = []
        forces_files = []

        # Regular expression to extract numeric parts
        file_pattern = re.compile(r'part_(\d+)\.(pos|forces)')

        for file in os.listdir(directory):
            match = file_pattern.match(file)
            if match:
                index = int(match.group(1))
                if start <= index <= end:
                    if file.endswith('.pos'):
                        pos_files.append(file)
                    elif file.endswith('.forces'):
                        forces_files.append(file)

        # Ensure both lists are sorted for matching pairs
        pos_files.sort()
        forces_files.sort()

        # Match pos and forces files
        sampled_files = []
        for pos_file in pos_files:
            corresponding_forces = pos_file.replace('.pos', '.forces')
            if corresponding_forces in forces_files:
                sampled_files.append((pos_file, corresponding_forces))

        return sampled_files

    def process(self):
        if self.start is not None:
            end = self.end if self.end is not None else float('inf')
            sampled_files = self.sample_files(self.raw_dir, self.start, end)
        else:
            # Match all .pos and .forces files if no sampling is applied
            sampled_files = self.sample_files(self.raw_dir, float('-inf'), float('inf'))

        for idx, (pos_file, forces_file) in enumerate(sampled_files):
            pos_path = os.path.join(self.raw_dir, pos_file)
            forces_path = os.path.join(self.raw_dir, forces_file)

            # Parse position and forces files
            atom_types, positions, energy = self._parse_file(pos_path)
            _, forces, _ = self._parse_file(forces_path)

            # Create edge index (optional: adjust threshold based on your requirements)
            edge_index = self._get_edge_index(positions)

            # Convert to tensors
            z = torch.tensor(atom_types, dtype=torch.long)  # Node features: atom types
            pos = torch.tensor(positions, dtype=torch.float)  # Node positions
            energy = torch.tensor([energy], dtype=torch.float)  # Energy (label)
            forces = torch.tensor(forces, dtype=torch.float)  # Forces (label dy)

            # Create PyG Data object
            data = Data(z=z, pos=pos, edge_index=edge_index, y=energy, neg_dy=forces)

            # Save processed data
            torch.save(data, os.path.join(self.processed_dir, f'data_{idx}.pt'))

    def _parse_file(self, file_path):
        """Parse position or forces file."""
        with open(file_path, 'r') as f:
            lines = f.readlines()

        # First two lines are headers
        atom_count = int(lines[0].strip())
        metadata = lines[1].strip()

        # Extract energy from metadata
        energy = float(metadata.split('E =')[-1].strip())

        # Extract atomic data
        atom_types = []
        values = []
        for line in lines[2:]:
            parts = line.split()
            atom_types.append(self._atom_to_type(parts[0]))
            values.append([float(v) for v in parts[1:]])

        return atom_types, values, energy

    def _atom_to_type(self, atom):
        """Convert atom symbol to atomic number (e.g., 'C' -> 6, 'H' -> 1)."""
        atom_map = {'H': 1, 'C': 6}  # Extend this map as needed
        return atom_map.get(atom, 0)

    def _get_edge_index(self, positions, threshold=1.6):
        """Generate edge index based on distance threshold."""
        import itertools
        edge_index = []
        for i, j in itertools.combinations(range(len(positions)), 2):
            dist = ((torch.tensor(positions[i]) - torch.tensor(positions[j])) ** 2).sum().sqrt()
            if dist < threshold:
                edge_index.append([i, j])
                edge_index.append([j, i])  # Undirected graph

        return torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        return torch.load(os.path.join(self.processed_dir, f'data_{idx}.pt'),
                          weights_only=False)  # , weights_only=False


class MDCustomInMemory(MDCustom):
    def __init__(self, root, start=None, end=None, transform=None, pre_transform=None):
        super().__init__(root, start, end, transform, pre_transform)
        # Load all data into memory once
        self.data_list = []
        for i in range(self.len()):
            # print(os.path.join(self.processed_dir, f'data_{i}.pt'))
            self.data_list.append(torch.load(os.path.join(self.processed_dir, f'data_{i}.pt'), weights_only=False))

    def get(self, idx):
        data = self.data_list[idx]
        if self.transform:
            data = self.transform(data)
        return data


if __name__ == "__main__":
    # Test the dataset class with and without sampling
    dataset_root = '/home/amir/Projects/Butene/'  # Replace with the actual dataset path

    # Without sampling
    dataset_no_sampling = MDCustom(root=dataset_root)
    print(f"Dataset loaded without sampling: {len(dataset_no_sampling)} samples.")

    # Sampling with start and end
    start_index = 10
    end_index = 1000
    dataset_with_sampling = MDCustom(root=dataset_root, start=start_index, end=end_index)
    print(f"Dataset loaded with sampling from {start_index} to {end_index}: {len(dataset_with_sampling)} samples.")

    # Sampling with only start
    start_index_only = 500
    dataset_with_start_only = MDCustom(root=dataset_root, start=start_index_only)
    print(f"Dataset loaded with sampling from {start_index_only} onward: {len(dataset_with_start_only)} samples.")

    # Access a sample
    print(dataset_with_sampling[0]["dy"])
