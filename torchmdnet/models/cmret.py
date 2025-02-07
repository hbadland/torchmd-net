# -*- coding: utf-8 -*-
# Author: Nianze A. Tao (Omozawa Sueno)
"""
Model representation.
"""
from typing import Dict, Optional, List
import torch
import torch.nn as nn
from torch import Tensor
from typing_extensions import Self
# from .output import (
#     EquivariantDipoleMoment,
#     EquivariantScalar,
#     EquivariantPolarizability,
#     ElectronicSpatial,
# )
from typing import Tuple, Optional
import torch
import torch.nn as nn
from torch import Tensor


ORBITALS = "1s 2s 2p 3s 3p 4s 3d 4p 5s 4d 5p 6s 4f 5d 6p 7s 5f 6d 7p 6f 7d 7f".split()
POSSIBLE_ELECTRONS = dict(s=2, p=6, d=10, f=14)


def _electron_config(atomic_num: int) -> List[int]:
    """
    Generate electron configuration for a given atomic number.

    :param atomic_num: atomic number
    :return: electron configuration
    """
    electron_count, last_idx, config = 0, -1, []
    for i in ORBITALS:
        if electron_count < atomic_num:
            config.append(POSSIBLE_ELECTRONS[i[-1]])
            electron_count += POSSIBLE_ELECTRONS[i[-1]]
            last_idx += 1
        else:
            config.append(0)
    if electron_count > atomic_num:
        config[last_idx] -= electron_count - atomic_num
    return config


electron_config = torch.tensor([_electron_config(i) for i in range(119)], dtype=torch.float32)


class Embedding(nn.Module):
    def __init__(self, embedding_dim: int = 128) -> None:
        """
        Nuclear embedding block.

        :param embedding_dim: embedding dimension
        """
        super().__init__()
        self.embed = nn.Embedding.from_pretrained(electron_config)
        self.out = nn.Linear(in_features=22, out_features=embedding_dim, bias=True)

    def forward(self, z: Tensor) -> Tensor:
        """
        :param z: nuclear charges;  shape: (1, n_a)
        :return: embedded tensor;   shape: (1, n_a, embedding_dim)
        """
        return self.out(self.embed(z))


class RBF1(nn.Module):
    def __init__(self, cell: float = 5.0, n_kernel: int = 20) -> None:
        """
        Bessel RBF kernel for 3D input.

        :param cell: unit cell length
        :param n_kernel: number of kernels
        """
        super().__init__()
        self.register_buffer("cell", torch.tensor([cell]))
        offsets = torch.linspace(1.0, n_kernel, n_kernel)
        self.register_buffer("offsets", offsets[None, None, None, :])

    def forward(self, d: Tensor) -> Tensor:
        """
        :param d: a tensor of distances;  shape: (1, n_a, n_a - 1)
        :return: RBF-extanded distances;  shape: (1, n_a, n_a - 1, n_k)
        """
        out = (torch.pi * self.offsets * d[:, :, :, None] / self.cell).sin()
        return out / d.masked_fill(d == 0, torch.inf)[:, :, :, None]


class RBF2(nn.Module):
    def __init__(self, cell: float = 5.0, n_kernel: int = 20) -> None:
        """
        Gaussian RBF kernel for 3D input.

        :param cell: unit cell length
        :param n_kernel: number of kernels
        """
        super().__init__()
        self.register_buffer("cell", torch.tensor([cell]))
        self.register_buffer("n_kernel", torch.tensor([n_kernel]))
        offsets = torch.linspace((-self.cell).exp().item(), 1, n_kernel)
        offsets = offsets[None, None, None, :]
        coeff = ((1 - (-self.cell).exp()) / n_kernel).pow(-2) * torch.ones_like(offsets)
        self.offsets = nn.Parameter(offsets, requires_grad=True)
        self.coeff = nn.Parameter(coeff / 4, requires_grad=True)

    def forward(self, d: Tensor) -> Tensor:
        """
        :param d: a tensor of distances;  shape: (1, n_a, n_a - 1)
        :return: RBF-extanded distances;  shape: (1, n_a, n_a - 1, n_k)
        """
        out = (-self.coeff * ((-d[:, :, :, None]).exp() - self.offsets).pow(2)).exp()
        return out


class CosinCutOff(nn.Module):
    def __init__(self, cutoff: float = 5.0) -> None:
        """
        Compute cosin-cutoff mask.

        :param cutoff: cutoff radius
        """
        super().__init__()
        self.register_buffer("cutoff", torch.tensor([cutoff]))

    def forward(self, d: Tensor) -> Tensor:
        """
        :param d: pair-wise distances;     shape: (1, n_a, n_a - 1)
        :return: cutoff mask;              shape: (1, n_a, n_a - 1)
        """
        cutoff = 0.5 * (torch.pi * d / self.cutoff).cos() + 0.5
        cutoff *= (d <= self.cutoff).float()
        return cutoff


class Distance(nn.Module):
    def __init__(self) -> None:
        """
        Compute pair-wise distances and normalised vectors.
        """
        super().__init__()

    def forward(
        self,
        r: Tensor,
        batch_mask: Tensor,
        loop_mask: Tensor,
        lattice: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        :param r: nuclear coordinates;    shape: (1, n_a, 3)
        :param batch_mask: batch mask;    shape: (1, n_a, n_a, 1)
        :param loop_mask: loop mask;      shape: (1, n_a, n_a)
        :param lattice: lattice vectors;  shape: (1, n_a, 3, 3)
        :return: d, vec_norm;             shape: (1, n_a, n_a - 1), (1, n_a, n_a - 1, 3, 1)
        """
        n_b, n_a, _ = r.shape
        vec = r[:, :, None, :] - r[:, None, :, :]
        vec = vec * batch_mask  # reomve 'off-diagonal' elements
        if lattice is not None:
            # compute distances under periodic boundary conditions
            r_shift1 = r + lattice[::, ::, 0]
            r_shift2 = r + lattice[::, ::, 1]
            r_shift3 = r + lattice[::, ::, 2]
            vec_shift1 = r[:, :, None, :] - r_shift1[:, None, :, :]
            vec_shift2 = r[:, :, None, :] - r_shift2[:, None, :, :]
            vec_shift3 = r[:, :, None, :] - r_shift3[:, None, :, :]
            vecs = torch.cat([vec, vec_shift1, vec_shift2, vec_shift3], dim=0)
            ds = torch.linalg.norm(vecs, 2, -1)
            d_min = torch.min(ds, dim=0)  # find min distances
            d, d_key = d_min.values[None, :, :], d_min.indices
            vec = torch.gather(vecs, 0, d_key[None, :, :, None].repeat(1, 1, 1, 3))
            d_tril = torch.tril(d, -1)
            d_triu = torch.triu(d, 0).transpose_(-2, -1)
            d_tri = torch.cat([d_tril.unsqueeze(0), d_triu.unsqueeze(0)], 0)
            d_tri_min = torch.min(d_tri, dim=0)  # use symmetry
            d_tri, d_key = d_tri_min.values, d_tri_min.indices
            vec_tril = vec * (d_tril != 0).float().unsqueeze_(-1)
            vec_triu = (vec * (d_triu == 0).float().unsqueeze_(-1)).transpose(-2, -3)
            vec_tri = torch.cat([vec_tril, vec_triu], 0)
            vec = torch.gather(vec_tri, 0, d_key[:, :, :, None].repeat(1, 1, 1, 3))
            vec = vec - vec.transpose(-2, -3)
            vec = vec[loop_mask].view(n_b, n_a, n_a - 1, 3)  # remove 0 vectors
            d = (d_tri + d_tri.transpose(-2, -1))[loop_mask].view(1, n_a, n_a - 1)
        else:
            vec = vec[loop_mask].view(n_b, n_a, n_a - 1, 3)  # remove 0 vectors
            d = torch.linalg.norm(vec, 2, -1)
        vec_norm = vec / d.masked_fill(d == 0, torch.inf)[:, :, :, None]
        return d, vec_norm.unsqueeze_(dim=-1)


class EquivariantScalar(nn.Module):
    def __init__(
        self,
        n_feature: int = 128,
        n_output: int = 2,
        dy: bool = False,
        return_vector_feature: bool = False,
    ) -> None:
        """
        Equivariant Scalar output block.

        :param n_feature: input feature
        :param n_output: number of output layers
        :param dy: whether to calculater -dy
        :param return_vector_feature: whether to return the vector features
        """
        super().__init__()
        self.dy = dy
        self.return_v = return_vector_feature
        self.block = nn.ModuleList(
            [GatedEquivariant(n_feature=n_feature) for _ in range(n_output)]
        )
        self.out = nn.Linear(in_features=n_feature, out_features=1, bias=True)

    def forward(self, kargv: Dict[str, Tensor]) -> Dict[str, Tensor]:
        s, v, pos, batch_mask = kargv["s"], kargv["v"], kargv["pos"], kargv["batch"]
        for layer in self.block:
            s, v = layer(s, v)
        s = self.out(s)
        y = (s.repeat(batch_mask.shape[0], 1, 1) * batch_mask).sum(dim=-2)
        # out = {"scalar": y}
        # if self.return_v:
        #     out["R"] = v.mean(dim=-1) + r
        return y


class ResML(nn.Module):
    def __init__(self, dim: int = 128) -> None:
        """
        Residual layer.

        :param dim: input dimension
        """
        super().__init__()
        self.res = nn.Sequential(
            nn.Linear(in_features=dim, out_features=dim, bias=True),
            nn.SiLU(),
            nn.Linear(in_features=dim, out_features=dim, bias=True),
        )
        nn.init.normal_(
            self.res[0].weight, 0, 0.5 * (6 / dim) ** 0.5
        )  # weight init: N[0.0, (3/(2 dim))^1/2]

    def forward(self, x: Tensor) -> Tensor:
        return self.res(x) + x


class CFConv(nn.Module):
    def __init__(self, n_kernel: int = 20, n_feature: int = 128) -> None:
        """
        Improved Contiunous-filter convolution block.

        :param n_kernel: number of RBF kernels
        :param n_feature: number of feature dimensions
        """
        super().__init__()
        self.w1 = nn.Sequential(
            nn.Linear(in_features=n_kernel, out_features=n_feature, bias=True),
            nn.SiLU(),
        )
        self.w2 = nn.Sequential(
            nn.Linear(in_features=n_kernel, out_features=n_feature, bias=True),
            nn.SiLU(),
        )
        self.phi = nn.Sequential(
            nn.Linear(in_features=n_feature, out_features=n_feature, bias=True),
            nn.SiLU(),
        )
        self.o = nn.Linear(
            in_features=n_feature * 2, out_features=n_feature * 3, bias=True
        )
        nn.init.uniform_(
            self.w1[0].weight, -((6 / n_kernel) ** 0.5), (6 / n_kernel) ** 0.5
        )  # weight init: U[-(6/in_dim)^1/2, (6/in_dim)^1/2]
        nn.init.uniform_(
            self.w2[0].weight, -((6 / n_kernel) ** 0.5), (6 / n_kernel) ** 0.5
        )  # weight init: U[-(6/in_dim)^1/2, (6/in_dim)^1/2]
        nn.init.normal_(
            self.phi[0].weight, 0, 0.5 * (6 / n_feature) ** 0.5
        )  # weight init: N[0.0, (3/(2 dim))^1/2]

    def forward(
        self, x: Tensor, e: Tensor, mask: Tensor, loop_mask: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        :param x: input info;              shape: (1, n_a, n_f)
        :param e: extended tensor;         shape: (1, n_a, n_a - 1, n_k)
        :param mask: neighbour mask;       shape: (1, n_a, n_a - 1, 1)
        :param loop_mask: self-loop mask;  shape: (1, n_a, n_a)
        :return: convoluted scalar info;   shape: (1, n_a, n_a - 1, n_f) * 3
        """
        w1, w2 = self.w1(e), self.w2(e)
        x = self.phi(x)
        _, n_a, f = x.shape
        x_nbs = x[:, None, :, :].repeat(1, n_a, 1, 1)
        x_nbs = x_nbs[loop_mask].view(1, n_a, n_a - 1, f)
        v1 = x[:, :, None, :] * w1 * mask
        v2 = x_nbs * w2 * mask
        v = self.o(torch.cat([v1, v2], dim=-1)) * mask
        s1, s2, s3 = v.chunk(3, -1)
        return s1, s2, s3


class NonLoacalInteraction(nn.Module):
    def __init__(
        self, n_feature: int = 128, num_head: int = 4, temperature_coeff: float = 2.0
    ) -> None:
        """
        NonLoacalInteraction block (single/multi-head self-attention).

        :param n_feature: number of feature dimension
        :param num_head: number of attention head
        :param temperature_coeff: temperature coefficient
        """
        super().__init__()
        assert (
            num_head > 0 and n_feature % num_head == 0
        ), f"Cannot split {num_head} attention heads when feature is {n_feature}."
        self.d = n_feature // num_head  # head dimension
        self.nh = num_head  # number of heads
        self.tp = (temperature_coeff * self.d) ** 0.5  # attention temperature
        self.q = nn.Linear(in_features=n_feature, out_features=n_feature, bias=True)
        self.k = nn.Linear(in_features=n_feature, out_features=n_feature, bias=True)
        self.v = nn.Linear(in_features=n_feature, out_features=n_feature, bias=True)
        self.activate = nn.Softmax(dim=-1)

    def forward(
        self, x: Tensor, batch_mask: Tensor, return_attn_matrix: bool = False
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        :param x: input tensor;            shape: (1, n_a, n_f)
        :param batch_mask: batch mask;     shape: (n_a, n_a)
        :param return_attn_matrix: whether to return the attenton matrix
        :return: attention-scored output;  shape: (1, n_a, n_f)
                 attention matrix;         shape: (1, n_a, n_a)
        """
        _, n_a, n_f = x.shape
        q = self.q(x).view(1, n_a, self.nh, self.d).permute(2, 0, 1, 3).contiguous()
        k_t = self.k(x).view(1, n_a, self.nh, self.d).permute(2, 0, 3, 1).contiguous()
        v = self.v(x).view(1, n_a, self.nh, self.d).permute(2, 0, 1, 3).contiguous()
        a = q @ k_t / self.tp
        alpha = self.activate(a.masked_fill_(batch_mask, -torch.inf))
        out = (alpha @ v).permute(1, 2, 0, 3).contiguous().view(1, n_a, n_f)
        if return_attn_matrix:
            return out, alpha.mean(dim=0)
        return out, None


class Interaction(nn.Module):
    def __init__(
        self,
        n_feature: int = 128,
        n_kernel: int = 50,
        num_head: int = 4,
        temperature_coeff: float = 2.0,
    ) -> None:
        """
        Interaction block.

        :param n_feature: number of feature dimension
        :param n_kernel: number of RBF kernels
        :param num_head: number of attention head
        :param temperature_coeff: temperature coefficient
        """
        super().__init__()
        self.cfconv = CFConv(n_kernel=n_kernel, n_feature=n_feature)
        self.nonloacl = NonLoacalInteraction(
            n_feature=n_feature, num_head=num_head, temperature_coeff=temperature_coeff
        )
        self.u = nn.Linear(
            in_features=n_feature, out_features=n_feature * 3, bias=False
        )
        self.o = nn.Linear(in_features=n_feature, out_features=n_feature * 3, bias=True)
        self.res = ResML(dim=n_feature)

    def forward(
        self,
        s: Tensor,
        o: Tensor,
        v: Tensor,
        e: Tensor,
        vec_norm: Tensor,
        mask: Tensor,
        loop_mask: Tensor,
        batch_mask: Tensor,
        return_attn_matrix: bool = False,
    ) -> Tuple[Tensor, Tensor, Tensor, Optional[Tensor]]:
        """
        :param s: scale info;                 shape: (1, n_a, n_f)
        :param o: scale from pervious layer;  shape: (1, n_a, n_f)
        :param v: vector info;                shape: (1, n_a, 3, n_f)
        :param e: rbf extended distances;     shape: (1, n_a, n_a - 1, n_k)
        :param vec_norm: normalised vec;      shape: (1, n_a, n_a - 1, 3, 1)
        :param mask: neighbour mask;          shape: (1, n_a, n_a - 1, 1)
        :param loop_mask: self-loop mask;     shape: (1, n_a, n_a)
        :param batch_mask: batch mask;        shape: (n_a, n_a)
        :param return_attn_matrix: whether to return the attenton matrix
        :return: new scale & output scale & vector info & attention matrix
        """
        s1, s2, s3 = self.cfconv(s, e, mask, loop_mask)
        v1, v2, v3 = self.u(v).chunk(3, -1)
        s_nonlocal, attn_matrix = self.nonloacl(s, batch_mask, return_attn_matrix)
        s_n1, s_n2, s_n3 = self.o(s + s_nonlocal + s1.sum(-2)).chunk(3, -1)
        s_m = s_n1 + s_n2 * (v1 * v2).sum(dim=-2)
        s_out = self.res(s_m) + o
        v = v[:, :, None, :, :]
        v_m = s_n3[:, :, None, :] * v3 + (
            s2[:, :, :, None, :] * v + s3[:, :, :, None, :] * vec_norm
        ).sum(dim=-3)
        return s_m, s_out, v_m, attn_matrix


class GatedEquivariant(nn.Module):
    def __init__(self, n_feature: int = 128) -> None:
        """
        Gated equivariant block.

        :param n_feature: number of feature dimension
        """
        super().__init__()
        self.u = nn.Linear(in_features=n_feature, out_features=n_feature, bias=False)
        self.v = nn.Linear(in_features=n_feature, out_features=n_feature, bias=False)
        self.a = nn.Sequential(
            nn.Linear(in_features=n_feature * 2, out_features=n_feature, bias=True),
            nn.SiLU(),
            nn.Linear(in_features=n_feature, out_features=n_feature * 2, bias=True),
        )

    def forward(self, s: Tensor, v: Tensor) -> Tuple[Tensor, Tensor]:
        """
        :param s: scale info;   shape: (1, n_a, n_f)
        :param v: vector info;  shape: (1, n_a, 3, n_f)
        :return: updated s, updated v
        """
        v1, v2 = self.u(v), self.v(v)
        # add 1e-8 to avoid nan in the gradient of gradient in some extreme cases
        s0 = self.a(torch.cat([s, torch.linalg.norm(v2 + 1e-8, 2, -2)], dim=-1))
        sg, ss = s0.chunk(2, -1)
        vg = v1 * ss[:, :, None, :]
        return sg, vg


if __name__ == "__main__":
    ...


class CMRET(nn.Module):
    def __init__(
        self,
        output: nn.Module,
        cutoff: float = 5.0,
        n_kernel: int = 50,
        n_atom_basis: int = 128,
        n_interaction: int = 6,
        rbf_type: str = "gaussian",
        num_head: int = 4,
        temperature_coeff: float = 2.0,
        dy: bool = True,
    ) -> None:
        """
        CMRET upper representaton.

        :param output: output model from `~cmret.model.output`
        :param cutoff: cutoff radius
        :param n_kernel: number of RBF kernels
        :param n_atom_basis: number of atomic basis
        :param n_interaction: number of interaction blocks
        :param rbf_type: type of rbf basis: 'bessel' or 'gaussian'
        :param num_head: number of attention head per layer
        :param temperature_coeff: temperature coefficient
        :param dy: whether to calculate -dy
        """
        super().__init__()
        self.dy = dy
        self.n = n_atom_basis
        self.embedding = Embedding(embedding_dim=n_atom_basis)
        self.distance = Distance()
        if rbf_type == "bessel":
            self.rbf = RBF1(cell=cutoff, n_kernel=n_kernel)
        elif rbf_type == "gaussian":
            self.rbf = RBF2(cell=cutoff, n_kernel=n_kernel)
        else:
            raise NotImplementedError
        self.cutoff = CosinCutOff(cutoff=cutoff)
        self.interaction = nn.ModuleList(
            [
                Interaction(
                    n_feature=n_atom_basis,
                    n_kernel=n_kernel,
                    num_head=num_head,
                    temperature_coeff=temperature_coeff,
                )
                for _ in range(n_interaction)
            ]
        )
        self.norm = nn.LayerNorm(normalized_shape=n_atom_basis)
        self.out = output

    def reset_parameters(self) -> None:
        """
        Initialize the parameters of the model.
        """
        self.embedding.out.reset_parameters()
        if isinstance(self.rbf, RBF1):
            self.rbf.offsets.data.uniform_(-1, 1)
        elif isinstance(self.rbf, RBF2):
            self.rbf.offsets.data.uniform_(-1, 1)
            self.rbf.coeff.data.fill_(0.25)
        for layer in self.interaction:
            layer.cfconv.w1[0].reset_parameters()
            layer.cfconv.w2[0].reset_parameters()
            layer.cfconv.phi[0].reset_parameters()
            layer.cfconv.o.reset_parameters()
            layer.nonloacl.q.reset_parameters()
            layer.nonloacl.k.reset_parameters()
            layer.nonloacl.v.reset_parameters()
            layer.u.reset_parameters()
            layer.o.reset_parameters()
            layer.res.res[0].reset_parameters()
            layer.res.res[2].reset_parameters()
        self.norm.reset_parameters()

    def forward(
        self,
        z: Tensor,
        pos: Tensor,
        batch: Tensor,
        box: Optional[Tensor] = None,
        q: Optional[Tensor] = None,
        s: Optional[Tensor] = None,
        return_attn_matrix: bool = False,
        average_attn_matrix_over_layers: bool = True,
    ) -> Dict[str, Tensor]:
        """
        :param mol: molecule = {
            "Z": nuclear charges tensor;        shape: (1, n_a)
            "pos": nuclear coordinates tensor;  shape: (1, n_a, 3)
            "batch": batch mask;                shape: (n_b, n_a, 1)
            "lattice": lattice vectors;         shape: (n_b, 3, 3) which is optional
            "Q": total charge tensor;           shape: (n_b, 1) which is optional
            "S": spin state tensor;             shape: (n_b, 1) which is optional
        }
        :param return_attn_matrix: whether to return the attention matrices
        :param average_attn_matrix_over_layers: whether to average the attention matrices over layers
        :return: molecular properties (e.g. energy, atomic forces, dipole moment)
        """
        # if "lattice" in mol:
        #     lattice = mol["lattice"]
        #     lattice = (lattice[:, None, :, :] * batch[:, :, :, None]).sum(0, True)
        v = torch.zeros_like(pos)[:, :, :, None].repeat(1, 1, 1, self.n)
        lattice: Optional[Tensor] = None
        if q is not None:
            q_info = q
            z_info = z.repeat(q_info.shape[0], 1) * batch.squeeze(dim=-1)
            q_info = q_info / z_info.sum(dim=-1, keepdim=True)
            v_ = v.repeat(q_info.shape[0], 1, 1, 1) * batch[:, :, :, None]
            v_ = v_ - q_info[:, :, None, None]
            v = v_[batch.squeeze(-1) != 0].view(v.shape)
        if s is not None:
            s_info = s
            z_info = z.repeat(s_info.shape[0], 1) * batch.squeeze(dim=-1)
            s_info = s_info / z_info.sum(dim=-1, keepdim=True)
            v_ = v.repeat(s_info.shape[0], 1, 1, 1) * batch[:, :, :, None]
            v_ = v_ + s_info[:, :, None, None]
            v = v_[batch.squeeze(-1) != 0].view(v.shape)
        # --------- compute loop mask that removes the self-loop ----------------
        loop_mask = torch.eye(z.shape[-1], device=z.device)
        loop_mask = loop_mask[None, :, :] == 0
        # -----------------------------------------------------------------------
        s = self.embedding(z)
        o = torch.zeros_like(s)
        # ---- compute batch mask that seperates atoms in different molecules ----
        # batch = batch.view(1, -1)
        batch_mask_ = batch.squeeze(-1).float().transpose(-2, -1) @ batch.squeeze(-1).float()
        # batch_mask_ = batch.squeeze(-1).transpose(-2, -1) @ batch.squeeze(-1)
        batch_mask = batch_mask_[None, :, :, None]  # shape: (1, n_a, n_a, 1)
        batch_mask_ = batch_mask_ == 0  # shape: (n_a, n_a)
        # ------------------------------------------------------------------------
        d, vec_norm = self.distance(pos, batch_mask, loop_mask, lattice)
        cutoff = self.cutoff(d).unsqueeze(dim=-1)
        h = batch_mask.shape[1]
        cutoff *= batch_mask[loop_mask].view(1, h, h - 1, 1)
        e = self.rbf(d) * cutoff
        attn = []
        for layer in self.interaction:
            s, o, v, _attn = layer(
                s=s,
                o=o,
                v=v,
                e=e,
                vec_norm=vec_norm,
                mask=cutoff,
                loop_mask=loop_mask,
                batch_mask=batch_mask_,
                return_attn_matrix=return_attn_matrix,
            )
            if _attn is not None:
                attn.append(_attn)
        o = self.norm(o)
        x = self.out(dict(z=z, s=o, v=v, pos=pos, batch=batch))
        # if retuof feature dimensionrn_attn_matrix:
        #     if average_attn_matrix_over_layers:
        #         out["attn_matrix"] = torch.cat(attn, dim=0).mean(dim=0)
        #     else:
        #         out["attn_matrix"] = torch.cat(attn, dim=0)
        return x, None, z, pos, batch


class CMRETModel(nn.Module):
    def __init__(
        self,
        cutoff: float = 5.0,
        n_kernel: int = 50,
        n_atom_basis: int = 128,
        n_interaction: int = 6,
        n_output: int = 2,
        rbf_type: str = "gaussian",
        num_head: int = 4,
        temperature_coeff: float = 2.0,
        output_mode: str = "energy-force",
    ) -> None:
        """
        CMRET model.

        :param cutoff: cutoff radius
        :param n_kernel: number of RBF kernels
        :param n_atom_basis: number of atomic basis
        :param n_interaction: number of interaction blocks
        :param n_output: number of output blocks
        :param rbf_type: type of rbf basis: 'bessel' or 'gaussian'
        :param num_head: number of attention head per layer
        :param temperature_coeff: temperature coefficient
        :param output_mode: output properties
        """
        super().__init__()
        dy: bool = False
        self.unit = None  # this parameter will be a 'str' after loading trained model
        out = EquivariantScalar(n_feature=n_atom_basis, n_output=n_output, dy=False)
        # if output_mode == "energy-force":
        #     out = EquivariantScalar(n_feature=n_atom_basis, n_output=n_output, dy=True)
        #     dy = True
        # elif output_mode == "energy":
        #     out = EquivariantScalar(n_feature=n_atom_basis, n_output=n_output, dy=False)
        # elif output_mode == "dipole moment":
        #     out = EquivariantDipoleMoment(n_feature=n_atom_basis, n_output=n_output)
        # elif output_mode == "polarizability":
        #     out = EquivariantPolarizability(n_feature=n_atom_basis, n_output=n_output)
        # elif output_mode == "electronic spatial":
        #     out = ElectronicSpatial(n_feature=n_atom_basis, n_output=n_output)
        # elif output_mode == "pretrain":
        #     out = EquivariantScalar(
        #         n_feature=n_atom_basis,
        #         n_output=n_output,
        #         dy=dy,
        #         return_vector_feature=True,
        #     )
        # else:
        #     raise NotImplementedError(f"{output_mode} is not defined.")
        self.model = CMRET(
            output=out,
            cutoff=cutoff,
            n_kernel=n_kernel,
            n_atom_basis=n_atom_basis,
            n_interaction=n_interaction,
            rbf_type=rbf_type,
            num_head=num_head,
            temperature_coeff=temperature_coeff,
            dy=dy,
        )
        self.args = dict(
            cutoff=cutoff,
            n_kernel=n_kernel,
            n_atom_basis=n_atom_basis,
            n_interaction=n_interaction,
            n_output=n_output,
            rbf_type=rbf_type,
            num_head=num_head,
            temperature_coeff=temperature_coeff,
            output_mode=output_mode,
        )

    def reset_parameters(self) -> None:
        """
        Initialize the parameters of the model.
        """
        self.model.reset_parameters()

    def forward(
        self,
        z: Tensor,
        pos: Tensor,
        batch: Tensor,
        box: Optional[Tensor] = None,
        q: Optional[Tensor] = None,
        s: Optional[Tensor] = None,
        return_attn_matrix: bool = False,
        average_attn_matrix_over_layers: bool = True,
    ) -> Dict[str, Tensor]:
        """
        :param mol: molecule = {
            "Z": nuclear charges tensor;      shape: (1, n_a)
            "R": nuclear coordinates tensor;  shape: (1, n_a, 3)
            "batch": batch mask;              shape: (n_b, n_a, 1)
            "lattice": lattice vectors;       shape: (n_b, 3, 3) which is optional
            "Q": total charge tensor;         shape: (n_b, 1) which is optional
            "S": spin state tensor;           shape: (n_b, 1) which is optional
        }
        :param return_attn_matrix: whether to return the attention matrices
        :param average_attn_matrix_over_layers: whether to average the attention matrices over layers
        :return: molecular properties (e.g. energy, atomic forces, dipole moment)
        """
        n_b = batch.max().item() + 1
        n_a = batch.size(0)

        # Create a one-hot encoded tensor
        batch_one_hot = torch.nn.functional.one_hot(batch, num_classes=n_b).float()

        # Reshape to the desired shape (n_b, n_a, 1)
        batch = batch_one_hot.permute(1, 0).unsqueeze(-1)
        z = z.unsqueeze(0)
        pos = pos.unsqueeze(0).view(1, -1, 3)
        # batch = batch.view(20, -1, 1)
        # pos.requires_grad_(self.dy)

        return self.model(
            z,
            pos,
            batch,
            box,
            q,
            s,
            return_attn_matrix,
            average_attn_matrix_over_layers,
        )

    @classmethod
    def from_checkpoint(cls, file: str) -> Self:
        """
        Return the model from a checkpoint (with 'args' key stored).

        :param file: checkpoint file name <file>
        :return: stored model
        """
        with open(file, mode="rb") as f:
            state_dict = torch.load(f, map_location="cpu")
        nn = state_dict["nn"]
        args = state_dict["args"]
        unit = state_dict["unit"]
        model = CMRETModel(
            cutoff=args["cutoff"],
            n_kernel=args["n_kernel"],
            n_atom_basis=args["n_atom_basis"],
            n_interaction=args["n_interaction"],
            n_output=args["n_output"],
            rbf_type=args["rbf_type"],
            num_head=args["num_head"],
            temperature_coeff=args["temperature_coeff"],
            output_mode=args["output_mode"],
        )
        model.load_state_dict(state_dict=nn)
        model.unit = unit
        return model


if __name__ == "__main__":
    # Create a mock molecule input
    z = torch.tensor([[1, 6, 8]])  # Example nuclear charges (H, C, O)
    pos = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])  # Example coordinates
    batch = torch.tensor([[[1], [1], [1]]])  # Batch mask
    q = torch.tensor([[0.0]])  # Example total charge
    s = torch.tensor([[0.5]])  # Example spin state

    # Initialize the model
    model = CMRET(
        cutoff=5.0,
        n_kernel=50,
        n_atom_basis=128,
        n_interaction=6,
        rbf_type="gaussian",
        num_head=4,
        temperature_coeff=2.0,
    )

    # Run the forward pass
    output = model.forward(
        z=z,
        pos=pos,
        batch=batch,
        q=q,
        s=s,
        return_attn_matrix=False,
        average_attn_matrix_over_layers=True,
    )

    # Print the output
    print(output)