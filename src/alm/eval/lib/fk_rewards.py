"""Feynman-Kac rewards for inference-time steering of MatterGen."""
from __future__ import annotations


import sys
from pathlib import Path
from typing import Iterable, Mapping, Protocol

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

_ALM_DIR = Path(__file__).resolve().parents[1]
if str(_ALM_DIR) not in sys.path:
    sys.path.insert(0, str(_ALM_DIR))

from composition_utils import symbol_to_z  # noqa: E402


class Reward(Protocol):
    """Callable contract for FK rewards, called once per FK-active denoising step."""
    name: str

    def __call__(self, x_hat_0: dict, t: float, t_idx: int) -> torch.Tensor: ...


class WeightedSum:
    """Multi-reward composer: total = sum_i lambda_i * reward_i(...)."""

    def __init__(self, rewards: list[tuple[Reward, float]]):
        self.rewards = rewards
        self.last_components: dict[str, torch.Tensor] = {}
        self.last_weights: dict[str, float] = {r.name: lam for r, lam in rewards}

    def __call__(self, x_hat_0, t, t_idx):
        total = None
        self.last_components = {}
        for r, lam in self.rewards:
            v = r(x_hat_0, t, t_idx)
            self.last_components[r.name] = v.detach()
            weighted = lam * v
            total = weighted if total is None else total + weighted
        return total


def _expand_multiset_to_size(target_counts: Mapping[int, int], n_atoms: int) -> list[int]:
    """Scale target_counts to n_atoms via largest-remainder rounding; returns length-n_atoms Z list."""
    if n_atoms <= 0:
        return []
    total = sum(target_counts.values())
    if total == 0:
        return []
    scaled = {z: n * n_atoms / total for z, n in target_counts.items()}
    floors = {z: int(v) for z, v in scaled.items()}
    remainders = sorted(
        ((z, scaled[z] - floors[z]) for z in scaled),
        key=lambda zr: -zr[1],
    )
    deficit = n_atoms - sum(floors.values())
    for i in range(deficit):
        floors[remainders[i % len(remainders)][0]] += 1
    multiset: list[int] = []
    for z, n in floors.items():
        multiset.extend([z] * n)
    if len(multiset) > n_atoms:
        return multiset[:n_atoms]
    if len(multiset) < n_atoms:
        # Defensive pad with the highest-count target element if rounding undershoots.
        pad_z = max(target_counts, key=lambda z: target_counts[z])
        multiset.extend([pad_z] * (n_atoms - len(multiset)))
    return multiset


class StoichiometricMatchReward:
    """Per-atom Hungarian-assignment NLL between predicted Z distribution and the target multiset.

    eps=1e-4 floors per-atom NLL at ~9.2; a smaller floor saturates the FK hook's
    log_w_clip and degenerates resampling into uniform.
    """

    name = "stoich_match"

    def __init__(self, target_counts: Mapping[int, int], eps: float = 1e-4):
        self.target_counts = dict(target_counts)
        self.eps = eps

    def __call__(self, x_hat_0, t, t_idx, *, _probs_override=None):
        if _probs_override is not None:
            probs = _probs_override
        else:
            probs = torch.softmax(x_hat_0["atomic_numbers_logits"][:, :100], dim=-1)
        batch_idx = x_hat_0["batch_idx"]
        n_particles = int(batch_idx.max().item()) + 1
        scores = torch.zeros(n_particles, device=probs.device, dtype=probs.dtype)

        for p in range(n_particles):
            mask = batch_idx == p
            atom_probs = probs[mask]                            # (N_p, 100)
            n_p = int(atom_probs.shape[0])
            if n_p == 0:
                scores[p] = -20.0
                continue
            target_zs = _expand_multiset_to_size(self.target_counts, n_p)
            target_cols = [z - 1 for z in target_zs]            # 0-indexed
            p_at_targets = atom_probs[:, target_cols]           # (N_p, N_p)
            cost = -(p_at_targets + self.eps).log()             # (N_p, N_p)
            cost_np = cost.detach().cpu().numpy()
            rows, cols = linear_sum_assignment(cost_np)
            assigned_costs = cost[rows, cols]                   # (N_p,)
            scores[p] = -assigned_costs.mean()
        return scores


class StoichCountL1Reward:
    """Per-particle hard-argmax count L1 vs target counts (rescaled to N_p), in [-1, 0]."""

    name = "count_l1"

    def __init__(self, target_counts: Mapping[int, int]):
        self.target_counts = dict(target_counts)

    def __call__(self, x_hat_0, t, t_idx, *, _probs_override=None):
        if _probs_override is not None:
            probs = _probs_override
        else:
            probs = torch.softmax(x_hat_0["atomic_numbers_logits"][:, :100], dim=-1)
        batch_idx = x_hat_0["batch_idx"]
        n_particles = int(batch_idx.max().item()) + 1
        argmax_z = probs.argmax(dim=-1)                          # 0-indexed
        scores = torch.zeros(n_particles, device=probs.device, dtype=probs.dtype)
        for p in range(n_particles):
            mask = batch_idx == p
            n_p = int(mask.sum().item())
            if n_p == 0:
                scores[p] = -1.0
                continue
            target_zs = _expand_multiset_to_size(self.target_counts, n_p)
            tgt = torch.zeros(100, device=probs.device, dtype=probs.dtype)
            for z in target_zs:
                tgt[z - 1] += 1.0
            gen = torch.zeros(100, device=probs.device, dtype=probs.dtype)
            gen.scatter_add_(
                0, argmax_z[mask], torch.ones(n_p, device=probs.device, dtype=probs.dtype)
            )
            scores[p] = -0.5 * (gen - tgt).abs().sum() / n_p
        return scores


class StoichRatioKLReward:
    """Jensen-Shannon between hard-count and target distributions, normalized to [-1, 0]."""

    name = "ratio_kl"

    def __init__(self, target_counts: Mapping[int, int]):
        self.target_counts = dict(target_counts)

    def __call__(self, x_hat_0, t, t_idx, *, _probs_override=None):
        if _probs_override is not None:
            probs = _probs_override
        else:
            probs = torch.softmax(x_hat_0["atomic_numbers_logits"][:, :100], dim=-1)
        batch_idx = x_hat_0["batch_idx"]
        n_particles = int(batch_idx.max().item()) + 1
        argmax_z = probs.argmax(dim=-1)
        scores = torch.zeros(n_particles, device=probs.device, dtype=probs.dtype)
        log2 = float(np.log(2.0))
        for p in range(n_particles):
            mask = batch_idx == p
            n_p = int(mask.sum().item())
            if n_p == 0:
                scores[p] = -1.0
                continue
            target_zs = _expand_multiset_to_size(self.target_counts, n_p)
            q = torch.zeros(100, device=probs.device, dtype=probs.dtype)
            for z in target_zs:
                q[z - 1] += 1.0
            q = q / n_p
            p_dist = torch.zeros(100, device=probs.device, dtype=probs.dtype)
            p_dist.scatter_add_(
                0, argmax_z[mask], torch.ones(n_p, device=probs.device, dtype=probs.dtype)
            )
            p_dist = p_dist / n_p
            m = 0.5 * (p_dist + q)
            kl_p = (p_dist * ((p_dist + 1e-12).log() - (m + 1e-12).log())).sum()
            kl_q = (q * ((q + 1e-12).log() - (m + 1e-12).log())).sum()
            js = 0.5 * (kl_p + kl_q)
            scores[p] = -(js / log2)
        return scores


class InSetCompletenessReward:
    """Mean over target elements of P(at least one atom present), via inclusion-exclusion."""

    name = "in_set_completeness"

    def __init__(self, target_elements: Iterable[str], mode: str = "soft"):
        s2z = symbol_to_z()
        self.target_zs = sorted(
            {s2z[s.strip()] for s in target_elements if s.strip() in s2z}
        )
        self.mode = mode

    def __call__(self, x_hat_0, t, t_idx, *, _probs_override=None):
        if self.mode == "hard":
            if _probs_override is not None:
                raise ValueError("hard mode does not support _probs_override")
            zs = x_hat_0["atomic_numbers_logits"][:, :100].argmax(dim=-1) + 1
            n_particles = int(x_hat_0["batch_idx"].max().item()) + 1
            scores = []
            for p in range(n_particles):
                present = set(zs[x_hat_0["batch_idx"] == p].tolist())
                hit = sum(1 for z in self.target_zs if z in present)
                scores.append(hit / max(len(self.target_zs), 1))
            return torch.tensor(scores, device=zs.device, dtype=torch.float32)

        if _probs_override is not None:
            probs = _probs_override
        else:
            probs = torch.softmax(x_hat_0["atomic_numbers_logits"][:, :100], dim=-1)
        batch_idx = x_hat_0["batch_idx"]
        n_particles = int(batch_idx.max().item()) + 1
        target_cols = [z - 1 for z in self.target_zs]
        scores = torch.zeros(n_particles, device=probs.device, dtype=probs.dtype)
        for p in range(n_particles):
            atom_probs_targets = probs[batch_idx == p][:, target_cols]
            p_present = 1.0 - (1.0 - atom_probs_targets).prod(dim=0)
            scores[p] = p_present.mean()
        return scores


class PhysicalSanityReward:
    """Soft [-1,0] penalty for violating calibrated population bounds on density,
    vol/atom, min pair distance, aspect ratio, and lattice angles (no per-row leakage).
    """

    name = "physical_sanity"

    def __init__(self, bounds: Mapping[str, float]):
        from ase.data import atomic_masses
        self._mass = torch.tensor(list(atomic_masses), dtype=torch.float32)
        self.density_min       = float(bounds["density_min"])
        self.density_max       = float(bounds["density_max"])
        self.vol_per_atom_min  = float(bounds["vol_per_atom_min"])
        self.vol_per_atom_max  = float(bounds["vol_per_atom_max"])
        self.min_pair_dist_min = float(bounds["min_pair_distance_min"])
        self.aspect_ratio_max  = float(bounds["aspect_ratio_max"])
        self.angle_min_min     = float(bounds["angle_min_min"])
        self.angle_max_max     = float(bounds["angle_max_max"])
        self._density_scale = 1.66054  # amu/A^3 -> g/cm^3

    @staticmethod
    def _viol_lo(val: float, lo: float) -> float:
        return max(0.0, (lo - val) / max(abs(lo), 1e-3))

    @staticmethod
    def _viol_hi(val: float, hi: float) -> float:
        return max(0.0, (val - hi) / max(abs(hi), 1e-3))

    def __call__(self, x_hat_0, t, t_idx, *, _probs_override=None):
        if _probs_override is not None:
            probs = _probs_override
        else:
            probs = torch.softmax(x_hat_0["atomic_numbers_logits"][:, :100], dim=-1)
        pos = x_hat_0["pos"]            # (N_atoms_total, 3) cartesian
        cell = x_hat_0["cell"]          # (n_particles, 3, 3)
        batch_idx = x_hat_0["batch_idx"]  # (N_atoms_total,)
        if pos is None or cell is None:
            n_particles = int(batch_idx.max().item()) + 1
            return torch.zeros(n_particles, device=batch_idx.device,
                               dtype=torch.float32)
        n_particles = int(batch_idx.max().item()) + 1
        device = pos.device
        dtype = pos.dtype
        argmax_z = probs.argmax(dim=-1) + 1  # 1-indexed Z
        mass_table = self._mass.to(device=device, dtype=dtype)
        scores = torch.zeros(n_particles, device=device, dtype=dtype)

        for p in range(n_particles):
            mask = batch_idx == p
            n_p = int(mask.sum().item())
            if n_p == 0:
                scores[p] = -1.0
                continue
            zs = argmax_z[mask]
            pp = pos[mask]                                          # (n_p, 3)
            C = cell[p]                                             # (3, 3)
            vol = torch.abs(torch.det(C)).clamp(min=1e-3)
            mass_amu = mass_table[zs].sum()
            density = float(self._density_scale * mass_amu / vol)
            vpa = float(vol / n_p)
            # Min pair distance, minimum-image convention.
            if n_p > 1:
                inv_C = torch.linalg.inv(C)
                d = pp.unsqueeze(0) - pp.unsqueeze(1)               # (n_p, n_p, 3)
                df = d @ inv_C
                df = df - torch.round(df)
                d_min = df @ C
                dists = torch.linalg.vector_norm(d_min, dim=-1)
                eye = torch.eye(n_p, device=device, dtype=torch.bool)
                dists = dists.masked_fill(eye, float("inf"))
                min_pair = float(dists.min())
            else:
                min_pair = float("inf")
            lengths = torch.linalg.vector_norm(C, dim=-1)           # (3,)
            ar = float(lengths.max() / lengths.min().clamp(min=1e-3))
            a, b, c = C[0], C[1], C[2]
            la = torch.linalg.vector_norm(a)
            lb = torch.linalg.vector_norm(b)
            lc = torch.linalg.vector_norm(c)
            cos_alpha = (b @ c) / (lb * lc).clamp(min=1e-3)
            cos_beta  = (a @ c) / (la * lc).clamp(min=1e-3)
            cos_gamma = (a @ b) / (la * lb).clamp(min=1e-3)
            alpha = float(torch.rad2deg(torch.acos(cos_alpha.clamp(-1, 1))))
            beta  = float(torch.rad2deg(torch.acos(cos_beta.clamp(-1, 1))))
            gamma = float(torch.rad2deg(torch.acos(cos_gamma.clamp(-1, 1))))
            angle_min = min(alpha, beta, gamma)
            angle_max = max(alpha, beta, gamma)

            viols = [
                min(1.0, self._viol_lo(density,   self.density_min)
                       + self._viol_hi(density,   self.density_max)),
                min(1.0, self._viol_lo(vpa,       self.vol_per_atom_min)
                       + self._viol_hi(vpa,       self.vol_per_atom_max)),
                min(1.0, self._viol_lo(min_pair,  self.min_pair_dist_min)),
                min(1.0, self._viol_hi(ar,        self.aspect_ratio_max)),
                min(1.0, self._viol_lo(angle_min, self.angle_min_min)),
                min(1.0, self._viol_hi(angle_max, self.angle_max_max)),
            ]
            scores[p] = -float(sum(viols)) / len(viols)
        return scores


class MatterSimEnergyReward:
    """Soft [-1,0] penalty proportional to MatterSim energy/atom; biases toward low-energy structures."""

    name = "mattersim_energy"

    def __init__(
        self,
        e_offset: float = -3.0,
        e_scale: float = 2.0,
        device: str = "cuda",
        potential_path: str | None = None,
        relative_to_batch: bool = True,
        direction: str = "lower",
    ):
        from mattersim.forcefield.potential import Potential
        if potential_path is not None:
            self.potential = Potential.from_checkpoint(load_path=potential_path, device=device)
        else:
            self.potential = Potential.from_checkpoint(device=device)
        self.potential.model.eval()
        self.e_offset = float(e_offset)
        self.e_scale = float(e_scale)
        self.device = device
        self.relative_to_batch = bool(relative_to_batch)
        # "lower" rewards low energy (stability); "higher" rewards less-stable polymorphs.
        assert direction in ("lower", "higher"), direction
        self.direction = direction
        self._sign = -1.0 if direction == "lower" else +1.0

    @torch.no_grad()
    def __call__(self, x_hat_0, t, t_idx):
        from ase import Atoms as _Atoms
        from mattersim.datasets.utils.build import build_dataloader

        pos = x_hat_0["pos"]
        cell = x_hat_0["cell"]
        batch_idx = x_hat_0["batch_idx"]
        n_particles = int(batch_idx.max().item()) + 1
        if pos is None or cell is None:
            return torch.zeros(n_particles, device=batch_idx.device, dtype=torch.float32)
        # Prefer observed Z (CSP-mode true chemistry); fall back to logits in DNG-mode.
        _obs_z = x_hat_0.get("atomic_numbers")
        if _obs_z is not None:
            argmax_z = _obs_z.detach().long().reshape(-1).cpu().tolist()
        else:
            probs = torch.softmax(x_hat_0["atomic_numbers_logits"][:, :100], dim=-1)
            argmax_z = (probs.argmax(dim=-1) + 1).cpu().tolist()
        pos_cpu = pos.detach().cpu().numpy()
        cell_cpu = cell.detach().cpu().numpy()
        batch_cpu = batch_idx.detach().cpu().tolist()

        # Skip particles with Z outside MatterSim's one-hot range (Z>94): the OOB
        # one-hot fires an async device assert that poisons the CUDA context and is
        # uncatchable by the try/except below. Same for non-finite cell/pos.
        import numpy as _np
        _MS_MAX_Z = int(getattr(getattr(self.potential, "model", None), "max_z", 94))
        atoms_list = []
        valid_indices = []
        for p in range(n_particles):
            slice_idx = [i for i, b in enumerate(batch_cpu) if b == p]
            if not slice_idx:
                continue
            pp = pos_cpu[slice_idx]
            zs = [argmax_z[i] for i in slice_idx]
            C = cell_cpu[p]
            if any((z < 1 or z > _MS_MAX_Z) for z in zs):
                continue
            if not _np.isfinite(C).all() or not _np.isfinite(pp).all():
                continue
            try:
                atoms = _Atoms(
                    numbers=zs, positions=pp, cell=cell_cpu[p], pbc=True,
                )
                atoms_list.append(atoms)
                valid_indices.append(p)
            except Exception:
                pass

        scores = torch.full((n_particles,), -1.0, device=pos.device, dtype=pos.dtype)
        if not atoms_list:
            return scores

        try:
            loader = build_dataloader(
                atoms=atoms_list, only_inference=True,
                batch_size=len(atoms_list), shuffle=False,
            )
            energies, _, _ = self.potential.predict_properties(
                loader, include_forces=False, include_stresses=False,
            )
        except Exception:
            return scores

        # Relative-to-batch: subtract per-call min so at least one particle scores 0,
        # preserving within-batch differentiation regardless of absolute energy level.
        per_particle_e_atom: list[tuple[int, float]] = []
        for p, e_total, atoms in zip(valid_indices, energies, atoms_list):
            n_atoms = max(1, len(atoms))
            try:
                e_atom = float(e_total) / n_atoms
            except (TypeError, ValueError):
                continue
            if not (e_atom == e_atom) or e_atom in (float("inf"), float("-inf")):
                continue
            per_particle_e_atom.append((p, e_atom))

        if not per_particle_e_atom:
            return scores

        # Maximize sign*e_atom (sign=-1 lower, +1 higher); best particle scores 0.
        us = [(p, self._sign * e_atom) for (p, e_atom) in per_particle_e_atom]
        if self.relative_to_batch:
            top = max(u for _, u in us)
        else:
            top = (self._sign * self.e_offset) if self.direction == "lower" \
                  else max(u for _, u in us)
        for p, u in us:
            v = (top - u) / max(self.e_scale, 1e-3)
            scores[p] = -max(0.0, min(1.0, v))
        return scores


class OrbV3EnergyReward:
    """Soft [-1,0] penalty on OrbV3 (MPtrj+Alex) energy/atom; faster sibling of MatterSimEnergyReward.

    Same architecture family as the ALM OrbV3 encoder (different checkpoint), so an
    ablation gain vs the MatterSim reward may partly reflect shared inductive biases.
    """

    name = "orbv3_energy"

    def __init__(
        self,
        e_offset: float = -3.0,
        e_scale: float = 2.0,
        device: str = "cuda",
        variant: str = "orb_v3_direct_20_mpa",
        relative_to_batch: bool = True,
    ):
        from orb_models.forcefield import pretrained
        loader = getattr(pretrained, variant)
        self.model = loader(device=device, precision="float32-high")
        self.model.eval()
        self.system_config = self.model.system_config
        self.e_offset = float(e_offset)
        self.e_scale = float(e_scale)
        self.device = device
        self._variant = variant
        self.relative_to_batch = bool(relative_to_batch)

    @torch.no_grad()
    def __call__(self, x_hat_0, t, t_idx):
        from ase import Atoms as _Atoms
        from orb_models.forcefield import atomic_system
        from orb_models.forcefield.base import batch_graphs

        probs = torch.softmax(x_hat_0["atomic_numbers_logits"][:, :100], dim=-1)
        pos = x_hat_0["pos"]
        cell = x_hat_0["cell"]
        batch_idx = x_hat_0["batch_idx"]
        n_particles = int(batch_idx.max().item()) + 1
        if pos is None or cell is None:
            return torch.zeros(n_particles, device=batch_idx.device, dtype=torch.float32)
        argmax_z = (probs.argmax(dim=-1) + 1).cpu().tolist()
        pos_cpu = pos.detach().cpu().numpy()
        cell_cpu = cell.detach().cpu().numpy()
        batch_cpu = batch_idx.detach().cpu().tolist()

        graphs = []
        valid_indices = []
        n_atoms_per_particle = []
        for p in range(n_particles):
            slice_idx = [i for i, b in enumerate(batch_cpu) if b == p]
            if not slice_idx:
                continue
            zs = [argmax_z[i] for i in slice_idx]
            pp = pos_cpu[slice_idx]
            try:
                atoms = _Atoms(numbers=zs, positions=pp, cell=cell_cpu[p], pbc=True)
                g = atomic_system.ase_atoms_to_atom_graphs(
                    atoms, system_config=self.system_config, device=self.device,
                )
                graphs.append(g)
                valid_indices.append(p)
                n_atoms_per_particle.append(len(slice_idx))
            except Exception:
                pass

        scores = torch.full((n_particles,), -1.0, device=pos.device, dtype=pos.dtype)
        if not graphs:
            return scores

        try:
            batch = batch_graphs(graphs)
            output = self.model.predict(batch)
            energies = output["energy"].detach().cpu().tolist()
        except Exception:
            return scores

        # Relative-to-batch shift (see MatterSimEnergyReward).
        per_particle_e_atom: list[tuple[int, float]] = []
        for p, e_total, n_p in zip(valid_indices, energies, n_atoms_per_particle):
            try:
                e_atom = float(e_total) / max(1, n_p)
            except (TypeError, ValueError):
                continue
            if not (e_atom == e_atom) or e_atom in (float("inf"), float("-inf")):
                continue
            per_particle_e_atom.append((p, e_atom))
        if not per_particle_e_atom:
            return scores
        if self.relative_to_batch:
            shift = min(e for _, e in per_particle_e_atom)
        else:
            shift = self.e_offset
        for p, e_atom in per_particle_e_atom:
            v = (e_atom - shift) / max(self.e_scale, 1e-3)
            v = max(0.0, min(1.0, v))
            scores[p] = -v
        return scores


class SpaceGroupMatchReward:
    """Per-particle 0 if detected space group matches the prompt's target_sg, else -1."""

    name = "sg_match"

    def __init__(self, target_sg: str, symprec: float = 0.1):
        self.target_sg = str(target_sg)
        self.symprec = float(symprec)

    def __call__(self, x_hat_0, t, t_idx):
        from ase import Atoms as _Atoms
        from pymatgen.io.ase import AseAtomsAdaptor
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
        import warnings as _w

        probs = torch.softmax(x_hat_0["atomic_numbers_logits"][:, :100], dim=-1)
        pos = x_hat_0["pos"]
        cell = x_hat_0["cell"]
        batch_idx = x_hat_0["batch_idx"]
        n_particles = int(batch_idx.max().item()) + 1
        if pos is None or cell is None:
            return torch.zeros(n_particles, device=batch_idx.device, dtype=torch.float32)
        argmax_z = (probs.argmax(dim=-1) + 1).cpu().tolist()
        pos_cpu = pos.detach().cpu().numpy()
        cell_cpu = cell.detach().cpu().numpy()
        batch_cpu = batch_idx.detach().cpu().tolist()

        scores = torch.full((n_particles,), -1.0, device=pos.device, dtype=pos.dtype)
        for p in range(n_particles):
            slice_idx = [i for i, b in enumerate(batch_cpu) if b == p]
            if not slice_idx:
                continue
            zs = [argmax_z[i] for i in slice_idx]
            pp = pos_cpu[slice_idx]
            try:
                atoms = _Atoms(numbers=zs, positions=pp, cell=cell_cpu[p], pbc=True)
                struct = AseAtomsAdaptor.get_structure(atoms)
                with _w.catch_warnings():
                    _w.simplefilter("ignore")
                    sg = SpacegroupAnalyzer(struct, symprec=self.symprec).get_space_group_symbol()
                if sg == self.target_sg:
                    scores[p] = 0.0
            except Exception:
                pass
        return scores


class DensityDirectionReward:
    """Geometric density-direction reward (no MLFF): "higher" rewards denser cells, "lower" more open."""

    name = "density_direction"

    def __init__(self, direction: str = "higher", scale: float = 0.5):
        assert direction in ("lower", "higher"), direction
        self.direction = direction
        self._sign = +1.0 if direction == "higher" else -1.0
        self.scale = float(scale)  # g/cm^3 span for [-1,0] normalization

    @torch.no_grad()
    def __call__(self, x_hat_0, t, t_idx):
        import numpy as _np
        from ase.data import atomic_masses
        cell = x_hat_0["cell"]
        batch_idx = x_hat_0["batch_idx"]
        n_particles = int(batch_idx.max().item()) + 1
        if cell is None:
            return torch.zeros(n_particles, device=batch_idx.device, dtype=torch.float32)
        # Prefer observed Z (CSP-mode); fall back to argmax logits in DNG-mode.
        _obs_z = x_hat_0.get("atomic_numbers")
        if _obs_z is not None:
            argmax_z = _obs_z.detach().long().reshape(-1).cpu().tolist()
        else:
            probs = torch.softmax(x_hat_0["atomic_numbers_logits"][:, :100], dim=-1)
            argmax_z = (probs.argmax(dim=-1) + 1).cpu().tolist()
        batch_cpu = batch_idx.detach().cpu().tolist()
        cell_cpu = cell.detach().cpu().numpy()
        dens: list = []
        for p in range(n_particles):
            slice_idx = [i for i, b in enumerate(batch_cpu) if b == p]
            if not slice_idx:
                dens.append(None); continue
            mass = sum(float(atomic_masses[argmax_z[i]]) for i in slice_idx)
            vol = abs(float(_np.linalg.det(cell_cpu[p])))
            d = (mass * 1.66054 / vol) if vol > 1e-6 else float("nan")
            dens.append(d if d == d else None)
        scores = torch.full((n_particles,), -1.0, device=cell.device, dtype=cell.dtype)
        us = [(p, self._sign * d) for p, d in enumerate(dens) if d is not None]
        if not us:
            return scores
        top = max(u for _, u in us)
        for p, u in us:
            v = (top - u) / max(self.scale, 1e-3)
            scores[p] = -max(0.0, min(1.0, v))
        return scores


class VolumeTargetReward:
    """Geometric reward (no MLFF) matching per-atom volume to a fixed target_vpa, in [-1, 0].

    Strain task: a signed direction reward would overshoot the gate's target magnitude,
    so steer toward the recorded target_vpa instead.
    """

    name = "volume_target"

    def __init__(self, target_vpa: float, scale: float = 5.0):
        self.target_vpa = float(target_vpa)
        self.scale = float(scale)  # A^3/atom span for [-1,0] normalization

    @torch.no_grad()
    def __call__(self, x_hat_0, t, t_idx):
        import numpy as _np
        cell = x_hat_0["cell"]
        batch_idx = x_hat_0["batch_idx"]
        n_particles = int(batch_idx.max().item()) + 1
        if cell is None:
            return torch.zeros(n_particles, device=batch_idx.device, dtype=torch.float32)
        batch_cpu = batch_idx.detach().cpu().tolist()
        cell_cpu = cell.detach().cpu().numpy()
        scores = torch.full((n_particles,), -1.0, device=cell.device, dtype=cell.dtype)
        for p in range(n_particles):
            n_p = sum(1 for b in batch_cpu if b == p)
            if n_p == 0:
                continue
            vol = abs(float(_np.linalg.det(cell_cpu[p])))
            if not (vol > 1e-6):
                continue  # degenerate cell -> full penalty
            vpa = vol / n_p
            v = abs(vpa - self.target_vpa) / max(self.scale, 1e-3)
            scores[p] = -max(0.0, min(1.0, v))
        return scores


REGISTRY: dict[str, type[Reward]] = {
    "stoich_match":        StoichiometricMatchReward,
    "count_l1":            StoichCountL1Reward,
    "ratio_kl":            StoichRatioKLReward,
    "in_set_completeness": InSetCompletenessReward,
    "physical_sanity":     PhysicalSanityReward,
    "mattersim_energy":    MatterSimEnergyReward,
    "orbv3_energy":        OrbV3EnergyReward,
    "density_direction":   DensityDirectionReward,
    "volume_target":       VolumeTargetReward,
    "sg_match":            SpaceGroupMatchReward,
}


def parse_rewards(
    spec: str,
    *,
    allowed_elements: list[str] | None = None,
    target_counts: Mapping[int, int] | None = None,
    physical_bounds: Mapping[str, float] | None = None,
    target_sg: str | None = None,
    direction: str = "lower",
    target_vpa: float | None = None,
) -> WeightedSum:
    """Parse a `name:weight;name:weight` reward spec into a WeightedSum."""
    out: list[tuple[Reward, float]] = []
    for entry in spec.split(";"):
        if not entry.strip():
            continue
        name, _, w = entry.partition(":")
        weight = float(w) if w else 1.0
        if name not in REGISTRY:
            raise ValueError(
                f"Unknown reward {name!r}; have {list(REGISTRY)}"
            )
        cls = REGISTRY[name]
        if name in ("stoich_match", "count_l1", "ratio_kl"):
            if not target_counts:
                raise ValueError(
                    f"{name} needs --fk_target_counts (Z:n,…) "
                    "OR --fk_target_counts_from_prompt_json <path:tag>"
                )
            out.append((cls(target_counts), weight))
        elif name == "in_set_completeness":
            if not allowed_elements:
                raise ValueError(
                    "in_set_completeness needs --allowed_elements"
                )
            out.append((cls(allowed_elements), weight))
        elif name == "physical_sanity":
            if not physical_bounds:
                raise ValueError(
                    "physical_sanity needs --fk_physical_bounds_path "
                    "(JSON written by scripts/calibrate_physical_priors.py)"
                )
            out.append((cls(physical_bounds), weight))
        elif name == "mattersim_energy":
            out.append((cls(direction=direction), weight))
        elif name == "orbv3_energy":
            out.append((cls(), weight))
        elif name == "density_direction":
            out.append((cls(direction=direction), weight))
        elif name == "volume_target":
            if target_vpa is None:
                raise ValueError(
                    "volume_target needs a target_vpa (target per-atom volume, "
                    "Å³/atom — the caller computes it from the row's input volume "
                    "and recorded dV%)"
                )
            out.append((cls(target_vpa=target_vpa), weight))
        elif name == "sg_match":
            if not target_sg:
                raise ValueError("sg_match needs a target_sg")
            out.append((cls(target_sg), weight))
    if not out:
        raise ValueError(f"No rewards parsed from spec={spec!r}")
    return WeightedSum(out)
