"""
Convert between cartesian coordinates and Internal Coordinates of protein
"""
import torch
import numpy as np

from deeprefine.protein.base import (
    xyz2ic_torch,
    ic2xyz_torch,
    ic2xyz_nerf_torch,
    decompose_Z_indices,
)
from deeprefine.protein.zmatrix import get_indices


def ics2xyz_local(ics, Z_indices, index2zorder, xyz, nerf=False):
    """For systems described in both internal coordinates and cartesian coordinates: convert ic to Cartesian

    Parameters
    ----------
    ics : Tenosr, [n_batches, n_atoms_to_place, 3]
        IC matrix by atom to place [[[bond1, angle1, torsion1], [bond2, angle2, torsion2], ...] ...]

    Z_indices : array, [n_atoms_to_place, 4]

    index2order: array, [n_atoms,]

    xyz : Tensor, [n_batch, n_refernce_atoms, 3]
        cartesian coordinates of reference atoms

    nerf : binary, default False

    """
    batchsize = ics.shape[0]
    natoms_to_place = Z_indices.shape[0]

    # get reference atoms coordinates
    p1s = xyz[:, index2zorder[Z_indices[:, 1]], :].view(batchsize*natoms_to_place, 3)
    p2s = xyz[:, index2zorder[Z_indices[:, 2]], :].view(batchsize*natoms_to_place, 3)
    p3s = xyz[:, index2zorder[Z_indices[:, 3]], :].view(batchsize*natoms_to_place, 3)

    ics_ = ics.reshape(batchsize*natoms_to_place, 3)
    if nerf:
        newpos = ic2xyz_nerf_torch(
            p1s, p2s, p3s, ics_[..., 0:1], ics_[..., 1:2], ics_[..., 2:3]
        )
    else:
        newpos = ic2xyz_torch(
            p1s, p2s, p3s, ics_[..., 0:1], ics_[..., 1:2], ics_[..., 2:3]
        )

    return newpos.view(batchsize, natoms_to_place, 3)


class ICConverter(object):
    """Conversion between Cartesian coordinates and Cartesian / internal coordinates"""

    def __init__(self, topology, vec_angles=True):
        """
        Parameters
        ----------
        topology: mdtraj.topology
            Topology of the protein

        vec_angles: binary, default True
            Whether to use vector representation of all angles
        """
        self.topology = topology
        self.cart_atom_indices, self.Z_indices_no_order = get_indices(self.topology)
        self.vec_angles = vec_angles

        self.cart_signal_indices = np.concatenate(
            [[i * 3, i * 3 + 1, i * 3 + 2] for i in self.cart_atom_indices]
        )
        self.dim_cart_signal = self.cart_signal_indices.size

        self.batchwise_Z_indices = decompose_Z_indices(
            self.cart_atom_indices, self.Z_indices_no_order
        )
        self.Z_indices = np.vstack(self.batchwise_Z_indices)

        if self.vec_angles:
            self.dim = 3 * self.cart_atom_indices.size + 5 * self.Z_indices.shape[0]
            n_internal = self.dim - self.dim_cart_signal
            self.bond_idxs = self.dim_cart_signal + np.arange(n_internal // 5) * 5 + 0
            self.sinangle_idxs = (
                self.dim_cart_signal + np.arange(n_internal // 5) * 5 + 1
            )
            self.cosangle_idxs = (
                self.dim_cart_signal + np.arange(n_internal // 5) * 5 + 2
            )
            self.sintorsion_idxs = (
                self.dim_cart_signal + np.arange(n_internal // 5) * 5 + 3
            )
            self.costorsion_idxs = (
                self.dim_cart_signal + np.arange(n_internal // 5) * 5 + 4
            )
        else:
            self.dim = 3 * (self.cart_atom_indices.size + self.Z_indices.shape[0])
            n_internal = self.dim - self.dim_cart_signal
            self.bond_idxs = self.dim_cart_signal + np.arange(n_internal // 3) * 3 + 0
            self.angle_idxs = self.dim_cart_signal + np.arange(n_internal // 3) * 3 + 1
            self.torsion_idxs = (
                self.dim_cart_signal + np.arange(n_internal // 3) * 3 + 2
            )

        self.atom_order = np.concatenate([self.cart_atom_indices, self.Z_indices[:, 0]])
        self.index2order = np.argsort(self.atom_order)

    @classmethod
    def from_dict(cls, d):
        CT = cls(d["topology"], vec_angles=d["vec_angles"])
        return CT

    def to_dict(self):
        d = {}
        d["topology"] = self.topology
        d["vec_angles"] = self.vec_angles
        return d

    def xyz2ic(self, x):
        """Convert cartesian coordinates to internal coordinates

        Parameters
        ----------
        x : Tensor, [n_batch, n_atoms * 3]
            flattened cartesian coordinates of full protein atoms
            The order of atoms is the same as the topology

        Returns
        -------
        z : Tensor, [n_batch, n_features]
            flattened internal coordinates of full protein atoms
            The order of atoms is the same as the grouped Z_indices
        """
        # split off Cartesian coordinates of root atoms
        x_cart = x[:, self.cart_signal_indices]

        # Compute internal coordinates for remaining atoms
        z_ics = xyz2ic_torch(x, self.Z_indices, vec_angles=self.vec_angles)

        # concatenate the output
        z = torch.concat([x_cart, z_ics], dim=1)
        return z

    def ic2xyz(self, z, nerf=False):
        """Convert internal coordinates to cartesian coordinates

        Parameters
        ----------
        z : Tensor, [n_batch, n_features]
            flattened internal coordinates of full protein atoms
            The order of atoms is the same as the grouped Z_indices

        nerf : binary, default False
            Whether or not to use nerf algorithm

        Returns
        -------
        x : Tensor, [n_batch, n_atoms * 3]
            flattened cartesian coordinates of full protein atoms
            The order of atoms is the same as the topology
        """

        # Get the cartesian of root atoms
        x_cart = z[:, : self.dim_cart_signal]
        # split by atom
        batchsize = z.shape[0]
        xyz = x_cart.view(batchsize, self.cart_atom_indices.size, 3)

        if self.vec_angles:
            bonds = z[:, self.bond_idxs]

            cosangle = z[:, self.cosangle_idxs]
            sinangle = z[:, self.sinangle_idxs]
            angles = torch.atan2(sinangle, cosangle)

            costorsion = z[:, self.costorsion_idxs]
            sintorsion = z[:, self.sintorsion_idxs]
            torsions = torch.atan2(sintorsion, costorsion)
        else:
            bonds = z[:, self.bond_idxs]
            angles = z[:, self.angle_idxs]
            torsions = z[:, self.torsion_idxs]

        z_ics = torch.stack([bonds, angles, torsions], dim=-1).view(bonds.shape[0], -1)

        istart = 0
        for Z_indices in self.batchwise_Z_indices:
            ics = z_ics[:, 3*istart : 3*(istart + Z_indices.shape[0])]
            newpos = ics2xyz_local(ics, Z_indices, self.index2order, xyz, nerf=nerf)
            xyz = torch.concat([xyz, newpos], dim=1)
            istart += Z_indices.shape[0]

        # reorganize all atom coordinates to topology order
        x = xyz[:, self.index2order, :].view(batchsize, -1)
        return x
