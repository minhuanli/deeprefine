import numpy as np
import torch


def dist(x1, x2):
    d = x2 - x1
    d2 = np.sum(d * d, axis=-1)
    return np.sqrt(d2)


def dist_torch(x1, x2):
    d = x2 - x1
    d2 = torch.sum(d * d, dim=-1)
    return torch.sqrt(d2)


def angle(x1, x2, x3, degrees=False):
    ba = x1 - x2
    ba /= np.linalg.norm(ba, axis=-1, keepdims=True)
    bc = x3 - x2
    bc /= np.linalg.norm(bc, axis=-1, keepdims=True)
    cosine_angle = np.sum(ba * bc, axis=-1)
    if degrees:
        angle = np.degrees(np.arccos(cosine_angle))  # Range [0,180]
        return angle
    else:  # Range [0, pi]
        return np.arccos(cosine_angle)


def angle_torch(x1, x2, x3, degrees=False):
    ba = x1 - x2
    ba = ba / torch.norm(ba, dim=-1, keepdim=True)
    bc = x3 - x2
    bc = bc / torch.norm(bc, dim=-1, keepdim=True)
    cosine_angle = torch.sum(ba * bc, dim=-1)
    if degrees:
        angle = np.float32(180.0 / np.pi) * torch.acos(cosine_angle)  # Range [0,180]
        return angle
    else:  # Range [0, pi]
        return torch.acos(cosine_angle)


# See Ref: https://stackoverflow.com/questions/20305272/dihedral-torsion-angle-from-four-points-in-cartesian-coordinates-in-python
def torsion(x1, x2, x3, x4, degrees=False):
    """Praxeolitic formula
    1 sqrt, 1 cross product"""
    b0 = -1.0 * (x2 - x1)
    b1 = x3 - x2
    b2 = x4 - x3
    # normalize b1 so that it does not influence magnitude of vector
    # rejections that come next
    b1 /= np.linalg.norm(b1, axis=-1, keepdims=True)

    # vector rejections
    # v = projection of b0 onto plane perpendicular to b1
    #   = b0 minus component that aligns with b1
    # w = projection of b2 onto plane perpendicular to b1
    #   = b2 minus component that aligns with b1
    v = b0 - np.sum(b0 * b1, axis=-1, keepdims=True) * b1
    w = b2 - np.sum(b2 * b1, axis=-1, keepdims=True) * b1

    # angle between v and w in a plane is the torsion angle
    # v and w may not be normalized but that's fine since tan is y/x
    x = np.sum(v * w, axis=-1)
    b1xv = np.cross(b1, v, axisa=-1, axisb=-1)
    y = np.sum(b1xv * w, axis=-1)
    if degrees:
        return np.degrees(np.arctan2(y, x))
    else:
        return np.arctan2(y, x)


def torsion_torch(x1, x2, x3, x4, degrees=False):
    """Praxeolitic formula
    1 sqrt, 1 cross product"""
    b0 = -1.0 * (x2 - x1)
    b1 = x3 - x2
    b2 = x4 - x3
    # normalize b1 so that it does not influence magnitude of vector
    # rejections that come next
    b1 = b1 / torch.norm(b1, dim=-1, keepdim=True)

    # vector rejections
    # v = projection of b0 onto plane perpendicular to b1
    #   = b0 minus component that aligns with b1
    # w = projection of b2 onto plane perpendicular to b1
    #   = b2 minus component that aligns with b1
    v = b0 - torch.sum(b0 * b1, dim=-1, keepdim=True) * b1
    w = b2 - torch.sum(b2 * b1, dim=-1, keepdim=True) * b1

    # angle between v and w in a plane is the torsion angle
    # v and w may not be normalized but that's fine since tan is y/x
    x = torch.sum(v * w, dim=-1)
    b1xv = torch.linalg.cross(b1, v)
    y = torch.sum(b1xv * w, dim=-1)
    if degrees:
        return np.float32(180.0 / np.pi) * torch.atan2(y, x)
    else:
        return torch.atan2(y, x)


def xyz2ic_torch(x, Z_indices, vec_angles=False):
    """Computes internal coordinates from Cartesian coordinates

    Parameters
    ----------
    x : Tensor, [n_batch, n_atoms*n_dim]
        Catesian coordinates of all atoms
    Z_indices : array, [n_icatoms, 4]
        Internal coordinate index definition.
    vec_angles : binary, default False
        whether or not to use cos and sin to vectorize the torsion angle

    Returns
    -------
    Tensor, [n_batch, n_icatoms]
    """
    bond_indices = Z_indices[:, :2]
    angle_indices = Z_indices[:, :3]
    torsion_indices = Z_indices[:, :4]
    atom_indices = np.arange(int(3 * (np.max(Z_indices) + 1))).reshape((-1, 3))
    xbonds = dist_torch(
        x[:, atom_indices[bond_indices[:, 0]]], x[:, atom_indices[bond_indices[:, 1]]]
    )
    xangles = angle_torch(
        x[:, atom_indices[angle_indices[:, 0]]],
        x[:, atom_indices[angle_indices[:, 1]]],
        x[:, atom_indices[angle_indices[:, 2]]],
    )
    xtorsions = torsion_torch(
        x[:, atom_indices[torsion_indices[:, 0]]],
        x[:, atom_indices[torsion_indices[:, 1]]],
        x[:, atom_indices[torsion_indices[:, 2]]],
        x[:, atom_indices[torsion_indices[:, 3]]],
    )
    if vec_angles:
        xangles_vec = torch.stack([torch.sin(xangles), torch.cos(xangles)], dim=-1)
        xtorsions_vec = torch.stack(
            [torch.sin(xtorsions), torch.cos(xtorsions)], dim=-1
        )
        ics = torch.concat(
            [xbonds[..., None], xangles_vec, xtorsions_vec], dim=-1
        ).reshape(xbonds.shape[0], -1)
    else:
        ics = torch.stack((xbonds, xangles, xtorsions), dim=-1).reshape(
            xbonds.shape[0], -1
        )
    return ics


def ic2xyz_torch(p1, p2, p3, d14, a412, t4123):
    """Compute Cartesian coordinates from internal coordinates

    Parameters
    ----------
    p1 : Tensor, [n_points, 3] or [n_batch, n_points, 3]
        Cartesian coordinates of reference point 1
    p2 : Tensor, [n_points, 3] or [n_batch, n_points, 3]
        Cartesian coordinates of reference point 2
    p3 : Tensor, [n_points, 3] or [n_batch, n_points, 3]
        Cartesian coordinates of refernece point 3
    d14 : Tensor, [n_points, 1] or [n_batch, n_points, 1]
        Bond length between target point 4 and reference point 1
    a412 : Tensor, [n_points, 1] or [n_batch, n_points, 1]
        Bond angle between bond target point 4 - reference point 1 and bond reference point 2 - reference point 1
    t4123 : Tensor, [n_points, 1] or [n_batch, n_points, 1]
        Torsion angle between bond target point 4 - reference point 1 and bond reference point 2 - reference point 3

    Returns
    -------
    Tensor, [n_points, 3] or [n_batch, n_points, 3]
        Cartesian Coordinates of target points
    """
    v1 = p1 - p2
    v2 = p1 - p3

    n = torch.linalg.cross(v1, v2)
    nn = torch.linalg.cross(v1, n)
    n = n / torch.norm(n, dim=-1, keepdim=True)
    nn = nn / torch.norm(nn, dim=-1, keepdim=True)

    n = n * (-torch.sin(t4123))
    nn = nn * torch.cos(t4123)

    v3 = n + nn
    v3 = v3 / torch.norm(v3, dim=-1, keepdim=True)
    v3 = v3 * d14 * torch.sin(a412)

    v1 = v1 / torch.norm(v1, dim=-1, keepdim=True)
    v1 = v1 * d14 * torch.cos(a412)

    position = p1 + v3 - v1

    return position


def ic2xyz_nerf_torch(p1, p2, p3, d14, a412, t4123):
    """Compute Cartesian coordinates from internal coordinates, with NeRF algorithm
    Parsons, Jerod, et al. Journal of computational chemistry 26.10 (2005): 1063-1068.

    Parameters
    ----------
    p1 : Tensor, [n_points, 3] or [n_batch, n_points, 3]
        Cartesian coordinates of reference point 1
    p2 : Tensor, [n_points, 3] or [n_batch, n_points, 3]
        Cartesian coordinates of reference point 2
    p3 : Tensor, [n_points, 3] or [n_batch, n_points, 3]
        Cartesian coordinates of refernece point 3
    d14 : Tensor, [n_points, 1] or [n_batch, n_points, 1]
        Bond length between target point 4 and reference point 1
    a412 : Tensor, [n_points, 1] or [n_batch, n_points, 1]
        Bond angle between bond target point 4 - reference point 1 and bond reference point 2 - reference point 1
    t4123 : Tensor, [n_points, 1] or [n_batch, n_points, 1]
        Torsion angle between bond target point 4 - reference point 1 and bond reference point 2 - reference point 3

    Returns
    -------
    Tensor, [n_points, 3] or [n_batch, n_points, 3]
        Cartesian Coordinates of target points
    """

    a412 = np.pi - a412
    BC = p1 - p2
    AB = p2 - p3

    bc = BC / torch.norm(BC, dim=-1, keepdim=True)
    n = torch.linalg.cross(AB, bc)
    n = n / torch.norm(n, dim=-1, keepdim=True)

    D2 = torch.concat(
        [
            d14 * torch.cos(a412),
            d14 * torch.sin(a412) * torch.cos(t4123),
            d14 * torch.sin(a412) * torch.sin(t4123),
        ],
        dim=-1,
    )  # [..., 3]
    M = torch.stack([bc, torch.linalg.cross(n, bc), n], dim=-1)  # [..., 3, 3]

    position = torch.einsum("...ij,...j->...i", M, D2) + p1
    return position


def decompose_Z_indices(cart_indices, Z_indices):
    """Decompose the atoms into groups, for a hierarchical placement later"""
    known_indices = cart_indices
    Z_placed = np.zeros(Z_indices.shape[0])
    Z_indices_decomposed = []
    while np.count_nonzero(Z_placed) < Z_indices.shape[0]:
        Z_indices_cur = []
        for i in range(Z_indices.shape[0]):
            if not Z_placed[i] and np.all(
                [Z_indices[i, j] in known_indices for j in range(1, 4)]
            ):
                Z_indices_cur.append(Z_indices[i])
                Z_placed[i] = 1
        Z_indices_cur = np.array(Z_indices_cur)
        known_indices = np.concatenate([known_indices, Z_indices_cur[:, 0]])
        Z_indices_decomposed.append(Z_indices_cur)

    return Z_indices_decomposed
