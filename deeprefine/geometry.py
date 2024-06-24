'''
Geometry related codes 
The SO(3) grid part codes are adpated from CryoDrgn (https://github.com/zhonge/cryodrgn) based on Hopf Fibration
All codes are tensorflow-ified
'''
import torch
import numpy as np
import healpy as hp
from deeprefine.utils import assert_tensor, assert_numpy

# Two vector representation of SO3
def construct_SO3(v1, v2):
    '''
    Construct a continuous representation of SO(3) ratation with two 3D vectors
    https://arxiv.org/abs/1812.07035
    Parameters
    ----------
    v1, v2: 3D tensors
        Real-valued tensor in 3D space
    Returns
    -------
    R, A 3*3 SO(3) rotation matrix
    '''
    e1 = v1 / torch.norm(v1)
    u2 = v2 - e1 * torch.tensordot(e1, v2, dims=1)
    e2 = u2 / torch.norm(u2)
    e3 = torch.cross(e1, e2)
    R = torch.stack((e1, e2, e3)).T
    return R


def decompose_SO3(R, a=1, b=1, c=1):
    '''
    Decompose the rotation matrix into the two vector representation
    This decomposition is not unique, so a, b, c can be set as arbitray constants you like
    C != 0
    Parameters
    ----------
    R: 3*3 tensors
        Real-valued rotation matrix
    Returns
    -------
    v1, v2: Two real-valued 3D tensors, as the continuous representation of the rotation matrix 
    '''
    assert c != 0, "Give a nonzero c!"
    v1 = a*R[:, 0]
    v2 = b*R[:, 0] + c*R[:, 1]

    return v1, v2

# Quaternion representation of SO(3) and Hopf Fibration grid
def grid_s1(resol=1):
    Npix = 6*2**resol
    dt = 2*np.pi/Npix
    grid = np.arange(Npix)*dt + dt/2
    return grid


def grid_s2(resol=1):  
    Nside = 2**resol
    Npix = 12*Nside*Nside
    theta, phi = hp.pix2ang(Nside, np.arange(Npix), nest=True)
    return theta, phi


def hopf_to_quat(theta, phi, psi) -> np.ndarray:
    '''
    Hopf coordinates to quaternions
    theta: [0,pi]
    phi: [0, 2pi)
    psi: [0, 2pi)
    already normalized
    '''
    ct = np.cos(theta/2)
    st = np.sin(theta/2)
    quat = np.array([ct*np.cos(psi/2),
                     ct*np.sin(psi/2),
                     st*np.cos(phi+psi/2),
                     st*np.sin(phi+psi/2)])
    return quat.T.astype(np.float32)


def grid_SO3(resol) -> np.ndarray:
    theta, phi = grid_s2(resol)
    psi = grid_s1(resol)
    quat = hopf_to_quat(np.repeat(theta, len(psi)),  # repeats each element by len(psi)
                        # repeats each element by len(psi)
                        np.repeat(phi, len(psi)),
                        np.tile(psi, len(theta)))  # tiles the array len(theta) times
    return quat


def quat_distance(q1, q2):
    """
    q1: [n1, 4]
    q2: [n2, 4]
    
    Return:
        [n1, n2]
    """
    q1 = q1 / np.linalg.norm(q1, ord=2, axis=-1, keepdims=True)
    q2 = q2 / np.linalg.norm(q2, ord=2, axis=-1, keepdims=True)
    args = np.abs(np.sum(q1[:, None, :]*q2[None,...], axis=-1))
    return 2.*np.arccos(args)


def mat_distance(R1, R2):
    """
    R1: [a, 3, 3]
    R2: [b, 3, 3]

    Return:
        [a, b]
    """
    R = np.einsum("axy,bzy->abxz", R1, R2)
    args = (np.einsum("abii", R) - 1) / 2.0
    return np.arccos(args)


def quaternions_to_SO3(q) -> torch.Tensor:
    """
    Normalizes q and maps to group matrix.
    https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation#Quaternion-derived_rotation_matrix
    https://danceswithcode.net/engineeringnotes/quaternions/quaternions.html
    """
    q = assert_tensor(q, torch.float32)
    q = q / q.norm(p=2, dim=-1, keepdim=True)
    r, i, j, k = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    return torch.stack(
        [
            1 - 2 * j * j - 2 * k * k,
            2 * (i * j - r * k),
            2 * (i * k + r * j),
            2 * (i * j + r * k),
            1 - 2 * i * i - 2 * k * k,
            2 * (j * k - r * i),
            2 * (i * k - r * j),
            2 * (j * k + r * i),
            1 - 2 * i * i - 2 * j * j,
        ],
        -1,
    ).view(*q.shape[:-1], 3, 3)


def SO3_to_quaternions(mats) -> np.array:
    """
    mat: [n, 3, 3]

    Return: [n, 4]
    """
    mats = assert_numpy(mats)
    w, v = np.linalg.eig(mats)
    ind = np.argwhere(np.isclose(w, 1.0))
    us = np.real(v[ind[:,0], :, ind[:, 1]])
    
    # take a random vector v orthogonal to u, determine sign of sin(theta/2)
    # https://math.stackexchange.com/questions/893984/conversion-of-rotation-matrix-to-quaternion
    v_rand = np.random.rand(3)
    vs = np.cross(us, v_rand)
    signs = np.sign(np.einsum("nx,nx->n", np.cross(vs, np.einsum("nxy,ny->nx", mats, vs)), us))

    args = (np.einsum("nii", mats) - 1)/2
    thetas = np.arccos(args)
    cos_halftheta = np.cos(thetas/2.0)
    sin_halftheta = np.sin(thetas/2.0) * signs
    qs = np.concatenate([cos_halftheta[:, None], sin_halftheta[:, None] * us], axis=1)
    qs = qs / np.linalg.norm(qs, axis=-1, keepdims=True)
    return qs


# Neighbors on the Hopf grid
def get_s1_neighbor(mini, curr_res):
    '''
    Return the 2 nearest neighbors on S1 at the next resolution level
    '''
    Npix = 6*2**(curr_res+1)
    dt = 2*np.pi/Npix
    # return np.array([2*mini, 2*mini+1])*dt + dt/2
    # the fiber bundle grid on SO3 is weird
    # the next resolution level's nearest neighbors in SO3 are not
    # necessarily the nearest neighbor grid points in S1
    # include the 13 neighbors for now... eventually learn/memoize the mapping
    ind = np.arange(2*mini-1, 2*mini+3)
    if ind[0] < 0:
        ind[0] += Npix
    return ind*dt+dt/2, ind


def get_s2_neighbor(mini, curr_res):
    '''
    Return the 4 nearest neighbors on S2 at the next resolution level
    '''
    Nside = 2**(curr_res+1)
    ind = np.arange(4)+4*mini
    return hp.pix2ang(Nside, ind, nest=True), ind


def get_base_ind(ind, base_resol=1):
    '''
    Return the corresponding S2 and S1 grid index for an index on the base SO3 grid
    '''
    psii = ind % (6*2**base_resol)
    thetai = ind // (6*2**base_resol)
    return thetai, psii


def get_neighbor_SO3(quat, s2i, s1i, curr_res):
    '''
    Return the 8 nearest neighbors on SO3 at the next resolution level
    '''
    (theta, phi), s2_nexti = get_s2_neighbor(s2i, curr_res)
    psi, s1_nexti = get_s1_neighbor(s1i, curr_res)
    quat_n = hopf_to_quat(np.repeat(theta, len(psi)),
                          np.repeat(phi, len(psi)),
                          np.tile(psi, len(theta)))
    ind = np.array([np.repeat(s2_nexti, len(psi)),
                    np.tile(s1_nexti, len(theta))])
    ind = ind.T
    # find the 8 nearest neighbors of 16 possible points
    # need to check distance from both +q and -q
    dists = np.minimum(np.sum((quat_n-quat)**2, axis=1),
                       np.sum((quat_n+quat)**2, axis=1))
    ii = np.argsort(dists)[:8]
    return quat_n[ii], ind[ii]

# Loss based neighbor search
def getbestneighbors_base_SO3(loss, base_quats, N=10, base_resol=1):
    sort_index = np.argsort(loss)
    bestN_index = sort_index[:N]
    best_quats = base_quats[bestN_index]
    s2_index, s1_index = get_base_ind(bestN_index, base_resol)
    allnb_quats = np.array([]).reshape(0, 4)
    allnb_s2s1 = np.array([]).reshape(0, 2)
    for i in range(N):
        nb_quats_i, nb_s2_s1_i = get_neighbor_SO3(
            best_quats[i], s2_index[i], s1_index[i], base_resol)
        allnb_quats = np.concatenate((allnb_quats, nb_quats_i), axis=0)
        allnb_s2s1 = np.concatenate((allnb_s2s1, nb_s2_s1_i), axis=0)
    return allnb_quats, allnb_s2s1


def getbestneighbors_next_SO3(loss, quats, s2s1_arr, curr_res=2, N=50):
    sort_index = np.argsort(loss)
    bestN_index = sort_index[:N]
    best_quats = quats[bestN_index]
    s2_index = s2s1_arr[bestN_index, 0].astype(int)
    s1_index = s2s1_arr[bestN_index, 1].astype(int)
    allnb_quats = np.array([]).reshape(0, 4)
    allnb_s2s1 = np.array([]).reshape(0, 2)
    for i in range(N):
        nb_quats_i, nb_s2_s1_i = get_neighbor_SO3(
            best_quats[i], s2_index[i], s1_index[i], curr_res=curr_res)
        allnb_quats = np.concatenate((allnb_quats, nb_quats_i), axis=0)
        allnb_s2s1 = np.concatenate((allnb_s2s1, nb_s2_s1_i), axis=0)
    return allnb_quats, allnb_s2s1

# TODO: support non-uniform basegrid
def getbestneighbours_cartesian(loss, current_uvw_array_frac, basegrid=24.0, 
                                asu_brick_lim = [1.0, 1.0, 1.0],
                                curr_res=1, N=40, drop_duplicates=True, polar_axis=None):
    sort_index = np.argsort(loss)
    bestN_index = sort_index[:N]
    current_best_uvw_frac = current_uvw_array_frac[bestN_index]
    
    
    xlim, ylim, zlim = asu_brick_lim
    scale = (1./(basegrid-1.))/2**(curr_res)
    du_list = np.linspace(-1,1,3) * xlim
    dv_list = np.linspace(-1,1,3) * ylim 
    dw_list = np.linspace(-1,1,3) * zlim
    if polar_axis is not None:
        if 0 in polar_axis:
            du_list = np.array([0.0])
        if 1 in polar_axis:
            dv_list = np.array([0.0])
        if 2 in polar_axis:
            dw_list = np.array([0.0])

    duvw_array_frac = scale * np.array(np.meshgrid(du_list, dv_list, dw_list)).T.reshape(-1,3)
    
    nb_uvw_frac = current_best_uvw_frac[:, None, :] + duvw_array_frac[None, ...] 
    nb_uvw_frac = nb_uvw_frac % 1.0
    nb_uvw_frac = np.round(nb_uvw_frac.reshape(-1,3), 4)

    if drop_duplicates:
        return np.unique(nb_uvw_frac, axis=0)
    else:
        return nb_uvw_frac