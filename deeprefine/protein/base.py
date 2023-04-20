import numpy as np
import torch

def dist(x1, x2):
    d = x2-x1
    d2 = np.sum(d*d, axis=-1)
    return np.sqrt(d2)

def dist_torch(x1, x2):
    d = x2-x1
    d2 = torch.sum(d*d, axis=-1)
    return torch.sqrt(d2)

def angle(x1, x2, x3, degrees=True):
    ba = x1 - x2
    ba /= np.linalg.norm(ba, axis=-1, keepdims=True)
    bc = x3 - x2
    bc /= np.linalg.norm(bc, axis=-1, keepdims=True)
    cosine_angle = np.sum(ba*bc, axis=-1)
    if degrees:
        angle = np.degrees(np.arccos(cosine_angle)) # Range [0,180]
        return angle 
    else:  # Range [0, pi]
        return np.arccos(cosine_angle) 

def angle_torch(x1, x2, x3, degrees=True):
    ba = x1 - x2
    ba /= torch.norm(ba, dim=-1, keepdims=True)
    bc = x3 - x2
    bc /= torch.norm(bc, axis=-1, keepdims=True)
    cosine_angle = torch.sum(ba*bc, axis=-1)
    if degrees:
        angle = np.float32(180.0 / np.pi) * torch.acos(cosine_angle) # Range [0,180]
        return angle
    else: # Range [0, pi]
        return torch.acos(cosine_angle) 

# See Ref: https://stackoverflow.com/questions/20305272/dihedral-torsion-angle-from-four-points-in-cartesian-coordinates-in-python 
def torsion(x1, x2, x3, x4):
    """Praxeolitic formula
    1 sqrt, 1 cross product"""
    b0 = -1.0*(x2 - x1)
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
    v = b0 - np.sum(b0*b1, axis=-1, keepdims=True) * b1
    w = b2 - np.sum(b2*b1, axis=-1, keepdims=True) * b1

    # angle between v and w in a plane is the torsion angle
    # v and w may not be normalized but that's fine since tan is y/x
    x = np.sum(v*w, axis=-1)
    b1xv = np.cross(b1, v, axisa=-1, axisb=-1)
    y = np.sum(b1xv*w, axis=-1)
    return np.degrees(np.arctan2(y, x))

def torsion_torch(x1, x2, x3, x4, degrees=True):
    """Praxeolitic formula
    1 sqrt, 1 cross product"""
    b0 = -1.0*(x2 - x1)
    b1 = x3 - x2
    b2 = x4 - x3
    # normalize b1 so that it does not influence magnitude of vector
    # rejections that come next
    b1 /= torch.norm(b1, dim=-1, keepdims=True)

    # vector rejections
    # v = projection of b0 onto plane perpendicular to b1
    #   = b0 minus component that aligns with b1
    # w = projection of b2 onto plane perpendicular to b1
    #   = b2 minus component that aligns with b1
    v = b0 - torch.sum(b0*b1, axis=-1, keepdims=True) * b1
    w = b2 - torch.sum(b2*b1, axis=-1, keepdims=True) * b1

    # angle between v and w in a plane is the torsion angle
    # v and w may not be normalized but that's fine since tan is y/x
    x = torch.sum(v*w, axis=-1)
    b1xv = torch.cross(b1, v)
    y = torch.sum(b1xv*w, axis=-1)
    if degrees:
        return np.float32(180.0 / np.pi) * torch.atan2(y, x)
    else:
        return torch.atan2(y, x)


