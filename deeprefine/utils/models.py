import numpy as np
import torch
import deeprefine as dr


def plddt2pseudoB(plddt):
    '''
    Convert PLDDT to pseudo Bfactors, using equations from Beak et al. Science. 2021 Aug 20; 373(6557): 871–876.

    Args:
        plddt (torch.Tensor | np.ndarray): The tensor/array of plddt value

    Returns:
        torch.Tensor | np.ndarray : The converted pseudo B factors 
    '''
    if isinstance(plddt, torch.Tensor):
        rmsd = 1.5 * torch.exp(4. * (0.7 - 0.01 * plddt))
        pseudoB = 8.*np.pi**2 * rmsd**2 / 3.
    elif isinstance(plddt, np.ndarray):
        rmsd = 1.5 * np.exp(4. * (0.7 - 0.01 * plddt))
        pseudoB = 8.*np.pi**2 * rmsd**2 / 3.
    else:
        raise ValueError("plddt should be torch.Tensor or np.ndarray!")
    return pseudoB


def rmsd(xyz1, xyz2):
    return torch.square(xyz1 - xyz2).sum(dim=-1).mean(dim=0).sqrt()


def rbr_quat_lbfgs(xyz, dcp, lr=150.0, n_steps=15, loss_track=[], solvent=False, verbose=True):
    def closure():
        optimizer.zero_grad()
        temp_R = dr.geometry.quaternions_to_SO3(q)
        temp_model = torch.matmul(propose_rmcom.clone().detach(), temp_R) + propose_com.clone().detach() + trans_vec
        dcp.calc_fprotein(atoms_position_tensor=temp_model)
        if solvent:
            dcp.calc_fsolvent()
        Fmodel = dcp.calc_ftotal()
        Fmodel_mag = torch.abs(Fmodel)
        loss = torch.sum((dcp.Fo[working_set] - Fmodel_mag[working_set])**2/dcp.SigF[working_set]**2)
        loss.backward()
        return loss
    
    xyz = xyz.to(dcp.device)
    propose_rmcom = xyz - torch.mean(xyz, dim=0)
    propose_com = torch.mean(xyz, dim=0)
    q = torch.tensor(
        [1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=dcp.device, requires_grad=True
    )
    trans_vec = torch.tensor([0.0, 0.0, 0.0], device=dcp.device, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [q, trans_vec],
        lr=lr,
        line_search_fn="strong_wolfe",
        tolerance_change=1e-3,
        max_iter=1,
    )
    working_set = (~dcp.free_flag) & (~dcp.Outlier)
    
    for k in range(n_steps):
        temp = optimizer.step(closure)
        loss_track.append(temp.item())
    
    transform = dr.geometry.quaternions_to_SO3(q)
    rbred_xyz = torch.matmul(propose_rmcom, transform.detach()) + propose_com + trans_vec.detach()

    if verbose:
        # Initial R factors
        dcp.calc_fprotein(atoms_position_tensor=xyz)
        dcp.calc_fsolvent()
        dcp.calc_ftotal(Return=False)
        rw_i, rf_i = dcp.get_rfactors()
        
        # RBRed R factors
        dcp.calc_fprotein(atoms_position_tensor=rbred_xyz)
        dcp.calc_fsolvent()
        dcp.calc_ftotal(Return=False)
        rw_f, rf_f = dcp.get_rfactors()
    
        # RBR rmsd
        rmsd_rbr = rmsd(xyz, rbred_xyz)
        print(f"RBR RMSD: {rmsd_rbr:.2f}, R factors: {rw_i.item():.3f}/{rf_i.item():.3f} --> {rw_f.item():.3f}/{rf_f.item():.3f}", flush=True)

    return loss_track, rbred_xyz, transform.detach(), trans_vec.detach()