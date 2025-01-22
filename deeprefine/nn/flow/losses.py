import torch
import math
from deeprefine.nn.utils import linlogcut
import numpy as np


class MLlossNormal:
    """
    The maximum-likelihood loss (or forward KL loss) of the MD data
    U(Txz(x)) - log_det_Jxz, U(z) = (0.5/std**2) * \sum z**2
    """

    def __init__(self, iwae):
        self.iwae = iwae

    def __call__(self, args, std_z=1.0):
        output_z, log_det_Jxz = args[0], args[1]

        energy_z = (0.5 / std_z**2) * torch.sum(output_z**2, dim=1, keepdim=True)
        if self.iwae:
            loss = -torch.logsumexp(-energy_z + log_det_Jxz, 0)[0]
            return loss
        else:
            return torch.mean(energy_z - log_det_Jxz)

class KLloss:
    """
    The reverse KL loss of the normalizing flow
    U(F(zx(z))) - log_det_Jzx, U(x) is given by the energy function
    """

    def __init__(self, energy_function, transform, iwae=False):
        """
        energy_function: callable energy function
        """
        self.energy_function = energy_function
        self.iwae = iwae
        self.transform = transform

    def __call__(self, args, temperature=1.0, Ehigh=1e4, Emax=2e4):
        output_x, log_det_Jzx = args[0], args[1]
        E = self.energy_function(output_x) / temperature
        Ereg = linlogcut(self.transform(E), Ehigh, Emax)
        if Ereg.isnan().sum().item() > 0:
            print(f"Got {Ereg.isnan().sum().item()}/{len(Ereg)} nan in the energy, skip that sample...", flush=True)
        if log_det_Jzx.isnan().sum().item() > 0:
            print(f"Got {log_det_Jzx.isnan().sum().item()}/{len(log_det_Jzx)} nan in the log_det_Jzx, skip that sample...", flush=True)
        if self.iwae:
            loss = -torch.logsumexp(-Ereg + log_det_Jzx, 0)[0]
            return loss
        else:
            return torch.nanmean(Ereg - log_det_Jzx)

class RClossV1:
    """
    Use negative variance of RC features as RC loss
    """
    def __init__(self, rc_feature_indices):
        self.rc_feature_indices = rc_feature_indices

    def __call__(self, args):
        output_x, _ = args[0], args[1]
        rc_coords = output_x[:, self.rc_feature_indices]
        return -torch.mean(torch.var(rc_coords, dim=0))


class RClossV2:
    """
    Use negative variance of RC features as RC loss, with adaptive scale
    """
    def __init__(self, rc_feature_indices, target_var):
        self.rc_feature_indices = rc_feature_indices
        self.target = target_var

    def __call__(self, args):
        output_x, _ = args[0], args[1]
        rc_coords = output_x[:, self.rc_feature_indices]
        sample_var = torch.mean(torch.var(rc_coords, dim=0))
        adaptive_scale = 10.0**torch.log(self.target/sample_var)
        return -sample_var * adaptive_scale.detach()


class SSEloss:
    """
    Structure Factor Sum Square Error Loss
    """
    def __init__(self, dcp, unit_change=1.0) -> None:
        self.dcp = dcp
        self.n_atoms = dcp.n_atoms
        self.unit_change = unit_change
        self.working_set = (~dcp.free_flag) & (~dcp.Outlier)
    
    def __call__(self, output_x, NLLhigh=1e7, NLLmax=1e10, w_NLL=1.0) -> torch.Tensor:
        self.dcp.calc_fprotein_batch(self.unit_change * output_x.reshape(-1, self.n_atoms, 3))
        self.dcp.calc_fsolvent_batch()
        Fm_batch = self.dcp.calc_ftotal_batch()
        NLL = w_NLL * ((Fm_batch.abs() - self.dcp.Fo).square()/(2*self.dcp.SigF.square()))[..., self.working_set].sum(1)
        NLLreg = linlogcut(NLL, NLLhigh, NLLmax)
        return NLLreg
    

class SSElossV2:
    """
    Normalized structure Factor Sum Square Error Loss
    """
    def __init__(self, dcp, unit_change=1.0) -> None:
        self.dcp = dcp
        self.n_atoms = dcp.n_atoms
        self.unit_change = unit_change
        self.working_set = (~dcp.free_flag) & (~dcp.Outlier)
    
    def __call__(self, output_x, NLLhigh=1e7, NLLmax=1e10, w_NLL=1.0) -> torch.Tensor:
        self.dcp.calc_fprotein_batch(self.unit_change * output_x.reshape(-1, self.n_atoms, 3))
        self.dcp.calc_fsolvent_batch()
        Fc_batch = self.dcp.calc_ftotal_batch()
        Ec_batch = self.dcp.calc_Ec(Fc_batch)
        NLL = w_NLL * ((Ec_batch.abs() - self.dcp.Eo).square()/(2*self.dcp.SigEo.square()))[..., self.working_set].sum(1)
        NLLreg = linlogcut(NLL, NLLhigh, NLLmax)
        return NLLreg


class InflatedSSEloss:
    """
    Normalized structure Factor Sum Square Error Loss with inflated variance
    \sum (Eo - sigmaA*|Ec|)^2 / (1-sigmaA^2 + 2*sigma_E^2)
    """
    def __init__(self, dcp, unit_change=1.0) -> None:
        self.dcp = dcp
        self.n_atoms = dcp.n_atoms
        self.unit_change = unit_change
        self.working_set = (~dcp.free_flag) & (~dcp.Outlier)
    
    def __call__(self, output_x, NLLhigh=1e7, NLLmax=1e10, w_NLL=1.0) -> torch.Tensor:
        self.dcp.calc_fprotein_batch(self.unit_change * output_x.reshape(-1, self.n_atoms, 3))
        self.dcp.calc_fsolvent_batch()
        Fc_batch = self.dcp.calc_ftotal_batch()
        Ec_batch = self.dcp.calc_Ec(Fc_batch)
        sigmaAs = self.dcp.get_sigmaA(Ec_batch)
        NLL = 0.0
        for i in range(self.dcp.n_bins):
            index_i = self.dcp.bins[self.working_set] == i
            Ec_i = Ec_batch[..., self.working_set][..., index_i]
            Eo_i = self.dcp.Eo[self.working_set][index_i]
            SigEi = self.dcp.SigEo[self.working_set][index_i]
            NLL_i = ((Eo_i - sigmaAs[..., i].unsqueeze(-1)*Ec_i.abs()).square() / (1.0 - sigmaAs[..., i].unsqueeze(-1)**2 + 2*SigEi.square().unsqueeze(0))).sum(1)
            NLL = NLL + w_NLL * NLL_i
        NLLreg = linlogcut(NLL, NLLhigh, NLLmax)
        return NLLreg
    

class NLLloss:
    """
    Crystallographic Log Likelihood loss, using (3a) and (3b) from Read, 2016
    """
    def __init__(self, dcp, unit_change=1.0) -> None:
        self.dcp = dcp
        self.n_atoms = dcp.n_atoms
        self.unit_change = unit_change
        self.working_set = (~dcp.free_flag) & (~dcp.Outlier)
    
    def get_acentric_NLL(self, sigmaA, Eo, Ec, sigmaE):
        """
        Acentric reflections
        """
        inf_variance = (1 - sigmaA.square().unsqueeze(-1) + 2*sigmaE.square().unsqueeze(0)) # [N_batch, N_hkl]
        bessel_arg = (2 * sigmaA.unsqueeze(-1) * Eo * Ec) / inf_variance # [N_batch, N_hkl]
        exp_bessel = torch.special.i0e(bessel_arg) # [N_batch, N_hkl]
        nll_a = (Eo - sigmaA.unsqueeze(-1) * Ec).square() / inf_variance - torch.log(exp_bessel) - torch.log(2*Eo/inf_variance) # [N_batch, N_hkl]
        return nll_a.sum(-1)
    
    def get_centric_NLL(self, sigmaA, Eo, Ec, sigmaE):
        """
        Centric reflections
        """
        inf_variance = (1 - sigmaA.square().unsqueeze(-1) + sigmaE.square().unsqueeze(0)) # [N_batch, N_hkl] 
        cosh_arg = (sigmaA.unsqueeze(-1) * Eo * Ec) / inf_variance # [N_batch, N_hkl] 
        nll_c = (Eo.square() + sigmaA.square().unsqueeze(-1) * Ec.square()) / (2 * inf_variance) - self.logcosh(cosh_arg) - 0.5*torch.log(2/(math.pi*inf_variance)) # [N_batch, N_hkl]
        return nll_c.sum(-1)
    
    def logcosh(self, x):
        """
        https://stackoverflow.com/questions/57785222/avoiding-overflow-in-logcoshx
        """
        # s always has real part >= 0
        s = torch.sign(x) * x
        p = torch.exp(-2 * s)
        return s + torch.log1p(p) - math.log(2)

    def __call__(self, output_x, NLLhigh=1e7, NLLmax=1e10, w_NLL=1.0, sub_ratio=1.0) -> torch.Tensor:
        self.dcp.calc_fprotein_batch(self.unit_change * output_x.reshape(-1, self.n_atoms, 3))
        self.dcp.calc_fsolvent_batch()
        Fc_batch = self.dcp.calc_ftotal_batch()
        # Do ensemble average first
        Fc_average = torch.nanmean(Fc_batch, dim=0)
        Ec_average = self.dcp.calc_Ec(Fc_average)
        sigmaAs = self.dcp.get_sigmaA(Ec_average)
        NLL = 0.0
        # Do a subsampling of the working index
        working_set = self.working_set.copy()
        num_true_to_false = int((1.0 - sub_ratio) * working_set.sum())
        true_indices = np.random.choice(np.flatnonzero(working_set), size=num_true_to_false, replace=False)
        working_set[true_indices] = False
        for i in range(self.dcp.n_bins):
            index_i = self.dcp.bins[working_set] == i
            Ec_i = Ec_average[working_set][index_i].abs()
            Eo_i = self.dcp.Eo[working_set][index_i]
            SigEi = self.dcp.SigEo[working_set][index_i]
            Centrici = self.dcp.centric[working_set][index_i]
            NLL_a = self.get_acentric_NLL(
                sigmaA=sigmaAs[i],
                Eo=Eo_i[~Centrici],
                Ec=Ec_i[~Centrici],
                sigmaE=SigEi[~Centrici]
            )
            NLL_c = self.get_centric_NLL(
                sigmaA=sigmaAs[i],
                Eo=Eo_i[Centrici],
                Ec=Ec_i[Centrici],
                sigmaE=SigEi[Centrici]
            )
            NLL_i = NLL_a + NLL_c
            NLL = NLL + w_NLL * NLL_i
        NLLreg = linlogcut(NLL, NLLhigh, NLLmax)
        return NLLreg
    
class MixNLLloss(NLLloss):
    def __init__(self, dcp, unit_change=1, sse_transform=None, nll_transform=None) -> None:
        super().__init__(dcp, unit_change)
        self.sse_transform = sse_transform
        self.nll_transform = nll_transform

    def __call__(self, output_x, NLLhigh=1e7, NLLmax=1e10, w_NLL=1.0, r_NLL=0.0, resol_cut=0.1) -> torch.Tensor:
        self.dcp.calc_fprotein_batch(self.unit_change * output_x.reshape(-1, self.n_atoms, 3))
        self.dcp.calc_fsolvent_batch()
        Fc_batch = self.dcp.calc_ftotal_batch()
        # Do ensemble average first
        Fc_average = torch.nanmean(Fc_batch, dim=0)
        Ec_average = self.dcp.calc_Ec(Fc_average)
        sigmaAs = self.dcp.get_sigmaA(Ec_average)
        NLL = 0.0
        # Do a resolution cutoff
        full_working_set = self.working_set.copy()
        resol_bool = self.dcp.dHKL > resol_cut
        working_set = (full_working_set) & (resol_bool)
        for i in range(self.dcp.n_bins):
            index_i = self.dcp.bins[working_set] == i
            if index_i.sum() == 0:
                continue
            Ec_i = Ec_average[working_set][index_i].abs()
            Eo_i = self.dcp.Eo[working_set][index_i]
            SigEi = self.dcp.SigEo[working_set][index_i]
            Centrici = self.dcp.centric[working_set][index_i]
            NLL_a = self.get_acentric_NLL(
                sigmaA=sigmaAs[i],
                Eo=Eo_i[~Centrici],
                Ec=Ec_i[~Centrici],
                sigmaE=SigEi[~Centrici]
            )
            NLL_c = self.get_centric_NLL(
                sigmaA=sigmaAs[i],
                Eo=Eo_i[Centrici],
                Ec=Ec_i[Centrici],
                sigmaE=SigEi[Centrici]
            )
            NLL_i = NLL_a + NLL_c
            NLL = NLL + NLL_i
        #NLLreg = linlogcut(NLL, NLLhigh, NLLmax)
        NLLreg = self.nll_transform(NLL)
        if NLLreg.isnan().sum().item() > 0:
            print(f"Got nan in the NLL, skip that batch...", flush=True)
        SSE = ((Fc_average.abs() - self.dcp.Fo).square()/(2*self.dcp.SigF.square()))[self.working_set].sum()
        SSEreg = self.sse_transform(SSE)
        if SSEreg.isnan().sum().item() > 0:
            print(f"Got nan in the SSE, skip that batch...", flush=True)
        #SSEreg = linlogcut(SSE, NLLhigh, NLLmax)
        LLreg = w_NLL * (NLLreg * r_NLL + SSEreg * (1-r_NLL))
        return LLreg, NLLreg.item(), SSEreg.item()


        