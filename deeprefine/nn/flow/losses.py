import torch
from deeprefine.nn.utils import linlogcut


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

    def __init__(self, energy_function, iwae=False):
        """
        energy_function: callable energy function
        """
        self.energy_function = energy_function
        self.iwae = iwae

    def __call__(self, args, temperature=1.0, Ehigh=20000, Emax=1e10):
        output_x, log_det_Jzx = args[0], args[1]
        E = self.energy_function(output_x) / temperature
        Ereg = linlogcut(E, Ehigh, Emax)
        if self.iwae:
            loss = -torch.logsumexp(-Ereg + log_det_Jzx, 0)[0]
            return loss
        else:
            return torch.mean(Ereg - log_det_Jzx)


class RClossV1:
    """
    Use negative variance of RC features as RC loss
    """
    def __init__(self, rc_feature_indices):
        self.rc_feature_indices = rc_feature_indices

    def __call__(self, args):
        output_x, _ = args[0], args[1]
        rc_coords = output_x[:, self.rc_feature_indices]
        return 1.0, -torch.mean(torch.var(rc_coords, dim=0))


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
        return adaptive_scale.detach(), -sample_var

