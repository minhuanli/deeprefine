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
            loss = -torch.logsumexp(-energy_z + log_det_Jxz)
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
            loss = -torch.logsumexp(-Ereg + log_det_Jzx)
            return loss
        else:
            return torch.mean(Ereg - log_det_Jzx)
