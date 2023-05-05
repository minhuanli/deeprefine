import torch

class MLlossNormal:
    """
    The maximum-likelihood loss (or forward KL loss) of the MD data
    U(Txz(x)) - log_det_Jxz, U(z) = (0.5/std**2) * \sum z**2
    """

    def __init__(self, std_z=1.0, iwae=False):
        self.std_z = std_z
        self.iwae = iwae

    def __call__(self, args):
        output_z, log_det_Jxz = args[0], args[1]

        energy_z = (0.5/self.std_z**2) * \
                torch.sum(output_z**2, dim=1, keepdim=True)
        if self.iwae:
            loss = - torch.logsumexp(- energy_z + log_det_Jxz)
            return loss
        else:
            return torch.mean(energy_z - log_det_Jxz)