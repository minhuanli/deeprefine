import torch
import numpy as np
import torch.nn as nn

from deeprefine.nn.flow.basics import Flow
from deeprefine.nn.flow.invertible_blocks import SplitChannels, MergeChannels, RealNVP
from deeprefine.protein.icconverter import ICConverter
from deeprefine.protein.whiten import Whitener
from deeprefine.utils import assert_numpy, assert_tensor, try_gpu

# TODO: test the save and load of Sequential flow
class SequentialFlow(Flow):
    def __init__(self, blocks):
        """
        Represents a diffeomorphism that can be computed
        as a discrete finite stack of layers.
        
        Returns the transformed variable and the log determinant
        of the Jacobian matrix.
            
        Parameters
        ----------
        blocks : Tuple / List of flow blocks
        """
        super().__init__()
        self._blocks = nn.ModuleList(blocks)
    
    @property
    def dim_in(self):
        return self._blocks[0].dim_in
    
    @property
    def dim_out(self):
        return self._blocks[-1].dim_out
    
    def summarize(self):
        for block in self._blocks:
            print(f"{block.__class__.__name__:<15}: {str(block.dim_in):>12}  ->  {str(block.dim_out):>12}")

    def forward(self, xs, inverse=False, **kwargs):
        """
        Transforms the input along the diffeomorphism and returns
        the transformed variable together with the volume change.
            
        Parameters
        ----------
        x : PyTorch Floating Tensor.
            Input variable to be transformed. 
            Tensor of shape `[..., n_dimensions]`.
        inverse: boolean.
            Indicates whether forward or inverse transformation shall be performed.
            If `True` computes the inverse transformation.
        
        Returns
        -------
        z: PyTorch Floating Tensor.
            Transformed variable. 
            Tensor of shape `[..., n_dimensions]`.
        log_det_J : PyTorch Floating Tensor.
            Total volume change as a result of the transformation.
            Corresponds to the log determinant of the Jacobian matrix.
        """
        log_det_J = 0.0
        blocks = self._blocks
        if inverse:
            blocks = reversed(blocks)
        for i, block in enumerate(blocks):
            xs, ddlogp = block(xs, inverse=inverse, **kwargs)
            log_det_J += ddlogp
        return xs, log_det_J

    def _forward(self, *args, **kwargs):
        return self.forward(*args, **kwargs, inverse=False)

    def _inverse(self, *args, **kwargs):
        return self.forward(*args, **kwargs, inverse=True)

    def __iter__(self):
        return iter(self._blocks)

    def __getitem__(self, index):
        if isinstance(index, int):
            return self._blocks[index]
        else:
            indices = np.arange(len(self))[index]
            return SequentialFlow([self._blocks[i] for i in indices])

    def __len__(self):
        return len(self._blocks)
    

class BoltzmannGenerator(object):
    def __init__(self, flow, icconverter=None, whitener=None, energy_model=None, prior='normal'):
        self.flow = flow
        self.icconverter = icconverter
        self.whitener = whitener
        self.energy_model = energy_model
        self.prior = prior
        dim_in = flow.dim_in
        
        if whitener is not None:
            assert dim_in == whitener.dim_out, "dim_out of whitener must be equal to dim_in of flow!"
            dim_in = whitener.dim_in
        
        if icconverter is not None:
            assert dim_in == icconverter.dim_out, "dim_out of iccoverter must be equal to dim_in of whitener or flow!"
            dim_in = icconverter.dim_in
        
        self.dim_in = dim_in
        self.dim_out = flow.dim_out        

    def TxzJ(self, x):
        """Transform x space to z space, with Jacobian
        x : Tensor, [n_batch, dim_in]
        """
        if self.icconverter is not None:
            x = self.icconverter.xyz2ic(x)
        if self.whitener is not None:
            x = self.whitener.whiten(x)
        z, log_det_Jxz = self.flow(x)
        return z, log_det_Jxz
    
    def TzxJ(self, z):
        """Transform z space to x space, with Jacobian
        z : Tensor, [n_batch, dim_out]
        """
        x, log_det_Jzx = self.flow(z, inverse=True)
        if self.whitener is not None:
            x = self.whitener.blacken(x)
        if self.icconverter is not None:
            x = self.icconverter.ic2xyz(x)
        return x, log_det_Jzx
    
    def summarize(self):
        """Print layer type and dim transformation"""
        if self.icconverter is not None:
            print(f"{self.icconverter.__class__.__name__:<15}: {str(self.icconverter.dim_in):>12}  ->  {str(self.icconverter.dim_out):>12}")
        if self.whitener is not None:
            print(f"{self.whitener.__class__.__name__:<15}: {str(self.whitener.dim_in):>12}  ->  {str(self.whitener.dim_out):>12}")
        self.flow.summarize()
       
    def energy_z(self, z, temperature=1.0):
        """Calculate the effective energy of z in latent sapce given a prior
        z : Tensor or array, [n_batch, dim_out]
        """
        z = assert_numpy(z)
        if self.prior == 'normal':
            E = self.dim_out * np.log(np.sqrt(temperature)) + \
                np.sum(z**2 / (2*temperature), axis=1)
        elif self.prior == 'lognormal':
            sample_z_normal = np.log(z)
            E = np.sum(sample_z_normal**2 / (2*temperature),
                       axis=1) + np.sum(sample_z_normal, axis=1)
        elif self.prior == 'cauchy':
            E = np.sum(np.log(1 + (z/temperature)**2), axis=1)
        return E

    def sample_z(self, std=1.0, nsample=10000, return_energy=False):
        """ Samples from prior distribution in z
        Parameters:
        -----------
        temperature : float
            Relative temperature. Equal to the variance of the isotropic Gaussian sampled in z-space.
        nsample : int
            Number of samples
        Returns:
        --------
        sample_z : array
            Samples in z-space
        energy_z : array
            Energies of z samples (optional)
        """
        sample_z = None
        energy_z = None
        if self.prior == 'normal':
            sample_z = std * np.random.randn(nsample, self.dim_out)
        elif self.prior == 'lognormal':
            sample_z_normal = std * np.random.randn(nsample, self.dim_out)
            sample_z = np.exp(sample_z_normal)
        elif self.prior == 'cauchy':
            from scipy.stats import cauchy
            sample_z = cauchy(loc=0, scale=std **
                              2).rvs(size=(nsample, self.dim_out))
        else:
            raise NotImplementedError(
                'Sampling for prior ' + self.prior + ' is not implemented.')

        if return_energy:
            energy_z = self.energy_z(sample_z)
            return sample_z, energy_z
        else:
            return sample_z
    
    def sample(self, std=1.0, temperature=1.0, nsample=10000):
        """ Samples from prior distribution in z and produces generated x configurations

        Parameters:
        -----------
        std : float
            Standard deviation, used to sample prior in z space

        temperature : float
            Relative temperature. Used to calcualte energy of configurations in x space

        nsample : int
            Number of samples

        Returns:
        --------
        sample_z : array
            Samples in z-space

        sample_x : array
            Samples in x-space

        energy_z : array
            Energies of z samples

        energy_x : array
            Energies of x samples

        log_w : array
            Log weight of samples
        """

        sample_z, energy_z = self.sample_z(
            std=std, nsample=nsample, return_energy=True)
        sample_z = assert_tensor(sample_z)
        sample_x, Rzx = self.TzxJ(sample_z)
        if self.energy_model is not None:
            energy_x = self.energy_model.energy(sample_x) / temperature
        else:
            energy_x = 0.0
        sample_x = assert_numpy(sample_x)
        energy_x = assert_numpy(energy_x)
        Rzx = assert_numpy(Rzx)
        logw = -energy_x + energy_z + Rzx
        return sample_z, sample_x, energy_z, energy_x, logw
    

def construct_bg(icconverter, whitener, n_realnvp=8, energy_model=None,
           n_layers=2, n_hidden=100, n_layers_scale=None, n_hidden_scale=None,
           activation='relu', activation_scale='tanh', init_output_scale=None, prior='normal', device=try_gpu(), **layer_args):
    """
    
    """    
    blocks = []
    sflow = SplitChannels(whitener.dim_out, nchannels=2)
    blocks.append(sflow)
    for _ in range(n_realnvp):
        blocks.append(RealNVP(sflow.dim_out[0], sflow.dim_out[1], 
                              n_layers=n_layers, n_hidden=n_hidden,
                              activation=activation, n_layers_scale=n_layers_scale, 
                              activation_scale=activation_scale, n_hidden_scale=n_hidden_scale,
                              init_output_scale=init_output_scale, **layer_args))
    mflow = MergeChannels(whitener.dim_out, nchannels=2)
    blocks.append(mflow)
    flow = SequentialFlow(blocks).to(device)
    bg = BoltzmannGenerator(flow, icconverter, whitener, energy_model, prior)
    return bg


    


