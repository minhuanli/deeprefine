import torch
import numpy as np
import torch.nn as nn

from deeprefine.nn.flow.basics import Flow
from deeprefine.nn.flow.invertible_blocks import SplitChannels, MergeChannels, RealNVP
from deeprefine.protein.icconverter import ICConverter
from deeprefine.protein.whiten import Whitener
from deeprefine.protein.featurefreezer import FeatureFreezer
from deeprefine.utils import assert_numpy, assert_tensor, try_gpu
from deeprefine.nn.utils import count_parameters

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
            print(
                f"{block.__class__.__name__:<15}: {str(block.dim_in):>12}  ->  {str(block.dim_out):>12}"
            )

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
    def __init__(
        self, flow, icconverter=None, featurefreezer=None, whitener=None, energy_model=None, prior="normal"
    ):
        self.flow = flow
        self.icconverter = icconverter
        self.featurefreezer = featurefreezer
        self.whitener = whitener
        self.energy_model = energy_model
        self.prior = prior
        dim_in = flow.dim_in

        if whitener is not None:
            assert (
                dim_in == whitener.dim_out
            ), "dim_out of whitener must be equal to dim_in of flow!"
            dim_in = whitener.dim_in
        
        if featurefreezer is not None:
            assert (
                dim_in == featurefreezer.dim_out
            ), "dim_out of featurefreezer must be equal to dim_in of whitener!"
            dim_in = featurefreezer.dim_in

        if icconverter is not None:
            assert (
                dim_in == icconverter.dim_out
            ), "dim_out of iccoverter must be equal to dim_in of featurefreezer, whitener or flow!"
            dim_in = icconverter.dim_in

        self.dim_in = dim_in
        self.dim_out = flow.dim_out

    @property
    def device(self):
        return self.whitener.Twhiten.device

    def TxzJ(self, x):
        """Transform x space to z space, with Jacobian
        x : Tensor, [n_batch, dim_in]

        Returns:
        z : Tensor, [n_batch, dim_out]
        Rxz : Tensor, [n_batch, 1]
        """
        if self.icconverter is not None:
            x = self.icconverter.xyz2ic(x)
        if self.featurefreezer is not None:
            x = self.featurefreezer.forward(x)
        if self.whitener is not None:
            x = self.whitener.whiten(x)
        z, log_det_Jxz = self.flow(x)
        return z, log_det_Jxz

    def TzxJ(self, z):
        """Transform z space to x space, with Jacobian
        z : Tensor, [n_batch, dim_out]

        Returns:
        x : Tensor, [n_batch, dim_in]
        Rzx : Tensor, [n_batch, 1]
        """
        x, log_det_Jzx = self.flow(z, inverse=True)
        if self.whitener is not None:
            x = self.whitener.blacken(x)
        if self.featurefreezer is not None:
            x = self.featurefreezer.backward(x)
        if self.icconverter is not None:
            x = self.icconverter.ic2xyz(x)
        return x, log_det_Jzx

    def summarize(self):
        """Print layer type and dim transformation"""
        if self.icconverter is not None:
            print(
                f"{self.icconverter.__class__.__name__:<15}: {str(self.icconverter.dim_in):>12}  ->  {str(self.icconverter.dim_out):>12}"
            )

        if self.featurefreezer is not None:
            print(
                f"{self.featurefreezer.__class__.__name__:<15}: {str(self.featurefreezer.dim_in):>12}  ->  {str(self.featurefreezer.dim_out):>12}"
            )

        if self.whitener is not None:
            print(
                f"{self.whitener.__class__.__name__:<15}: {str(self.whitener.dim_in):>12}  ->  {str(self.whitener.dim_out):>12}"
            )
        self.flow.summarize()
        print(f"{'Number of parameters':<15}: {count_parameters(self.flow):>12}")

    def energy_z(self, z, temperature=1.0):
        """Calculate the effective energy of z in latent sapce given a prior
        z : Tensor or array, [n_batch, dim_out]

        Returns:
        E : Tensor, [n_batch, 1]
        """
        z = assert_tensor(z)
        if self.prior == "normal":
            E = self.dim_out * torch.log(
                torch.tensor(np.sqrt(temperature)).to(z)
            ) + torch.sum(z**2 / (2 * temperature), dim=1)
        elif self.prior == "lognormal":
            sample_z_normal = torch.log(z)
            E = torch.sum(sample_z_normal**2 / (2 * temperature), dim=1) + torch.sum(
                sample_z_normal, dim=1
            )
        elif self.prior == "cauchy":
            E = torch.sum(torch.log(1 + (z / temperature) ** 2), dim=1)
        return E.view(-1, 1)

    def sample_z(self, std=1.0, nsample=10000, return_energy=False):
        """Samples from prior distribution in z
        Parameters:
        -----------
        temperature : float
            Relative temperature. Equal to the variance of the isotropic Gaussian sampled in z-space.
        nsample : int
            Number of samples
        Returns:
        --------
        sample_z : Tensor [n_batch, dim_out]
            Samples in z-space
        energy_z : Tensor, [n_batch, 1]
            Energies of z samples (optional)
        """
        sample_z = None
        energy_z = None
        if self.prior == "normal":
            sample_z = std * torch.randn(nsample, self.dim_out, device=try_gpu())
        elif self.prior == "lognormal":
            sample_z_normal = std * torch.randn(nsample, self.dim_out, device=try_gpu())
            sample_z = torch.exp(sample_z_normal)
        elif self.prior == "cauchy":
            sample_z = (
                torch.distributions.Cauchy(loc=0, scale=std**2)
                .sample([nsample, self.dim_out])
                .to(try_gpu())
            )
        else:
            raise NotImplementedError(
                "Sampling for prior " + self.prior + " is not implemented."
            )

        if return_energy:
            energy_z = self.energy_z(sample_z)
            return sample_z, energy_z
        else:
            return sample_z

    def sample(self, std=1.0, temperature=1.0, nsample=10000, return_Tensor=True):
        """Samples from prior distribution in z and produces generated x configurations

        Parameters:
        -----------
        std : float
            Standard deviation, used to sample prior in z space

        temperature : float
            Relative temperature. Used to calcualte energy of configurations in x space

        nsample : int
            Number of samples

        return_Tensor : boolean
            Whether to return torch.Tensor or np.array, default Tensor

        Returns:
        --------
        sample_z : array or Tensor, [nsample, dim_out]
            Samples in z-space

        sample_x : array or Tensor, [nsample, dim_in]
            Samples in x-space

        energy_z : array or Tensor, [nsample, 1]
            Energies of z samples

        energy_x : array or Tensor, [nsample, 1]
            Energies of x samples

        log_w : array or Tensor, [nsample, 1]
            Log weight of samples
        """

        sample_z, energy_z = self.sample_z(std=std, nsample=nsample, return_energy=True)
        sample_x, Rzx = self.TzxJ(sample_z)
        if self.energy_model is not None:
            energy_x = self.energy_model.energy(sample_x) / temperature
        else:
            energy_x = 0.0
        logw = -energy_x + energy_z + Rzx
        if return_Tensor:
            return sample_z, sample_x, energy_z, energy_x, logw
        else:
            sample_z = assert_numpy(sample_z)
            sample_x = assert_numpy(sample_x)
            energy_z = assert_numpy(energy_z)
            energy_x = assert_numpy(energy_x)
            logw = assert_numpy(logw)
            return sample_z, sample_x, energy_z, energy_x, logw


def construct_bg(
    icconverter,
    featurefreezer,
    whitener,
    n_realnvp=8,
    energy_model=None,
    n_layers=2,
    n_hidden=100,
    n_layers_scale=None,
    n_hidden_scale=None,
    activation=torch.relu,
    activation_scale=torch.tanh,
    init_output_scale=None,
    prior="normal",
    device=try_gpu(),
    **layer_args,
):
    """ """
    blocks = []
    sflow = SplitChannels(whitener.dim_out, nchannels=2)
    blocks.append(sflow)
    for _ in range(n_realnvp):
        blocks.append(
            RealNVP(
                sflow.dim_out[0],
                sflow.dim_out[1],
                n_layers=n_layers,
                n_hidden=n_hidden,
                activation=activation,
                n_layers_scale=n_layers_scale,
                activation_scale=activation_scale,
                n_hidden_scale=n_hidden_scale,
                init_output_scale=init_output_scale,
                **layer_args,
            )
        )
    mflow = MergeChannels(whitener.dim_out, nchannels=2)
    blocks.append(mflow)
    flow = SequentialFlow(blocks).to(device)
    bg = BoltzmannGenerator(flow, icconverter, featurefreezer, whitener, energy_model, prior)
    return bg


def save_bg(bg, filepath):
    """Save bg to a dictionary, without saving the energy model"""
    fulldict = {}

    # Save icconverter if exists
    if bg.icconverter is not None:
        fulldict["icconverter"] = bg.icconverter.to_dict()

    # Save featurefreezer if exists
    if bg.featurefreezer is not None:
        fulldict["featurefreezer"] = bg.featurefreezer.to_dict()

    # Save Whitener if exists
    if bg.whitener is not None:
        fulldict["whitener"] = bg.whitener.to_dict()

    # Save flow in a hacky way
    flowdict = {}
    flowdict["n_realnvp"] = len(bg.flow) - 2
    flowdict["realnvp_config"] = bg.flow[1].config
    flowdict["state_dict"] = bg.flow.state_dict()

    fulldict["flow"] = flowdict
    fulldict["prior"] = bg.prior

    import pickle

    with open(filepath, "wb") as f:
        pickle.dump(fulldict, f, pickle.HIGHEST_PROTOCOL)


def load_bg(filepath, energy_model, device=try_gpu()):
    import pickle

    with open(filepath, "rb") as f:
        fulldict = pickle.load(f)

    if fulldict.get("icconverter") is None:
        icconverter = None
    else:
        icconverter = ICConverter.from_dict(fulldict["icconverter"])

    if fulldict.get("featurefreezer") is None:
        featurefreezer = None
    else:
        featurefreezer = FeatureFreezer.from_dict(fulldict["featurefreezer"])

    if fulldict.get("whitener") is None:
        whitener = None
    else:
        whitener = Whitener.from_dict(fulldict["whitener"])

    flowdict = fulldict["flow"]
    bg = construct_bg(
        icconverter,
        featurefreezer,
        whitener,
        flowdict["n_realnvp"],
        energy_model,
        **flowdict["realnvp_config"],
        prior=fulldict["prior"],
        device=device,
    )
    bg.flow.load_state_dict(flowdict["state_dict"])
    return bg
