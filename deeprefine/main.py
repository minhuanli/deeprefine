import deeprefine as dr
import torch
import numpy as np
from openmm import unit
import os

def train():

  base_dir = '/scratch/pr-kdd-1/gw/deeprefine'
  traj_file = os.path.join(base_dir, "data/1BTI/1bti_implicit_traj.h5")
  pdb_file = os.path.join(base_dir, "data/1BTI/1bti_fixed.pdb")

  sim_x, top = dr.utils.align_md(traj_file, shuffle=True, ref_pdb=pdb_file)
  top2, mm_1bti = dr.setup_protein(pdb_file, 300,
                                   implicit_solvent=True,
                                   platform='CUDA',
                                   length_scale=unit.nanometer)

  icconverter = dr.ICConverter(top, vec_angles=True)
  ic0 = icconverter.xyz2ic(dr.assert_tensor(sim_x))
  cosangle_idx = np.concatenate([icconverter.cosangle_idxs, icconverter.costorsion_idxs])
  sinangle_idx = np.concatenate([icconverter.sinangle_idxs, icconverter.sintorsion_idxs])
  featurefreezer = dr.FeatureFreezer(ic0, bond_idx=icconverter.bond_idxs,
                                     cosangle_idx=cosangle_idx, sinangle_idx=sinangle_idx)
  ic1 = featurefreezer.forward(ic0)
  whitener = dr.Whitener(X0=ic1,
                         dim_cart_signal=icconverter.dim_cart_signal,
                         keepdims=-6)

  realnvp_args = {
      "n_layers" : 4,
      "n_hidden" : [128,256,128],
      "activation" : torch.relu,
      "activation_scale" : torch.tanh,
      "init_output_scale" : 0.01
  }
  bg = dr.construct_bg(icconverter, featurefreezer, whitener,
                       n_realnvp=8, **realnvp_args, prior='normal')
  bg.summarize()
  optim = torch.optim.Adam(bg.flow.parameters(), lr=0.001)
  mltrainer = dr.nn.flow.MLTrainer(bg, optim, iwae=False)
  X0 = torch.tensor(sim_x, dtype=torch.float32)

  batchsize = [128]*2 + [256]*2 + [512]*6 + [1024]*10 + [2048]*20
  epochs = 2 + 2 + 6 + 10 + 20
  mltrain_record = mltrainer.train(X0, epochs=epochs, batch_size=batchsize,
                                   checkpoint_epoch=4,
                                   checkpoint_name=os.path.join(base_dir, "results/20230731/mltrain_"))


  bg = dr.load_bg(os.path.join(base_dir, "results/20230731/mltrain__39.pkl"), mm_1bti)
  optim2 = torch.optim.Adam(bg.flow.parameters(), lr=0.0001)
  kltrainer = dr.nn.flow.FlexibleTrainer(bg, optim2)
  epochs_KL     = [  1,   1,   1,   1,   1,   1,  1,  1,  2, 2, 2, 3, 4]
  high_energies = [1e10,  1e9,  1e8,  1e7,  1e6,  1e5,  1e5,  1e5,  5e4,  5e4,  2e4,  2e4, 2e4]
  w_KLs         = [1e-12, 1e-6, 1e-5, 1e-4, 1e-3, 1e-3, 5e-3, 1e-3, 5e-3, 5e-2, 0.05, 0.05, 0.05]
  report = []
  Elevels = list(set(high_energies))
  Elevels.sort()

  for s, epochs in enumerate(epochs_KL):
      report = kltrainer.train(X0,
                               epochs=epochs_KL[s], batchsize_ML=1024, batchsize_KL=1024,
                               w_KL=w_KLs[s], Ehigh=high_energies[s],
                               record=report, checkpoint_name=os.path.join(base_dir,f"results/20230731/kltrain_{s}"))
      # Analyze
      samples_z = bg.sample_z(nsample=2000, return_energy=False)
      samples_x, _ = bg.TzxJ(samples_z)
      samples_e = dr.assert_numpy(bg.energy_model.energy(samples_x))
      energy_violations = [np.count_nonzero(samples_e > E) for E in Elevels]
      print('Energy violations:', flush=True)
      for E, V in zip(Elevels, energy_violations):
          print(V, '\t>\t', E, flush=True)


def generate():
  base_dir = '/scratch/pr-kdd-1/gw/deeprefine'
  pdb_file = os.path.join(base_dir, "data/1BTI/1bti_fixed.pdb")
  _, mm_1bti = dr.setup_protein(pdb_file, 300,
                                   implicit_solvent=True,
                                   platform='CUDA',
                                   length_scale=unit.nanometer)
  final_checkpoint = os.path.join(base_dir, "results/20230731/kltrain_11_2.pkl")
  bg = dr.load_bg(final_checkpoint, mm_1bti)
  dist_z = torch.distributions.MultivariateNormal(torch.zeros(bg.dim_out).to('cuda'), torch.diag_embed(torch.ones(bg.dim_out).to('cuda')))
  n_batch = 500
  samples_z = dist_z.sample((n_batch,))
  log_prob_z = dist_z.log_prob(samples_z)
  samples_x, _ = bg.TzxJ(samples_z)
  samples_e = torch.from_numpy(dr.assert_numpy(bg.energy_model.energy(samples_x)))
  temp = 300
  log_prob_e_montecarlo_unscaled = -samples_e/temp
  zum_over_all_ztates = (log_prob_e_montecarlo_unscaled.exp()).sum()
  log_prob_e_montecarlo = log_prob_e_montecarlo_unscaled - zum_over_all_ztates
  return samples_x, samples_e, log_prob_z, log_prob_e_montecarlo

if __name__ == '__main__':
  train()
  generate()