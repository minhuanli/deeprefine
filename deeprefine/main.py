import deeprefine as dr
import torch
import numpy as np
from openmm import unit
import os
import argparse


def quick_paths():
  base_dir = '/Users/gw/repos/deeprefine' # #'/scratch/pr-kdd-1/gw/deeprefine'
  traj_file = os.path.join(base_dir, "data/1BTI/1bti_implicit_traj.h5")
  pdb_file = os.path.join(base_dir, "data/1BTI/1bti_fixed.pdb")
  final_checkpoint_fname = os.path.join(base_dir, "results/20230731/mltrain__39.pkl")
  temp = 300.0
  final_kl_checkpoint_fname = os.path.join(base_dir, "results/20230731_argparse/kltrain_12_3.pkl")
  return base_dir, traj_file, pdb_file, final_checkpoint_fname, temp, final_kl_checkpoint_fname

def main():
  # TODO: add arg parse for hard coded paths
  parser = argparse.ArgumentParser()
  parser.add_argument("-base",
                      "--base_dir",
                      type=str,
                      default='example',
                      help="base deeprefine repo directory",
                      )
  parser.add_argument("-traj",
                      "--traj_file",
                      type=str,
                      default='example',
                      help="trajectory file",
                      )
  parser.add_argument("-pdb",
                      "--pdb_file",
                      type=str,
                      default='example',
                      help="trajectory file",
                      )
  parser.add_argument("-ml",
                      "--ml_checkpoint_prefix",
                      type=str,
                      default='example',
                      help="mltrainer checkpoint prefix",
                      )
  parser.add_argument("-fml",
                      "--final_ml_checkpoint_fname",
                      type=str,
                      default='example',
                      help="final mltrainer checkpoint fname",
                      )
  parser.add_argument("-kl",
                      "--kl_checkpoint_prefix",
                      type=str,
                      default='example',
                      help="kltrainer checkpoint prefix",
                      )
  parser.add_argument("-fkl",
                      "--final_kl_checkpoint_fname",
                      type=str,
                      default='example',
                      help="final kltrainer checkpoint fname",
                      )
  parser.add_argument("-nb",
                      "--n_batch",
                      type=int,
                      default=1,
                      help="number of batch samples",
                      )
  parser.add_argument("-t",
                      "--temp",
                      type=float,
                      default=300.0,
                      help="MD temperature",
                      )
  args = parser.parse_args()

  bg, mm_pdb, X0, top = setup_train(args.traj_file, args.pdb_file, args.temp)
  bg, report = train(bg, mm_pdb, X0, args.ml_checkpoint_prefix, args.final_ml_checkpoint_fname, args.kl_checkpoint_prefix)

  # base_dir, traj_file, pdb_file, final_checkpoint_fname, temp, final_kl_checkpoint_fname = quick_paths()
  # bg, _ = load_checkpoint(pdb_file, final_kl_checkpoint_fname, temp)
  atomic_states = encode(bg, X0)

  samples_z, samples_x, samples_e, log_prob_z, log_prob_e_montecarlo = generate(bg, args.n_batch, [0])
  dr.utils.save_samples_to_pdb(samples_x.detach().cpu(), top, os.path.join(args.base_dir, 'results/20230731_argparse/decoded_samples.pdb'))

def setup_train(traj_file, pdb_file, temp):
  sim_x, top = dr.utils.align_md(traj_file, shuffle=True, ref_pdb=pdb_file)
  sim_x = sim_x[:3]
  top2, mm_pdb = dr.setup_protein(pdb_file, temp,
                                   implicit_solvent=True,
                                   platform='CPU',
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
  X0 = torch.tensor(sim_x, dtype=torch.float32)
  return bg, mm_pdb, X0, top

def train(bg, mm_pdb, X0, ml_checkpoint_prefix, final_checkpoint_fname, kl_checkpoint_prefix):
  optim = torch.optim.Adam(bg.flow.parameters(), lr=0.001)
  mltrainer = dr.nn.flow.MLTrainer(bg, optim, iwae=False)
  batchsize = [128]*2 + [256]*2 + [512]*6 + [1024]*10 + [2048]*20
  epochs = 2 + 2 + 6 + 10 + 20
  mltrain_record = mltrainer.train(X0, epochs=epochs, batch_size=batchsize,
                                   checkpoint_epoch=4,
                                   checkpoint_name=ml_checkpoint_prefix)

  bg = dr.load_bg(final_checkpoint_fname, mm_pdb)
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
                               record=report, checkpoint_name=kl_checkpoint_prefix+f'{s}')
      # Analyze
      samples_z = bg.sample_z(nsample=2000, return_energy=False)
      samples_x, _ = bg.TzxJ(samples_z)
      samples_e = dr.assert_numpy(bg.energy_model.energy(samples_x))
      energy_violations = [np.count_nonzero(samples_e > E) for E in Elevels]
      print('Energy violations:', flush=True)
      for E, V in zip(Elevels, energy_violations):
          print(V, '\t>\t', E, flush=True)
  return bg, report, X0

def load_checkpoint(pdb_file, final_kl_checkpoint_fname, temp):
  _, mm_pdb = dr.setup_protein(pdb_file, temp,
                                   implicit_solvent=True,
                                   platform='CUDA',
                                   length_scale=unit.nanometer)
  bg = dr.load_bg(final_kl_checkpoint_fname, mm_pdb)
  return bg, mm_pdb

def generate(bg, n_batch, idx=[], dist_z=None):
  if dist_z is None:
    dist_z = torch.distributions.MultivariateNormal(torch.zeros(bg.dim_out).to('cuda'), torch.diag_embed(torch.ones(bg.dim_out).to('cuda')))
  samples_z = dist_z.sample((n_batch,))
  if len(idx) > 0:
    samples_z_sparse = torch.zeros_like(samples_z)
    for i in idx:
      samples_z_sparse[:,i] = samples_z[:,i]
    samples_z = samples_z_sparse
  log_prob_z = dist_z.log_prob(samples_z)
  samples_x, _ = bg.TzxJ(samples_z)
  samples_e = torch.from_numpy(dr.assert_numpy(bg.energy_model.energy(samples_x)))
  temp = 1
  log_prob_e_montecarlo_unscaled = -samples_e/temp
  zum_over_all_ztates = (log_prob_e_montecarlo_unscaled.exp()).sum()
  log_prob_e_montecarlo = log_prob_e_montecarlo_unscaled - zum_over_all_ztates
  return samples_z, samples_x, samples_e, log_prob_z, log_prob_e_montecarlo

def encode(bg, X0):
  # bg, _ = load_checkpoint(pdb_file, final_kl_checkpoint_fname, temp)
  n_batch = len(X0)
  n_atom = X0.shape[-1]//3
  latent_states = bg.TxzJ(X0.to('cuda'))[0].reshape(n_batch, bg.dim_out // 3, 3)
  return latent_states


if __name__ == '__main__':
  main()