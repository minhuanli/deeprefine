'''
Run molecule replacement with hierachical grid search and gradient descent
'''
import argparse
import json
import logging, sys, os
import deeprefine as dr
import SFC_Torch as sfc
import torch
import numpy as np
import gemmi

class ArgumentParser(argparse.ArgumentParser):
    def __init__(self):
        super().__init__(
            formatter_class=argparse.RawTextHelpFormatter, description=__doc__
        )

        # Required arguments
        self.add_argument("pdb", help="The path of the starting model PDB file")

        self.add_argument("mtz", help="The path of the experimental mtz file")

        # Optional arguments
        self.add_argument(
            "-o", 
            "--outdir", 
            type=str, 
            default='./mr_solution/',
            help="The path of output folder dir, default ./mr_solution/",
        )

        self.add_argument( 
            "--Fcolumn", 
            type=str, 
            default='FP',
            help="Column name of the structure factor magnitude, default FP"
        )

        self.add_argument( 
            "--SigFcolumn", 
            type=str, 
            default='SIGFP',
            help="Column name of the structure factor magnitude standard deviation, default SIGFP"
        )

        self.add_argument( 
            "--freeflag", 
            type=str, 
            default='R-free-flags',
            help="Column name of the freeflag value, default R-free-flags, Phenix(CNS/XPOLAR) convention"
        )

        self.add_argument( 
            "--testset_value", 
            type=int, 
            default=1,
            help="testset freeflag value, default 1"
        )

        self.add_argument( 
            "--dmin", 
            type=float, 
            default=4.0,
            help="High resolution cutoff, default 4.0"
        )

        self.add_argument( 
            "--pts_min", 
            type=float, 
            default=7.0,
            help="Minimum patterson vector length in rotation search, default 7.0"
        )

        self.add_argument( 
            "--pts_max", 
            type=float, 
            default=12.0,
            help="Maximum patterson vector length in rotation search, default 12.0"
        )

        self.add_argument( 
            "--n_rot", 
            type=int, 
            default=4,
            help="Number of hierachical rotational search rounds, default 4"
        )

        self.add_argument( 
            "--n_trans", 
            type=int, 
            default=4,
            help="Number of hierachical translational search rounds, default 4"
        )

        self.add_argument( 
            "--trans_basegrid", 
            type=int, 
            default=24,
            help="Number of grid per axis for initial translational search, default 24"
        )

        self.add_argument( 
            "--rot_basegrid", 
            type=int, 
            default=1,
            help="Grid resolution for base rotation search, 1 ~ 30', 2 ~ 15', default 1"
        )

        self.add_argument(
            "--solvent", 
            action="store_true",
            help="Use solvent mask in grid search"
        )

        self.add_argument(
            "--shell", 
            action="store_true",
            help="Use shell slicing in correlation calculation"
        )

        self.add_argument(
            "--plddt2pseudoB", 
            action="store_true",
            help="Treat thermal column as plddt, convert to pseudo B factors"
        )

def set_logger(outdir):
    logger = logging.getLogger("MR logger")
    logger.setLevel(logging.DEBUG)

    formatter_console = logging.Formatter('%(asctime)s - %(message)s')
    formatter_file = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # File handler
    file_handler = logging.FileHandler(os.path.join(outdir, "mr.log"), mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter_file)

    # stream handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)  # You can set the desired log level for console output
    console_handler.setFormatter(formatter_console)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

def zscoredot_torch(Po, Pc):
    Po_mean = torch.mean(Po)
    Po_std = torch.std(Po)
    Po_zscored = (Po - Po_mean)/Po_std  
    Pc_mean = torch.mean(Pc, dim=-1, keepdims=True)
    Pc_std = torch.std(Pc, dim=-1, keepdims=True)
    Pc_zscored = (Pc - Pc_mean)/Pc_std
    zscore_dot = torch.mean(Po_zscored*Pc_zscored, dim=-1)
    return zscore_dot

def zscoredot_torch_shell(Po, Pc, assignment):
    zdots = []
    n_bins = len(np.unique(assignment))
    for i in range(n_bins):
        resol_bool = (assignment == i)
        zdot_i = zscoredot_torch(Po[resol_bool], Pc[:, resol_bool])
        zdots.append(zdot_i)
    zscoredots = torch.stack(zdots).mean(0)
    return zscoredots

# TODO: change the partition number per user's GPU size
def search_rotations(rot_matrix, dcp, st_rmcom, propose_com,  patterson_uvw_arr_frac, Pu_o, solvent=False):
    propose_rmcoms = torch.einsum("bij,aj->bai", rot_matrix, st_rmcom)
    propose_models = propose_rmcoms + propose_com
    dcp.calc_fprotein_batch(atoms_position_batch=propose_models, PARTITION=400)
    if solvent:
        dcp.calc_fsolvent_batch(PARTITION=100)
    Fmodel_batch = dcp.calc_ftotal_batch()
    Fc_batch = torch.abs(Fmodel_batch)
    Pu_c_batch = sfc.patterson.Patterson_torch_batch(patterson_uvw_arr_frac, 
                                                     Fc_batch, 
                                                     dcp.HKL_array, 
                                                     dcp.unit_cell.volume, 
                                                     sharpen=True, remove_origin=True, 
                                                     PARTITION_uvw=10000,
                                                     PARTITION_batch=200,
                                                     no_grad=True)
    zdot_score = dr.assert_numpy(zscoredot_torch(Pu_o, Pu_c_batch))
    return zdot_score

def search_rotations_shell(rot_matrix, dcp, st_rmcom, propose_com, patterson_uvw_arr_frac, Pu_o, assignment, solvent=False):
    propose_rmcoms = torch.einsum("bij,aj->bai", rot_matrix, st_rmcom)
    propose_models = propose_rmcoms + propose_com
    
    dcp.calc_fprotein_batch(atoms_position_batch=propose_models, PARTITION=400)
    if solvent:
        dcp.calc_fsolvent_batch(PARTITION=100)
    Fmodel_batch = dcp.calc_ftotal_batch()
    Fc_batch = torch.abs(Fmodel_batch)
    Pu_c_batch = sfc.patterson.Patterson_torch_batch(patterson_uvw_arr_frac, 
                                                     Fc_batch, 
                                                     dcp.HKL_array, 
                                                     dcp.unit_cell.volume, 
                                                     sharpen=True, remove_origin=True, 
                                                     PARTITION_uvw=10000,
                                                     PARTITION_batch=200,
                                                     no_grad=True)
    zdot_score = dr.assert_numpy(zscoredot_torch_shell(Pu_o, Pu_c_batch, assignment))
    return zdot_score

def search_com(coms, propose_rmcom, dcp, solvent=False):
    propose_models = propose_rmcom[None,...] + coms[:,None,:]
    dcp.calc_fprotein_batch(atoms_position_batch=propose_models, PARTITION=200)
    if solvent:
        dcp.calc_fsolvent_batch(PARTITION=100)
    Fmodel_batch = dcp.calc_ftotal_batch()
    Fc_batch = torch.abs(Fmodel_batch)
    zdot_score_F = dr.assert_numpy(zscoredot_torch(dcp.Fo, Fc_batch))
    return zdot_score_F

def MR_pipeline(pdb_path, 
                mtz_path, 
                outdir, 
                n_rot=3, 
                n_trans=4, 
                trans_basegrid=24, 
                rot_basegrid=1, 
                freeflag="FREE", 
                Fcolumn="FP", 
                SIGFcolumn="SIGFP", 
                testset_value=0, 
                plddt2pseudoB=False, 
                solvent=False,
                dmin=3.0,
                pts_min=10.0,
                pts_max=13.0,
                shell=False):

    if os.path.exists(outdir):
        print(f"Output dir exits: {outdir}", flush=True)
    else:
        os.makedirs(outdir)
        print(f"Output dir created: {outdir}", flush=True)

    # Set up logger
    logger = set_logger(outdir)

    logger.info("Initialization...")
    logger.info(f"PDB file: {pdb_path}")
    logger.info(f"MTZ file: {mtz_path}")
    logger.info(f"Fcolumn: {Fcolumn}, SIGFcolumn: {SIGFcolumn}, Freeflag: {freeflag}, testset: {testset_value}")
    
    pdb_name = os.path.basename(mtz_path).split(".")[0]

    # Initialize the dcp    
    dcp = sfc.SFcalculator(
        pdbmodel=pdb_path,
        mtzdata=mtz_path,
        expcolumns=[Fcolumn, SIGFcolumn],
        freeflag=freeflag,
        testset_value=testset_value,
        dmin=dmin
    )

    logger.info(f"SpaceGroup: {dcp.space_group.hm}")
    logger.info(f"UnitCell: {dcp.unit_cell.parameters}")
    st_rmcom = dcp.atom_pos_orth - torch.mean(dcp.atom_pos_orth, dim=0)
    if plddt2pseudoB:
        logger.info("Converting PLDDT to pseudo B factors...")
        dcp.atom_b_iso = dr.utils.plddt2pseudoB(dcp.atom_b_iso)
    
    # Find the best coarse packing score to initialize
    logger.info("="*30)
    logger.info("Packing Socre Stage for scales...")
    vdw_rad = sfc.utils.vdw_rad_tensor(dcp.atom_name)
    uc_grid_orth_tensor = sfc.utils.unitcell_grid_center(dcp.unit_cell, 
                                                         spacing=4.5, 
                                                         return_tensor=True,
                                                         device=dcp.device)
    asu_brick_lims = np.array(gemmi.find_asu_brick(dcp.space_group).size)/24.0
    xlim, ylim, zlim = asu_brick_lims
    u_list = np.linspace(0,xlim,8)
    v_list = np.linspace(0,ylim,8)
    w_list = np.linspace(0,zlim,8)
    uvw_array_frac = np.array(np.meshgrid(u_list, v_list, w_list)).T.reshape(-1,3)
    uvw_array_orth = torch.tensordot(torch.tensor(uvw_array_frac).to(dcp.frac2orth_tensor), 
                                     dcp.frac2orth_tensor.T, dims=1)
    ps_asu1_list = []
    for trans_vec in uvw_array_orth:
        occupancy_temp, crash_temp = sfc.packingscore.packingscore_voxelgrid_torch(st_rmcom+trans_vec, 
                                                                                   dcp.unit_cell, 
                                                                                   dcp.space_group,
                                                                                   vdw_rad, 
                                                                                   uc_grid_orth_tensor, 
                                                                                   CUTOFF=0.0001)
        ps_asu1_list.append([dr.assert_numpy(occupancy_temp), dr.assert_numpy(crash_temp)])
    ps_asu1_array = np.array(ps_asu1_list)
    w1 = np.argmin(ps_asu1_array[:,1])
    logger.debug(f"Candidate packing score: {ps_asu1_array[w1, 0]*100:.2f}%, clashing score: {ps_asu1_array[w1, 1]*100:.2f}%")
    round1_com = uvw_array_orth[w1]
    dcp.atom_pos_orth = st_rmcom + round1_com
    dcp.inspect_data()
    dcp.calc_fprotein()
    if solvent:
        dcp.calc_fsolvent()
    dcp.get_scales_adam()
    rfree_init, rwork_init = dcp.r_free.item(), dcp.r_work.item()

    # Rotation Search Stage
    logger.info("="*30)
    logger.info("Rotation Search Stage...")
    # TODO: Find an algorithmic way to determine the range of patterson vector, instead of hard coding 
    patterson_uvw_arr_frac = sfc.patterson.uvw_array_frac(dcp.unit_cell, pts_min, pts_max, 0.3)
    
    logger.info("Experimental Patterson function value...")
    Pu_o = sfc.patterson.Patterson_torch(patterson_uvw_arr_frac, 
                                         dcp.Fo, dcp.HKL_array, dcp.unit_cell.volume,
                                         sharpen=True, remove_origin=True,
                                         PARTITION=10000)
    patterson_uvw_arr_orth = dcp.frac2orth(dr.assert_tensor(patterson_uvw_arr_frac, arr_type=torch.float32))
    patterson_len = dr.assert_numpy(torch.norm(patterson_uvw_arr_orth, dim=-1))
    assignment, _ = sfc.utils.bin_by_logarithmic(1./patterson_len, bins=20, Nmin=len(patterson_len) // 24)
    torch.cuda.empty_cache()
    logger.debug(f"Memory peak: {torch.cuda.max_memory_allocated() / 10**9:.2f}G, Memory current: {torch.cuda.memory_allocated() / 10**9:.2f}G")
    
    # logger.info("Choosing a starting COM with Patterson Score...")
    # batch_model = st_rmcom + uvw_array_orth[:,None,:]
    # dcp.calc_fprotein_batch(atoms_position_batch=batch_model, PARTITION=200)
    # if solvent:
    #     dcp.calc_fsolvent_batch(PARTITION=100)
    # Fmodel_batch = dcp.calc_ftotal_batch()
    # Fc_batch = torch.abs(Fmodel_batch)
    # logger.debug(f"Memory peak: {torch.cuda.max_memory_allocated() / 10**9:.2f}G, Memory current: {torch.cuda.memory_allocated() / 10**9:.2f}G")
    # Pu_c_batch = sfc.patterson.Patterson_torch_batch(patterson_uvw_arr_frac, 
    #                                                  Fc_batch, 
    #                                                  dcp.HKL_array, 
    #                                                  dcp.unit_cell.volume, 
    #                                                  sharpen=True, remove_origin=True, 
    #                                                  PARTITION_uvw=10000,
    #                                                  PARTITION_batch=200,
    #                                                  no_grad=True)
    # zdot_asu1_tensor = zscoredot_torch(Pu_o, Pu_c_batch)
    # clash_cutoff = np.percentile(ps_asu1_array[:,1], 20)
    # ww = ps_asu1_array[:,1] < clash_cutoff
    # uvw_array_orth_goodps = uvw_array_orth[ww]
    # zdot_asu1_tensor_goodps = zdot_asu1_tensor[ww]
    # w2 = torch.argmax(zdot_asu1_tensor_goodps)
    # propose_com = uvw_array_orth_goodps[w2]
    # logger.info(f"Candidate patterson zdot score: {zdot_asu1_tensor_goodps[w2].item():.3f}")
    # logger.debug(f"Candidate packing score: {ps_asu1_array[ww][w2.item(), 0]*100:.2f}%, clashing score: {ps_asu1_array[ww][w2.item(), 1]*100:.2f}%")
   
    # torch.cuda.empty_cache()
    # logger.debug(f"Memory peak: {torch.cuda.max_memory_allocated() / 10**9:.2f}G, Memory current: {torch.cuda.memory_allocated() / 10**9:.2f}G")
    propose_com = round1_com
    for i in range(n_rot):
        logger.info(f"Round {i+1}/{n_rot} Rotational Search...")
        if i == 0:
            roundi_quats = dr.geometry.grid_SO3(rot_basegrid)
            roundi_matrix = dr.geometry.quaternions_to_SO3(roundi_quats)
        elif i == 1:
            roundi_quats, roundi_s2s1 = dr.geometry.getbestneighbors_base_SO3(
                -zdot_score_roundi,  # noqa: F821
                roundi_quats, 
                N=40,
                base_resol=rot_basegrid
            )
            roundi_matrix = dr.geometry.quaternions_to_SO3(roundi_quats)
        else:
            roundi_quats, roundi_s2s1 = dr.geometry.getbestneighbors_next_SO3(
                -zdot_score_roundi,  # noqa: F821
                roundi_quats,
                roundi_s2s1,
                curr_res=i+rot_basegrid-1,
                N=40
            )
            roundi_matrix = dr.geometry.quaternions_to_SO3(roundi_quats)
        
        if shell:
            zdot_score_roundi = search_rotations_shell(
                roundi_matrix,
                dcp,
                st_rmcom,
                propose_com,
                patterson_uvw_arr_frac,
                Pu_o,
                assignment,
                solvent,
            )
        else:
            zdot_score_roundi = search_rotations(
                roundi_matrix,
                dcp,
                st_rmcom,
                propose_com,
                patterson_uvw_arr_frac,
                Pu_o,
                solvent,
            )
        logger.info(f"Best patterson zdot score {np.max(zdot_score_roundi): .3f}, top 20%: {np.percentile(zdot_score_roundi, 20): .3f}, worst: {np.min(zdot_score_roundi): .3f}")
    bestR_index = np.argmax(zdot_score_roundi)
    propose_R = roundi_matrix[bestR_index].T
    propose_rmcom = torch.matmul(st_rmcom, propose_R)
    dcp.atom_pos_orth = propose_rmcom
    dcp.savePDB(os.path.join(outdir, pdb_name + "_MR_stage1.pdb"))
    logger.info(f"Rotation Search Finished, model saved at {os.path.join(outdir, pdb_name + '_MR_stage1.pdb')}")
    
    torch.cuda.empty_cache()
    logger.debug(f"Memory peak: {torch.cuda.max_memory_allocated() / 10**9:.2f}G, Memory current: {torch.cuda.memory_allocated() / 10**9:.2f}G")

    logger.info("="*30)
    polar_axis = sfc.get_polar_axis(dcp.space_group)
    logger.info(f"Polar Axis: {polar_axis}")
    if polar_axis == [0, 1, 2]:
        logger.info(f"All Axes are polar, no need to do translational search, choose [0,5, 0.5, 0.5]...")
        best_com_frac = np.array([0.5, 0.5, 0.5])
        best_com = torch.matmul(torch.tensor(best_com_frac).to(dcp.frac2orth_tensor), dcp.frac2orth_tensor.T)
    else:
        for j in range(n_trans):
            logger.info(f"Round {j+1}/{n_trans} Translational Search...")
            if j == 0:
                logger.debug(f"Translation search base grid: {trans_basegrid}")
                u_list = np.linspace(0,1,trans_basegrid)
                v_list = np.linspace(0,1,trans_basegrid)
                w_list = np.linspace(0,1,trans_basegrid)
                if polar_axis is not None:
                    if 0 in polar_axis:
                        u_list = np.array([0.5])
                    if 1 in polar_axis:
                        v_list = np.array([0.5])
                    if 2 in polar_axis:
                        w_list = np.array([0.5])
                roundj_uvw_frac = np.array(np.meshgrid(u_list, v_list, w_list)).T.reshape(-1,3)
            else:
                roundj_uvw_frac = dr.geometry.getbestneighbours_cartesian(
                    -zfj,  # noqa: F821
                    roundj_uvw_frac,
                    basegrid=trans_basegrid,
                    curr_res=j,
                    N=40,
                    drop_duplicates=True,
                    polar_axis=polar_axis,
                )
            roundj_uvw_orth = torch.matmul(torch.tensor(roundj_uvw_frac).to(dcp.frac2orth_tensor), dcp.frac2orth_tensor.T)
            zfj = search_com(roundj_uvw_orth, propose_rmcom, dcp, solvent=solvent)
            logger.info(f"Best SF zdot score {np.max(zfj): .3f}, top 20%: {np.percentile(zfj, 20): .3f}, worst: {np.min(zfj): .3f}")
        best_com = roundj_uvw_orth[np.argmax(zfj)]
    replaced_model = propose_rmcom + best_com
    dcp.atom_pos_orth = replaced_model
    dcp.inspect_data()
    dcp.calc_fprotein()
    if solvent:
        dcp.calc_fsolvent()
    dcp.get_scales_adam()
    rfree_stage2, rwork_stage2 = dcp.r_free.item(), dcp.r_work.item()
    dcp.savePDB(os.path.join(outdir, pdb_name + "_MR_stage2.pdb"))
    logger.info(f"Grid Search Finished, model saved at {os.path.join(outdir, pdb_name + '_MR_stage2.pdb')}")
    logger.info(f"Rwork: {rwork_init:.3f} -> {rwork_stage2:.3f}, Rfree: {rfree_init:.3f} -> {rfree_stage2:.3f}")
    logger.debug(f"Memory peak: {torch.cuda.max_memory_allocated() / 10**9:.2f}G, Memory current: {torch.cuda.memory_allocated() / 10**9:.2f}G")

    logger.info("="*30)
    logger.info("Gradient Descent Stage...")
    
    _, rbred_model, _, _ = dr.utils.rbr_quat_adam(replaced_model, dcp, lr=0.001, n_steps=100, loss_track=[], solvent=True, verbose=False)
    rbr_rmsd = dr.utils.rmsd(rbred_model, replaced_model)
    dcp.atom_pos_orth = rbred_model
    dcp.inspect_data()
    dcp.calc_fprotein()
    if solvent:
        dcp.calc_fsolvent()
    dcp.get_scales_adam()
    rwork_stage3, rfree_stage3 = dcp.r_free.item(), dcp.r_work.item()
    dcp.savePDB(os.path.join(outdir, pdb_name + "_MR_stage3.pdb"))
    logger.info(f"Gradient Descent Finished, model saved at {os.path.join(outdir, pdb_name + '_MR_stage3.pdb')}")
    logger.info(f"RMSD: {rbr_rmsd:.3f} A. Rwork: {rwork_stage2:.3f} -> {rwork_stage3:.3f}, Rfree: {rfree_stage2:.3f} -> {rfree_stage3:.3f}")

    logger.info("Finished...")

def main():
    args = ArgumentParser().parse_args()
    MR_pipeline(
        args.pdb,
        args.mtz,
        args.outdir,
        n_rot=args.n_rot,
        n_trans=args.n_trans,
        trans_basegrid=args.trans_basegrid,
        rot_basegrid=args.rot_basegrid,
        freeflag=args.freeflag,
        Fcolumn=args.Fcolumn,
        SIGFcolumn=args.SigFcolumn,
        testset_value=args.testset_value,
        plddt2pseudoB=args.plddt2pseudoB,
        solvent=args.solvent,
        dmin=args.dmin,
        pts_min=args.pts_min,
        pts_max=args.pts_max,
        shell=args.shell
    )
    with open(os.path.join(args.outdir, "config.json"), 'w') as file:
        json.dump(vars(args), file, indent=4)




    








