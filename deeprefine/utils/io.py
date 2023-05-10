import mdtraj
import numpy as np

def prep_pdb(filename, outfile, pH=7.0, keepHOH=False, missingresidue="missres.log"):
    """
    Prepare a PDB file ready for the openmm simulation
    Parameters
    ----------
    filename: str
        path to the input PDB file

    outfile: str
        path of the output PDB file name

    pH: float, default 7.0
        pH value of the system, used to add missing hydrogen

    keepHOH: binary, default False
        When remove the Heterogens in PDB file, keep HOH or not.

    missingresidue: None, or str path to record file
        If not None, missing residue record will be saved to the file
    Returns
    -------
    None
    """
    from pdbfixer import PDBFixer
    fixer = PDBFixer(filename=filename)
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(pH)
    fixer.removeHeterogens(keepHOH)

    from openmm.app import PDBFile

    PDBFile.writeFile(fixer.topology, fixer.positions, open(outfile, "w"))
    print("Preparation Done! Saved at:", outfile, flush=True)

    if missingresidue is not None:
        import pickle

        with open(missingresidue, "wb") as f:
            pickle.dump(fixer.missingResidues, f, protocol=0)
        print("Missing Residue record has been saved at:", missingresidue, flush=True)

    # The following block will make sure the fixed PDB has
    # the correct spacegroup as the origin PDB, or it will always be P1
    with open(filename, "r") as file_a:
        lines_a = file_a.readlines()
    # Read the contents of file B
    with open(outfile, "r") as file_b:
        lines_b = file_b.readlines()

    # Find the index of the CRYST1 line in origin file
    cryst1_index_a = None
    for i, line in enumerate(lines_a):
        if line.startswith("CRYST1"):
            cryst1_index_a = i
            break
    # Find the index of the CRYST1 line in output file
    cryst1_index_b = None
    for i, line in enumerate(lines_b):
        if line.startswith("CRYST1"):
            cryst1_index_b = i
            break
    # Replace the CRYST1 line in file B with the CRYST1 line from file A
    if (cryst1_index_a is not None) and (cryst1_index_b is not None):
        lines_b[cryst1_index_b] = lines_a[cryst1_index_a]
    # Write the updated contents of file B
    with open(outfile, "w") as file_b:
        file_b.writelines(lines_b)


def align_md(traj_file, shuffle=True, ref_pdb=None):
    """
    traj_file: str
        File path to the md trajectory file

    shuffle: binary, default True
        Whether or not to shuffle the data after alignment

    ref_pdb: None or str
        File path to the reference PDB file. If not set, the alignment reference will be the
        first model in the traj file.

    Return
    ------
    sim_coordinate, topology
    """
    traj = mdtraj.load_hdf5(traj_file)
    sim_x = traj.xyz
    topology = traj.topology
    nframes = sim_x.shape[0]
    if ref_pdb is not None:
        ref = mdtraj.load(ref_pdb)
        assert list(ref.topology.atoms) == list(
            topology.atoms
        ), "Make sure your reference PDB file has the same topology with the traj file!"
    else:
        ref = traj[0]
    traj = traj.superpose(ref, atom_indices=topology.select("backbone"))
    sim_x = traj.xyz.reshape((nframes, -1))
    if shuffle:
        np.random.shuffle(sim_x)
    return sim_x, topology


def remove_hydrogens(old_traj_path, new_traj_path):
    """
    old_traj_path: str path to the old trajectory file, with hydrogens
    new_traj_path: str path you want to save the new trajectory file
    """
    old_traj = mdtraj.load(old_traj_path)
    old_topology = old_traj.topology
    noHydrogen_indices = old_topology.select_atom_indices("heavy")
    new_traj = old_traj.atom_slice(noHydrogen_indices)
    new_traj.save_hdf5(new_traj_path)


def fix_missingresidues(trajectory, record):
    """
    trajectory: mdtraj.Trajectory
    record: dict, record for the missing residues
    """
    count = 0
    conditions = []
    # Conditions for select out those added residues
    for chainid, insertid in record:
        startid = count + insertid
        endid = startid + len(record[chainid, insertid]) - 1
        count += len(record[chainid, insertid])
        conditions.append(f"(chainid == {chainid}) and (resid {startid} to {endid})")

    # The atom_indices for all added missing residues
    topology = trajectory.topology
    atom_slices = []
    for condition in conditions:
        atom_slices += topology.select(condition).tolist()

    # Do set difference for the atom indices of all remaining residues
    residuesleft = np.setdiff1d(np.arange(topology.n_atoms), np.array(atom_slices))
    traj_new = trajectory.atom_slice(residuesleft)
    return traj_new

def save_samples_to_pdb(samples, mdtraj_topology, filename=None, topology_fn=None):
    '''
    Save generated samples as a pdb file.
    `samples`: array, (Nsamples, n_atoms*n_dim)
    `mdtraj_topology`: an MDTraj Topology object of the molecular system
    `filename=None`: str, output filename with extension (all MDTraj compatible formats)
    '''
    import mdtraj as md
    trajectory = md.Trajectory(
        samples.reshape(-1, mdtraj_topology.n_atoms, 3), mdtraj_topology)
    if filename.split('.')[-1] == 'pdb':
        trajectory.save_pdb(filename)
    else:
        trajectory.save(filename)