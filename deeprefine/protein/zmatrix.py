import numpy as np

# template for each residue type
basis_Zs = {}

basis_Zs["ALA"] = [
    ["H", "N", "CA", "C"],
    ["O", "C", "CA", "N"],
    ["HA", "CA", "C", "O"],
    ["CB", "CA", "C", "O"],
    ["HB1", "CB", "CA", "N"],
    ["HB2", "CB", "CA", "N"],
    ["HB3", "CB", "CA", "N"],
]

basis_Zs["LEU"] = [
    ["H", "N", "CA", "C"],
    ["O", "C", "CA", "N"],
    ["HA", "CA", "C", "O"],
    ["CB", "CA", "C", "O"],
    ["CG", "CB", "CA", "N"],
    ["CD1", "CG", "CB", "CA"],
    ["CD2", "CG", "CB", "CA"],
    ["HB2", "CB", "CA", "N"],
    ["HB3", "CB", "CA", "N"],
    ["HG", "CG", "CB", "CA"],
    ["HD11", "CD1", "CG", "CB"],
    ["HD21", "CD2", "CG", "CB"],
    ["HD12", "CD1", "CG", "CB"],
    ["HD13", "CD1", "CG", "CB"],
    ["HD22", "CD2", "CG", "CB"],
    ["HD23", "CD2", "CG", "CB"],
]

basis_Zs["ILE"] = [
    ["H", "N", "CA", "C"],
    ["O", "C", "CA", "N"],
    ["HA", "CA", "C", "O"],
    ["CB", "CA", "C", "O"],
    ["CG1", "CB", "CA", "N"],
    ["CG2", "CB", "CA", "N"],
    ["CD1", "CG1", "CB", "CA"],
    ["HB", "CB", "CA", "N"],
    ["HG12", "CG1", "CB", "CA"],
    ["HG13", "CG1", "CB", "CA"],
    ["HD11", "CD1", "CG1", "CB"],
    ["HD12", "CD1", "CG1", "CB"],
    ["HD13", "CD1", "CG1", "CB"],
    ["HG21", "CG2", "CB", "CA"],
    ["HG22", "CG2", "CB", "CA"],
    ["HG23", "CG2", "CB", "CA"],
]

basis_Zs["CYS"] = [
    ["H", "N", "CA", "C"],
    ["O", "C", "CA", "N"],
    ["HA", "CA", "C", "O"],
    ["CB", "CA", "C", "O"],
    ["SG", "CB", "CA", "N"],
    ["HB2", "CB", "CA", "N"],
    ["HB3", "CB", "CA", "N"],
    ["HG", "SG", "CB", "CA"],
]

basis_Zs["HIS"] = [
    ["H", "N", "CA", "C"],
    ["O", "C", "CA", "N"],
    ["HA", "CA", "C", "O"],
    ["CB", "CA", "C", "O"],
    ["CG", "CB", "CA", "N"],
    ["ND1", "CG", "CB", "CA"],
    ["HD1", "ND1", "CG", "CB"],
    ["CD2", "CG", "CB", "CA"],
    ["CE1", "ND1", "CG", "CB"],
    ["NE2", "CD2", "CG", "CB"],
    ["HB2", "CB", "CA", "C"],
    ["HB3", "CB", "CA", "C"],
    ["HD2", "CD2", "CG", "CB"],
    ["HE1", "CE1", "ND1", "CG"],
    ["HE2", "NE2", "CD2", "CG"]
]

basis_Zs["ASP"] = [
    ["H", "N", "CA", "C"],
    ["O", "C", "CA", "N"],
    ["HA", "CA", "C", "O"],
    ["CB", "CA", "C", "O"],
    ["CG", "CB", "CA", "N"],
    ["OD1", "CG", "CB", "CA"],
    ["OD2", "CG", "CB", "CA"],
    ["HB2", "CB", "CA", "N"],
    ["HB3", "CB", "CA", "N"],
]

basis_Zs["ASN"] = [
    ["H", "N", "CA", "C"],
    ["O", "C", "CA", "N"],
    ["HA", "CA", "C", "O"],
    ["CB", "CA", "C", "O"],
    ["CG", "CB", "CA", "N"],
    ["OD1", "CG", "CB", "CA"],
    ["ND2", "CG", "CB", "CA"],
    ["HB2", "CB", "CA", "N"],
    ["HB3", "CB", "CA", "N"],
    ["HD21", "ND2", "CG", "CB"],
    ["HD22", "ND2", "CG", "CB"],
]

basis_Zs["GLN"] = [
    ["H", "N", "CA", "C"],
    ["O", "C", "CA", "N"],
    ["HA", "CA", "C", "O"],
    ["CB", "CA", "C", "O"],
    ["CG", "CB", "CA", "N"],
    ["CD", "CG", "CB", "CA"],
    ["OE1", "CD", "CG", "CB"],
    ["NE2", "CD", "CG", "CB"],
    ["HB2", "CB", "CA", "N"],
    ["HB3", "CB", "CA", "N"],
    ["HG2", "CG", "CB", "CA"],
    ["HG3", "CG", "CB", "CA"],
    ["HE21", "NE2", "CD", "CG"],
    ["HE22", "NE2", "CD", "CG"],
]

basis_Zs["GLU"] = [
    ["H", "N", "CA", "C"],
    ["O", "C", "CA", "N"],
    ["HA", "CA", "C", "O"],
    ["CB", "CA", "C", "O"],
    ["CG", "CB", "CA", "N"],
    ["CD", "CG", "CB", "CA"],
    ["OE1", "CD", "CG", "CB"],
    ["OE2", "CD", "CG", "CB"],
    ["HB2", "CB", "CA", "N"],
    ["HB3", "CB", "CA", "N"],
    ["HG2", "CG", "CB", "CA"],
    ["HG3", "CG", "CB", "CA"],
]

basis_Zs["GLY"] = [
    ["H", "N", "CA", "C"],
    ["O", "C", "CA", "N"],
    ["HA2", "CA", "C", "O"],
    ["HA3", "CA", "C", "O"],
]

basis_Zs["TRP"] = [
    ["H", "N", "CA", "C"],
    ["O", "C", "CA", "N"],
    ["HA", "CA", "C", "O"],
    ["CB", "CA", "C", "O"],
    ["HB2", "CB", "CA", "C"],
    ["HB3", "CB", "CA", "C"],
    ["CG", "CB", "CA", "N"],
    ["CD1", "CG", "CB", "CA"],
    ["HD1", "CD1", "CG", "CB"],
    ["CD2", "CG", "CB", "CA"],
    ["NE1", "CD1", "CG", "CB"],
    ["HE1", "NE1", "CD1", "CG"],
    ["CE2", "NE1", "CD1", "CG"],
    ["CE3", "CD2", "CG", "CB"],
    ["HE3", "CE3", "CD2", "CG"],
    ["CZ2", "CE2", "NE1", "CD1"],
    ["HZ2", "CZ2", "CE2", "NE1"],
    ["CZ3", "CE3", "CD2", "CG"],
    ["HZ3", "CZ3", "CE3", "CD2"],
    ["CH2", "CZ2", "CE2", "NE1"],
    ["HH2", "CH2", "CZ2", "CE2"],
]

basis_Zs["TYR"] = [
    ["H", "N", "CA", "C"],
    ["O", "C", "CA", "N"],
    ["HA", "CA", "C", "O"],
    ["CB", "CA", "C", "O"],
    ["CG", "CB", "CA", "N"],
    ["CD1", "CG", "CB", "CA"],
    ["CD2", "CG", "CB", "CA"],
    ["CE1", "CD1", "CG", "CB"],
    ["CE2", "CD2", "CG", "CB"],
    ["CZ", "CE1", "CD1", "CG"],
    ["OH", "CZ", "CE1", "CD1"],
    ["HB2", "CB", "CA", "N"],
    ["HB3", "CB", "CA", "N"],
    ["HD1", "CD1", "CG", "CB"],
    ["HD2", "CD2", "CG", "CB"],
    ["HE1", "CE1", "CD1", "CG"],
    ["HE2", "CE2", "CD2", "CG"],
    ["HH", "OH", "CZ", "CE1"],
]

basis_Zs["SER"] = [
    ["H", "N", "CA", "C"],
    ["O", "C", "CA", "N"],
    ["HA", "CA", "C", "O"],
    ["CB", "CA", "C", "O"],
    ["OG", "CB", "CA", "N"],
    ["HB2", "CB", "CA", "N"],
    ["HB3", "CB", "CA", "N"],
    ["HG", "OG", "CB", "CA"],
]

basis_Zs["PRO"] = [
    ["O", "C", "CA", "N"],
    ["HA", "CA", "C", "O"],
    ["CB", "CA", "C", "O"],
    ["CG", "CB", "CA", "N"],
    ["CD", "CG", "CB", "CA"],
    ["HB2", "CB", "CA", "N"],
    ["HB3", "CB", "CA", "N"],
    ["HG2", "CG", "CB", "CA"],
    ["HG3", "CG", "CB", "CA"],
    ["HD2", "CD", "CG", "CB"],
    ["HD3", "CD", "CG", "CB"],
]

basis_Zs["ARG"] = [
    ["H", "N", "CA", "C"],
    ["O", "C", "CA", "N"],
    ["HA", "CA", "C", "O"],
    ["CB", "CA", "C", "O"],
    ["CG", "CB", "CA", "N"],
    ["CD", "CG", "CB", "CA"],
    ["NE", "CD", "CG", "CB"],
    ["CZ", "NE", "CD", "CG"],
    ["NH1", "CZ", "NE", "CD"],
    ["NH2", "CZ", "NE", "CD"],
    ["HB2", "CB", "CA", "N"],
    ["HB3", "CB", "CA", "N"],
    ["HG2", "CG", "CB", "CA"],
    ["HG3", "CG", "CB", "CA"],
    ["HD2", "CD", "CG", "CB"],
    ["HD3", "CD", "CG", "CB"],
    ["HE", "NE", "CD", "CG"],
    ["HH11", "NH1", "CZ", "NE"],
    ["HH12", "NH1", "CZ", "NE"],
    ["HH21", "NH2", "CZ", "NE"],
    ["HH22", "NH2", "CZ", "NE"],
]

basis_Zs["LYS"] = [
    ["H", "N", "CA", "C"],
    ["O", "C", "CA", "N"],
    ["HA", "CA", "C", "O"],
    ["CB", "CA", "C", "O"],
    ["CG", "CB", "CA", "N"],
    ["CD", "CG", "CB", "CA"],
    ["CE", "CD", "CG", "CB"],
    ["NZ", "CE", "CD", "CG"],
    ["HB2", "CB", "CA", "N"],
    ["HB3", "CB", "CA", "N"],
    ["HG2", "CG", "CB", "CA"],
    ["HG3", "CG", "CB", "CA"],
    ["HD2", "CD", "CG", "CB"],
    ["HD3", "CD", "CG", "CB"],
    ["HE2", "CE", "CD", "CG"],
    ["HE3", "CE", "CD", "CG"],
    ["HZ1", "NZ", "CE", "CD"],
    ["HZ2", "NZ", "CE", "CD"],
    ["HZ3", "NZ", "CE", "CD"],
]

basis_Zs["MET"] = [
    ["H", "N", "CA", "C"],
    ["O", "C", "CA", "N"],
    ["HA", "CA", "C", "O"],
    ["CB", "CA", "C", "O"],
    ["CG", "CB", "CA", "N"],
    ["SD", "CG", "CB", "CA"],
    ["CE", "SD", "CG", "CB"],
    ["HB2", "CB", "CA", "N"],
    ["HB3", "CB", "CA", "N"],
    ["HG2", "CG", "CB", "CA"],
    ["HG3", "CG", "CB", "CA"],
    ["HE1", "CE", "SD", "CG"],
    ["HE2", "CE", "SD", "CG"],
    ["HE3", "CE", "SD", "CG"],
]

basis_Zs["THR"] = [
    ["H", "N", "CA", "C"],
    ["O", "C", "CA", "N"],
    ["HA", "CA", "C", "O"],
    ["CB", "CA", "C", "O"],
    ["OG1", "CB", "CA", "N"],
    ["CG2", "CB", "CA", "N"],
    ["HB", "CB", "CA", "N"],
    ["HG1", "OG1", "CB", "CA"],
    ["HG21", "CG2", "CB", "CA"],
    ["HG22", "CG2", "CB", "CA"],
    ["HG23", "CG2", "CB", "CA"],
]

basis_Zs["VAL"] = [
    ["H", "N", "CA", "C"],
    ["O", "C", "CA", "N"],
    ["HA", "CA", "C", "O"],
    ["CB", "CA", "C", "O"],
    ["CG1", "CB", "CA", "N"],
    ["CG2", "CB", "CA", "N"],
    ["HB", "CB", "CA", "N"],
    ["HG11", "CG1", "CB", "CA"],
    ["HG12", "CG1", "CB", "CA"],
    ["HG13", "CG1", "CB", "CA"],
    ["HG21", "CG2", "CB", "CA"],
    ["HG22", "CG2", "CB", "CA"],
    ["HG23", "CG2", "CB", "CA"],
]

basis_Zs["PHE"] = [
    ["H", "N", "CA", "C"],
    ["O", "C", "CA", "N"],
    ["HA", "CA", "C", "O"],
    ["CB", "CA", "C", "O"],
    ["CG", "CB", "CA", "N"],
    ["CD1", "CG", "CB", "CA"],
    ["CD2", "CG", "CB", "CA"],
    ["CE1", "CD1", "CG", "CB"],
    ["CE2", "CD2", "CG", "CB"],
    ["CZ", "CE1", "CD1", "CG"],
    ["HB2", "CB", "CA", "N"],
    ["HB3", "CB", "CA", "N"],
    ["HD1", "CD1", "CG", "CB"],
    ["HD2", "CD2", "CG", "CB"],
    ["HE1", "CE1", "CD1", "CG"],
    ["HE2", "CE2", "CD2", "CG"],
    ["HZ", "CZ", "CE1", "CD1"],
]


def mdtraj2Z(mdtraj_topology, cartesian=None):
    """Return Catesian and IC indices

    Parameters
    ----------
    mdtraj_topology : MDtraj.topology

    cartesian : MDTraj selection string, or None
        Selection of root atoms, which are represented by cartesian atoms. By default None

    Returns
    -------
    zmatrix, and root atoms index if provided
    """
    Z = []
    notIC = []
    # residue index for start and end of each chain
    counts = 0
    starts = []
    ends = []
    for i in mdtraj_topology.chains:
        starts.append(counts + 0)
        ends.append(counts + i.n_residues - 1)
        counts += i.n_residues

    if cartesian != None:
        notIC = mdtraj_topology.select(cartesian)

    for i, res in enumerate(mdtraj_topology.residues):
        nterm = i in starts
        cterm = i in ends

        resatoms = {a.name: a.index for a in res.atoms}
        resname = res.name
        for entry in basis_Zs[resname]:  # template entry:
            try:
                if resatoms[entry[0]] not in notIC:
                    Z.append([resatoms[_e] for _e in entry])
            except:
                continue
        if nterm:
            # set two additional N-term protons
            try:
                if resatoms["H2"] not in notIC:
                    Z.append(
                        [resatoms["H2"], resatoms["N"], resatoms["CA"], resatoms["C"]]
                    )
                if resatoms["H3"] not in notIC:
                    Z.append(
                        [resatoms["H3"], resatoms["N"], resatoms["CA"], resatoms["C"]]
                    )
            except:
                pass
        if cterm:
            # place OXT
            try:
                if resatoms["OXT"] not in notIC:
                    Z.append(
                        [resatoms["OXT"], resatoms["C"], resatoms["CA"], resatoms["N"]]
                    )
            except:
                pass

    if cartesian != None:
        return Z, notIC
    else:
        return Z


def get_indices(top, cart_sele_str="name CA name C name N"):
    """Returns Cartesian and IC indices

    Parameters
    ----------
    top: mdtraj.topology

    cart_sele_str: str
        mdtraj selection string for atoms represented by cartesian

    Returns
    -------
    cart : array
        Cartesian atom selection index
    Z : array
        Z index matrix

    """
    cart = top.select(cart_sele_str)
    Z_ = np.array(mdtraj2Z(top))
    # Check if the list is correct
    assert (
        np.sort(np.concatenate((cart, Z_[:, 0]))) == np.arange(top.n_atoms)
    ).all(), "Something wrong with the atom list!"
    return np.array(cart), Z_
