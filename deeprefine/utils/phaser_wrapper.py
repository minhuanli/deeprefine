"""
phaser_wrapper.py modifed from Dennis's MathcMaps
https://github.com/rs-station/matchmaps/blob/main/src/matchmaps/_phenix_utils.py#L267
"""

import shutil
import reciprocalspaceship as rs
import subprocess
from pathlib import Path


def phaser_wrapper(
    mtzfile: Path,
    pdb: Path,
    output_dir: Path,
    off_labels="FP,SIGFP",
    eff=None,
    verbose=False,
):
    """
    Handle simple phaser run from the command line

    mtzfile: Path
        path to mtz file
    pdb: Path
        path to pdb file
    output_dir: Path
        path to output directory
    off_labels: str
        labels for the mtz file, like "FP,SIGFP"
    eff: str
        path to eff file, default is None
    verbose: bool
        whether to print the output of the command line
    """

    if shutil.which("phenix.phaser") is None:
        raise OSError(
            "Cannot find executable, phenix.phaser. Please set up your phenix environment."
        )

    if eff is None:
        eff_contents = """
phaser {
  mode = ANO CCA EP_AUTO *MR_AUTO MR_FRF MR_FTF MR_PAK MR_RNP NMAXYZ SCEDS
  hklin = mtz_input
  labin = labels
  model = pdb_input
  model_identity = 100
  component_copies = 1
  search_copies = 1
  chain_type = *protein dna rna
  crystal_symmetry {
    unit_cell = cell_parameters
    space_group = sg
  }
  keywords {
    general {
      root = '''nickname'''
      title = '''phaser_MR'''
      mute = None
      xyzout = True
      xyzout_ensemble = True
      hklout = True
      jobs = 6
    }
  }
}
        """
    else:
        raise NotImplementedError("Custom phaser specifications are not yet supported")

    nickname = f"{mtzfile.name.removesuffix('.mtz')}_phased_with_{pdb.name.removesuffix('.pdb')}"

    similar_files = list(output_dir.glob(f"{nickname}_*"))
    if len(similar_files) == 0:
        nickname += "_0"
    else:
        nums = []
        for s in similar_files:
            try:
                nums.append(int(str(s).split("_")[-1].split(".")[0]))
            except ValueError:
                pass
        nickname += f"_{max(nums)+1}"

    mtz = rs.read_mtz(str(mtzfile))
    cell_string = f"{mtz.cell.a} {mtz.cell.b} {mtz.cell.c} {mtz.cell.alpha} {mtz.cell.beta} {mtz.cell.gamma}"
    sg = mtz.spacegroup.short_name()

    eff = output_dir / f"params_{nickname}.eff"

    params = {
        "sg": sg,
        "cell_parameters": cell_string,
        "pdb_input": str(pdb),
        "mtz_input": str(mtzfile),
        "nickname": str(output_dir / nickname),
        "labels": off_labels,  # should be prepackaged as a string
    }

    for key, value in params.items():
        eff_contents = eff_contents.replace(key, value)

    with open(eff, "w") as file:
        file.write(eff_contents)

    subprocess.run(
        f"phenix.phaser {eff}",
        shell=True,
        capture_output=(not verbose),
    )

    return output_dir / nickname