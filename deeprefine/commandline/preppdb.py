"""
Prepare PDB file for MD simulation with PDBFixer

Examples
--------
Fix the PDB file and save the missing residue record
    > dr.preppdb xxx.pdb -o xxx_fixed.pdb -m missres.log

Fix the PDB and keep the HOH for explicit solvent model
    > dr.preppdb xxx.pdb -o xxx_fixed.pdb --keepHOH

Fix the PDB at a specified pH value
    > dr.preppdb xxx.pdb -o xxx_fixed.pdb --ph 5.0
"""
import argparse
import deeprefine as dr

class ArgumentParser(argparse.ArgumentParser):
    def __init__(self):
        super().__init__(
            formatter_class=argparse.RawTextHelpFormatter,
            description=__doc__
        )

        # Required arguments
        self.add_argument(
            "pdb",
            help='PDB file to be fixed'
        )

        self.add_argument(
            "-o",
            "--output",
            type=str,
            help="Path of the output PDB file name"
        )

        # Optional arguments
        self.add_argument(
            "--ph",
            default=7.0,
            type=float,
            help="pH value of the system, used to add missing hydrogen"
        )

        self.add_argument(
            "--keepHOH",
            action="store_true",
            help="Keep the HOH for explicit solvent model"
        )

        self.add_argument(
            "-m",
            "--misslog",
            default=None,
            help="If not None, missing residue record will be saved to the file"
        )

def main():
    args = ArgumentParser().parse_args()
    dr.utils.prep_pdb(
        args.pdb,
        args.output,
        args.ph,
        args.keepHOH,
        args.misslog
    )
