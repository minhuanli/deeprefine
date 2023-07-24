"""
Run MD with a simple setting

Examples
--------
Run MD for 100ns after equilibration for 1ns, at 290K
    > dr.runmd xxx.pdb -o "xxx_implicit" -t 290 -e 1 -p 100

Run MD with 1fs stepsize
    > dr.runmd xxx.pdb -o "xxx_1fs" -s 0.001

Run MD with explicit water model
    > dr.runmd xxx.pdb -o "xxx_explicit" -i False

Run MD for 1ns without equilibration, and record every 0.001 ns
    > dr.runmd xxx.pdb -o "xxx_short" -e 0 -p 1 -s 0.001
"""
import argparse
import deeprefine as dr


class ArgumentParser(argparse.ArgumentParser):
    def __init__(self):
        super().__init__(
            formatter_class=argparse.RawTextHelpFormatter, description=__doc__
        )

        # Required arguments
        self.add_argument("pdb", help="The path of the starting model PDB file")

        self.add_argument(
            "-o", "--output", type=str, help="The prefix of the output file name"
        )

        # Optional arguments
        self.add_argument(
            "-t",
            "--temperature",
            default=290,
            type=float,
            help="Temperature of the simulation system, in Kelvin, default 290",
        )

        self.add_argument(
            "-e",
            "--eqtime",
            default=1,
            type=float,
            help="Time to simiulate for equillibration before production, in ns, default 1",
        )

        self.add_argument(
            "-p",
            "--prtime",
            default=100,
            type=float,
            help="Time to simulate for production, in ns, default 100",
        )

        self.add_argument(
            "--reportstep",
            default=0.1,
            type=float,
            help="Time between two state report, in ns, default 0.1",
        )

        self.add_argument(
            "--recordstep",
            default=0.01,
            type=float,
            help="Time between two state recordings in production run, in ns, default 0.01",
        )

        self.add_argument(
            "-s",
            "--stepsize",
            type=float,
            default=0.002,
            help="The time of each MD step, in unit of picosecond, default 0.002",
        )

        self.add_argument(
            "-i",
            "--implicitsolvent",
            default=True,
            help="Using implicit solvent model or explicit solvent model. \
                  It will determine the forcefield in use. \
                  Note: If you have HOH in your starting PDB file, \
                  you must use explicit solvent model, or it will give error.",
        )

        self.add_argument("-v", "--verbose", action="store_true")


def main():
    args = ArgumentParser().parse_args()
    dr.utils.run_md(
        args.pdb,
        args.output,
        temperature=args.temperature,
        eqtime=args.eqtime,
        prtime=args.prtime,
        reportstep=args.reportstep,
        recordstep=args.recordstep,
        stepsize=args.stepsize,
        implicit_solvent=args.implicitsolvent,
        verbose=args.verbose,
    )
