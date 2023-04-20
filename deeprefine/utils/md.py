from openmm.app import ForceField, PDBFile, Modeller, Simulation, StateDataReporter
from openmm import app, LangevinIntegrator
from openmm import unit
from mdtraj.reporters import HDF5Reporter


def run_md(
    startPDB,
    prefix,
    temperature=290,
    eqtime=1.0,
    prtime=100.0,
    reportstep=0.1,
    recordstep=0.01,
    implicit_solvent=True,
    stepsize=0.002,
    verbose=True,
):
    """
    Run MD simulation starting from a PDB file, with implicit or explcit solvent model.
    Parameters
    ----------
    startPDB: str
        The path of the starting model PDB file

    prefix: str
        The prefix of the output file name.

    temperature: int or float, default 290
        The temperature of the simulation system, in Kelvin

    eqtime: int/float, default 1
        Time to simiulate for equillibration before production, in ns
        The number of steps will be calculated by eqtime/stepsize

    prtime: int/float, default 100
        Time to simulate for production, in ns
        The number of steps will be calculated by prtime/stepsize

    reportstep: int/float, default 0.1
        Time between two state report, in ns

    recordstep: int/float, default 0.01
        Time between two state recordings in production run, in ns

    implicit_solvent: binary, default True
        Using implicit solvent model or explicit solvent model. It will determine the forcefield in use.
        Note: If you have HOH in your starting PDB file, you must use explicit solvent model, or it will give error.

    stepsize: int/float, default 0.002
        The time of each MD step in unit of picosecond.

    verbose: boolean, default True

    Returns
    -------
    None
    """
    if implicit_solvent:
        ff = ForceField("amber99sb.xml", "amber99_obc.xml")
        friction = 91
    else:
        ff = ForceField("amber99sb.xml", "amber/tip3p_standard.xml")
        friction = 1

    pdb = PDBFile(startPDB)
    modeller = Modeller(pdb.topology, pdb.positions)
    system = ff.createSystem(
        modeller.topology,
        removeCMMotion=False,
        nonbondedMethod=app.CutoffNonPeriodic,
        nonbondedCutoff=1.0 * unit.nanometers,
        constraints=None,
        rigidWater=True,
    )

    # 2) Setup MD simulation, minimize, and equilibrate
    integrator = LangevinIntegrator(
        temperature * unit.kelvin,
        friction / unit.picosecond,
        stepsize * unit.picoseconds,
    )
    simulation = Simulation(modeller.topology, system, integrator)
    platform = simulation.context.getPlatform()
    if verbose:
        print(f"Simulation running with {platform.getName()}", flush=True)
    simulation.context.setPositions(modeller.positions)
    if verbose:
        print("Minimizing... ", end="", flush=True)
    simulation.minimizeEnergy()
    if verbose:
        print("done", flush=True)

    # Reallocate the velocity, another "equillibriate time"
    simulation.context.setVelocitiesToTemperature(temperature * unit.kelvin)
    if verbose:
        from sys import stdout

        statereporter = StateDataReporter(
            stdout,
            int(1000 * reportstep / stepsize),
            step=True,
            time=True,
            volume=True,
            totalEnergy=True,
            temperature=True,
            elapsedTime=True,
        )
    else:
        statereporter = StateDataReporter(
            prefix + "_equi_report.csv",
            int(1000 * reportstep / stepsize),
            step=True,
            time=True,
            volume=True,
            totalEnergy=True,
            temperature=True,
            elapsedTime=True,
        )
    simulation.reporters.append(statereporter)
    simulation.step(int(1000 * eqtime / stepsize))
    if verbose:
        print("Equilibrating... done", flush=True)
    simulation.reporters = []

    # save the checkpoint, everything is inside
    simulation.saveCheckpoint(prefix + "_equilibrated.chkpt")

    # 3) Run production simulation
    if verbose:
        print("Production run... ", end="", flush=True)
    # simulation.loadCheckpoint("implicit_equilibrated.chkpt")
    simulation.context.setVelocitiesToTemperature(temperature * unit.kelvin)
    statereporter = StateDataReporter(
        prefix + "_report.csv",
        int(1000 * reportstep / stepsize),
        step=True,
        time=True,
        volume=True,
        totalEnergy=True,
        temperature=True,
        elapsedTime=True,
    )
    trajreporter = HDF5Reporter(prefix + "_traj.h5", int(1000 * recordstep / stepsize))
    simulation.reporters.append(statereporter)
    simulation.reporters.append(trajreporter)
    simulation.step(int(1000 * prtime / stepsize))
    trajreporter.close()
    if verbose:
        print("MD running done", flush=True)
