"""
CLI for m3gnet
"""

import argparse
import sys
import logging
import os

from pymatgen.core.structure import Structure
import tensorflow as tf
from m3gnet.models import Relaxer, MolecularDynamics

logging.captureWarnings(True)
tf.get_logger().setLevel(logging.ERROR)


def relax_structure(args):
    """
    Handle view commands.

    :param args: Args from command.
    """

    for fn in args.infile:
        s = Structure.from_file(fn)

        if args.verbose:
            print("Starting structure")
            print(s)
            print("Relaxing...")
        relaxer = Relaxer()
        relax_results = relaxer.relax(s)
        final_structure = relax_results["final_structure"]

        if args.suffix:
            basename, ext = os.path.splitext(fn)
            outfn = f"{basename}{args.suffix}{ext}"
            final_structure.to(filename=outfn)
            print(f"Structure written to {outfn}!")
        elif args.outfile is not None:
            final_structure.to(filename=args.outfile)
            print(f"Structure written to {args.outfile}!")
        else:
            print("Final structure")
            print(final_structure)

    return 0


def run_md(args):
    """
    Handle view commands.

    :param args: Args from command.
    """
    for fn in args.infile:
        s = Structure.from_file(fn)

        if args.verbose:
            print("Starting structure")
            print(s)
            print("Running MD...")
        md = MolecularDynamics(
            atoms=s,
            temperature=args.temp,
            ensemble=args.ensemble,
            timestep=args.timestep,
            trajectory=args.trajectory,
            logfile=args.logfile,
            loginterval=args.loginterval,
        )

        md.run(steps=args.nsteps)

    return 0


def main():
    """
    Handle main.
    """
    parser = argparse.ArgumentParser(
        description="""
    This script works based on several sub-commands with their own options. To see the options for the
    sub-commands, type "m3g sub-command -h".""",
        epilog="""Author: M3Gnet""",
    )

    subparsers = parser.add_subparsers()

    p_relax = subparsers.add_parser("relax", help="Relax crystal structures.")

    p_relax.add_argument(
        "-i",
        "--infile",
        dest="infile",
        nargs="+",
        required=True,
        help="Input file containing structure. Common structures support by pmg.Structure.from_file method.",
    )

    p_relax.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        default=False,
        action="store_true",
        help="Verbose output.",
    )

    groups = p_relax.add_mutually_exclusive_group(required=False)
    groups.add_argument(
        "-s",
        "--suffix",
        dest="suffix",
        help="Suffix to be added to input file names for relaxed structures. E.g., _relax.",
    )

    groups.add_argument(
        "-o",
        "--outfile",
        dest="outfile",
        help="Output filename.",
    )

    p_relax.set_defaults(func=relax_structure)

    p_md = subparsers.add_parser("md", help="Run molecular dynamics.")

    p_md.add_argument(
        "-i",
        "--infile",
        dest="infile",
        nargs="+",
        required=True,
        help="Input file containing structure. Common structures support by pmg.Structure.from_file method.",
    )

    p_md.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        default=False,
        action="store_true",
        help="Verbose output.",
    )

    p_md.add_argument(
        "-t",
        "--temp",
        dest="temp",
        required=True,
        type=float,
        help="Temperature of the simulation.",
    )

    p_md.add_argument(
        "-e",
        "--ensemble",
        dest="ensemble",
        required=True,
        type=str,
        default="nvt",
        help="Ensemble of the simulation.",
    )

    p_md.add_argument(
        "-dt",
        "--timestep",
        dest="timestep",
        required=True,
        type=float,
        default=2.0,
        help="Timestep of the simulation in fs.",
    )

    p_md.add_argument(
        "--traj",
        dest="trajectory",
        required=False,
        type=str,
        default="md.traj",
        help="Trajectory file of the simulation.",
    )

    p_md.add_argument(
        "--log",
        dest="logfile",
        required=False,
        type=str,
        default="md.log",
        help="Log file of the simulation.",
    )

    p_md.add_argument(
        "--loginterval",
        dest="loginterval",
        required=False,
        type=int,
        default=100,
        help="Log interval of the simulation.",
    )

    p_md.add_argument(
        "-n",
        "--nsteps",
        dest="nsteps",
        required=True,
        type=int,
        default=1000,
        help="Number of steps of the simulation.",
    )

    p_md.set_defaults(func=run_md)

    args = parser.parse_args()

    try:
        getattr(args, "func")
    except AttributeError:
        parser.print_help()
        sys.exit(-1)
    return args.func(args)


if __name__ == "__main__":
    main()
