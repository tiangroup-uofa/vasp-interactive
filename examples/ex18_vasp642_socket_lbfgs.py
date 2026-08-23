#!/usr/bin/env python3
"""VASP 6.4.2 fixed-cell socket example with ASE LBFGS.

Required environment variables:
    VASP_COMMAND          command that launches a VASP 6.4.2 executable
    VASP_PP_PATH          licensed VASP pseudopotential root understood by ASE
    VPI_EXAMPLE_WORKDIR   empty-or-new parent directory for this example's output

This is deliberately a fixed-cell example.  The compatibility branch supports
VASP 6.4.2's fixed-cell interactive protocol (ISIF=2); it is not a variable
cell relaxation example.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.calculators.socketio import SocketIOCalculator
from ase.optimize import LBFGS
from vasp_interactive import VaspInteractive


def make_rattled_si() -> Atoms:
    """Return an eight-atom cubic diamond-Si cell without ase.build."""
    lattice = 5.43
    fractional_positions = np.array(
        [
            (0.00, 0.00, 0.00),
            (0.00, 0.50, 0.50),
            (0.50, 0.00, 0.50),
            (0.50, 0.50, 0.00),
            (0.25, 0.25, 0.25),
            (0.25, 0.75, 0.75),
            (0.75, 0.25, 0.75),
            (0.75, 0.75, 0.25),
        ]
    )
    atoms = Atoms(
        "Si8",
        positions=fractional_positions * lattice,
        cell=np.eye(3) * lattice,
        pbc=True,
    )
    atoms.rattle(stdev=0.03, seed=20260728)
    return atoms


def main() -> None:
    for name in ("VASP_COMMAND", "VASP_PP_PATH", "VPI_EXAMPLE_WORKDIR"):
        if not os.environ.get(name):
            raise SystemExit(f"{name} must be set; see docs/vasp642_socket_fixed_cell.md")

    workdir = Path(os.environ["VPI_EXAMPLE_WORKDIR"])
    workdir.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix="vasp642-socket-lbfgs-", dir=workdir))
    atoms = make_rattled_si()
    initial_positions = atoms.get_positions().copy()

    vpi = VaspInteractive(
        directory=run_dir / "vasp",
        txt="vasp-interactive.out",
        xc="pbe",
        encut=300,
        ediff=1e-5,
        kpts=(2, 2, 2),
        gamma=True,
        ismear=0,
        sigma=0.05,
        lreal=False,
        algo="Normal",
        isym=0,
        isif=2,
        nsw=20,
        lwave=False,
        lcharg=False,
        allow_mpi_pause=False,
        allow_restart_process=False,
    )

    socket_name = f"vasp642_socket_{os.getpid()}"
    with SocketIOCalculator(
        calc=vpi,
        unixsocket=socket_name,
        log=str(run_dir / "socket-server.log"),
    ) as calculator:
        atoms.calc = calculator
        initial_energy = float(atoms.get_potential_energy())
        initial_max_force = float(np.abs(atoms.get_forces()).max())
        optimizer = LBFGS(
            atoms,
            logfile=str(run_dir / "lbfgs.log"),
            trajectory=str(run_dir / "lbfgs.traj"),
        )
        converged = bool(optimizer.run(fmax=0.10, steps=3))
        final_energy = float(atoms.get_potential_energy())
        final_max_force = float(np.abs(atoms.get_forces()).max())

    summary = {
        "run_dir": str(run_dir),
        "socket": "Unix domain socket",
        "fixed_cell": True,
        "optimizer": "ASE LBFGS",
        "optimizer_steps": int(optimizer.nsteps),
        "converged": converged,
        "initial_energy_eV": initial_energy,
        "final_energy_eV": final_energy,
        "initial_max_force_eV_per_A": initial_max_force,
        "final_max_force_eV_per_A": final_max_force,
        "max_displacement_A": float(
            np.linalg.norm(atoms.get_positions() - initial_positions, axis=1).max()
        ),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if not np.isfinite(
        [initial_energy, final_energy, initial_max_force, final_max_force]
    ).all():
        raise RuntimeError("socket calculator returned a non-finite energy or force")
    if not converged:
        raise RuntimeError("example did not converge within its three-step acceptance limit")
    print("VASP642_SOCKET_LBFGS_PASS")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
