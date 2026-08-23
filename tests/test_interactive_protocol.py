import io

import numpy as np
from ase import Atoms

from vasp_interactive import VaspInteractive


class FakeProcess:
    def __init__(self, stdout_text):
        self.stdin = io.StringIO()
        self.stdout = io.StringIO(stdout_text)

    def poll(self):
        return None


def _calculator(process):
    calc = VaspInteractive(txt=None, allow_mpi_pause=False)
    calc.process = process
    calc.sort = [1, 2, 0]
    calc.symbol_count = [("H", 2), ("O", 1)]
    return calc


def test_vasp64_prompt_selects_poscar_protocol():
    atoms = Atoms(
        "OH2",
        positions=[(0, 0, 0), (0.7, 0.7, 0), (0.7, 0, 0.7)],
        cell=[8, 8, 8],
        pbc=True,
    )
    calc = _calculator(
        FakeProcess(
            " vasp.6.6.0 01Jan00\n"
            "POSITIONS AND LATTICE: reading from stdin\n"
        )
    )
    try:
        calc._run(atoms, out=None, require_cell_stdin=False)
        assert calc._interactive_protocol == "poscar"
        assert calc.version == "6.6.0"
    finally:
        calc.process = None


def test_vasp64_writer_sends_full_poscar_record():
    atoms = Atoms(
        "OH2",
        positions=[(0, 0, 0), (0.7, 0.7, 0), (0.7, 0, 0.7)],
        cell=[8, 8, 8],
        pbc=True,
    )
    process = FakeProcess("POSITIONS AND LATTICE: read from stdin\n")
    calc = _calculator(process)
    calc._interactive_protocol = "poscar"
    try:
        calc._write_atoms_stdin(atoms, out=None, require_cell_stdin=True)
        lines = process.stdin.getvalue().splitlines()
        assert lines[:2] == ["VaspInteractive", "1.0"]
        vectors = np.asarray([np.fromstring(line, sep=" ") for line in lines[2:5]])
        assert vectors.shape == (3, 3)
        assert lines[5] == "2 1"
        assert lines[6] == "Direct"
        assert len(lines[7:]) == 3
    finally:
        calc.process = None


def test_legacy_writer_handles_separate_lattice_prompt():
    atoms = Atoms("H2", positions=[(0, 0, 0), (0.1, 0.1, 0)], cell=[8, 8, 8], pbc=True)
    process = FakeProcess(
        "POSITIONS: read from stdin\n"
        "LATTICE: reading from stdin\n"
        "New direct lattice vectors\n"
        "LATTICE: read from stdin\n"
    )
    calc = _calculator(process)
    calc.sort = [0, 1]
    calc.symbol_count = [("H", 2)]
    calc._interactive_protocol = "positions"
    try:
        calc._write_atoms_stdin(atoms, out=None, require_cell_stdin=True)
        lines = process.stdin.getvalue().splitlines()
        assert len(lines) == 5
        assert len(np.fromstring(lines[-1], sep=" ")) == 3
    finally:
        calc.process = None


def test_legacy_writer_preserves_non_lattice_stdout():
    atoms = Atoms("H2", positions=[(0, 0, 0), (0.1, 0.1, 0)], cell=[8, 8, 8], pbc=True)
    process = FakeProcess(
        "0.0 0.0 0.0\n"
        "0.1 0.1 0.0\n"
        "POSITIONS: read from stdin\n"
        "next electronic-step line\n"
    )
    calc = _calculator(process)
    calc.sort = [0, 1]
    calc.symbol_count = [("H", 2)]
    calc._interactive_protocol = "positions"
    try:
        calc._write_atoms_stdin(atoms, out=None, require_cell_stdin=False)
        assert calc._pending_stdout == "next electronic-step line\n"
    finally:
        calc.process = None
